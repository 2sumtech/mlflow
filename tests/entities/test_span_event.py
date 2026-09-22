import time

import pydantic
import pytest

import mlflow
from mlflow.entities import SpanEvent
from mlflow.entities.span_event import MAX_PARTIAL_COMPLETION_LENGTH
from mlflow.exceptions import MlflowException


def test_default_timestamp_is_in_nanoseconds():
    # Record the time before and after creating the event
    before_ns = int(time.time() * 1e9)
    event = SpanEvent(name="test_event")
    after_ns = int(time.time() * 1e9)

    # The event's timestamp should be between before and after in nanoseconds
    assert before_ns <= event.timestamp <= after_ns


def test_from_exception():
    exception = MlflowException("test")
    span_event = SpanEvent.from_exception(exception)
    assert span_event.name == "exception"
    assert span_event.attributes["exception.message"] == "test"
    assert span_event.attributes["exception.type"] == "MlflowException"
    assert span_event.attributes["exception.stacktrace"] is not None


@pytest.mark.parametrize(
    "event_attrs",
    [
        {},
        {"simple": "string"},
        {"number": 42, "float": 3.14, "bool": True},
        {"list": [1, 2, 3], "dict": {"nested": "value"}},
    ],
)
def test_span_event_to_otel_proto_conversion(event_attrs):
    # Create span event
    event = SpanEvent(
        name="test_event",
        timestamp=1234567890,
        attributes=event_attrs,
    )

    # Convert to OTel proto
    otel_proto_event = event.to_otel_proto()

    # Verify fields
    assert otel_proto_event.name == "test_event"
    assert otel_proto_event.time_unix_nano == 1234567890

    # Verify attributes
    from mlflow.tracing.utils.otlp import _decode_otel_proto_anyvalue

    decoded_attrs = {}
    for attr in otel_proto_event.attributes:
        decoded_attrs[attr.key] = _decode_otel_proto_anyvalue(attr.value)

    assert decoded_attrs == event_attrs


class _Message(pydantic.BaseModel):
    role: str
    content: str


class _Choice(pydantic.BaseModel):
    index: int
    finish_reason: str
    message: _Message


class _Completion(pydantic.BaseModel):
    """Mirrors the shape of ``openai.types.chat.ParsedChatCompletion``."""

    id: str
    model: str
    choices: list[_Choice]


class LengthFinishReasonError(Exception):
    """Mirrors ``openai.LengthFinishReasonError``, which is not a test dependency here."""

    def __init__(self, *, completion):
        super().__init__("Could not parse response content as the length limit was reached")
        self.completion = completion


class _MockOpenAIClient:
    """A client whose structured-output call always stops with ``finish_reason="length"``."""

    def __init__(self, content):
        self._content = content

    def parse(self, **kwargs):
        completion = _Completion(
            id="chatcmpl-123",
            model="gpt-4o-mini",
            choices=[
                _Choice(
                    index=0,
                    finish_reason="length",
                    message=_Message(role="assistant", content=self._content),
                )
            ],
        )
        raise LengthFinishReasonError(completion=completion)


def test_from_exception_appends_partial_completion():
    client = _MockOpenAIClient('{"name": "Alice", "age":')
    with pytest.raises(LengthFinishReasonError, match="length limit"):
        client.parse(model="gpt-4o-mini")

    try:
        client.parse(model="gpt-4o-mini")
    except LengthFinishReasonError as e:
        span_event = SpanEvent.from_exception(e)

    message = span_event.attributes["exception.message"]
    assert message.startswith("Could not parse response content as the length limit was reached")
    assert '{\\"name\\": \\"Alice\\", \\"age\\":' in message
    assert '"finish_reason":"length"' in message
    assert span_event.attributes["exception.type"] == "LengthFinishReasonError"


def test_from_exception_truncates_long_partial_completion():
    client = _MockOpenAIClient("a" * (MAX_PARTIAL_COMPLETION_LENGTH * 2))
    try:
        client.parse(model="gpt-4o-mini")
    except LengthFinishReasonError as e:
        span_event = SpanEvent.from_exception(e)

    message = span_event.attributes["exception.message"]
    partial = message.split("\nPartial completion: ", 1)[1]
    assert partial.endswith(f"... (truncated to {MAX_PARTIAL_COMPLETION_LENGTH} characters)")
    assert len(partial) == MAX_PARTIAL_COMPLETION_LENGTH + len(
        f"... (truncated to {MAX_PARTIAL_COMPLETION_LENGTH} characters)"
    )


def test_from_exception_ignores_unrelated_exception_with_completion_attribute():
    exception = MlflowException("boom")
    exception.completion = "should not be surfaced"
    span_event = SpanEvent.from_exception(exception)
    assert span_event.attributes["exception.message"] == "boom"


def test_from_exception_tolerates_unserializable_completion():
    class _Unserializable:
        def model_dump_json(self):
            raise ValueError("nope")

    exception = LengthFinishReasonError(completion=_Unserializable())
    span_event = SpanEvent.from_exception(exception)
    assert "Partial completion" not in span_event.attributes["exception.message"]


def test_record_exception_surfaces_partial_completion_on_span():
    client = _MockOpenAIClient('{"name": "Alice", "age":')
    with mlflow.start_span(name="parse") as span:
        try:
            client.parse(model="gpt-4o-mini")
        except LengthFinishReasonError as e:
            span.record_exception(e)

    trace = mlflow.get_trace(span.trace_id)
    event = trace.data.spans[0].events[0]
    assert event.name == "exception"
    assert "Partial completion: " in event.attributes["exception.message"]
