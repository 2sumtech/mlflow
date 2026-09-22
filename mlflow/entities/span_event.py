import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

from opentelemetry.proto.trace.v1.trace_pb2 import Span as OTelProtoSpan
from opentelemetry.util.types import AttributeValue

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.tracing.utils.otlp import _set_otel_proto_anyvalue

# Some LLM SDKs raise an exception that carries the partial completion generated before the
# call was cut short, and the partial output is lost unless we surface it. The canonical case
# is ``openai.LengthFinishReasonError``, raised when a structured-output request stops with
# ``finish_reason == "length"``; the tokens already generated are only reachable through the
# exception's ``completion`` field. Matching on the class name keeps this dependency-free.
# See https://github.com/mlflow/mlflow/issues/16232
_PARTIAL_COMPLETION_EXCEPTIONS = frozenset({"LengthFinishReasonError"})

# Cap the stringified completion so the exception message stays readable in the UI.
MAX_PARTIAL_COMPLETION_LENGTH = 2000


def _stringify_partial_completion(exception: BaseException) -> str | None:
    """
    Return the partial completion carried by ``exception``, or None if it carries none.

    The result is truncated to ``MAX_PARTIAL_COMPLETION_LENGTH`` characters.
    """
    if _PARTIAL_COMPLETION_EXCEPTIONS.isdisjoint(c.__name__ for c in type(exception).__mro__):
        return None

    completion = getattr(exception, "completion", None)
    if completion is None:
        return None

    try:
        # Pydantic models (the OpenAI SDK's completion objects) serialize to JSON.
        serialized = (
            completion.model_dump_json()
            if hasattr(completion, "model_dump_json")
            else str(completion)
        )
    except Exception:
        return None

    if len(serialized) > MAX_PARTIAL_COMPLETION_LENGTH:
        return (
            f"{serialized[:MAX_PARTIAL_COMPLETION_LENGTH]}... (truncated to "
            f"{MAX_PARTIAL_COMPLETION_LENGTH} characters)"
        )
    return serialized


@dataclass
class SpanEvent(_MlflowObject):
    """
    An event that records a specific occurrences or moments in time
    during a span, such as an exception being thrown. Compatible with OpenTelemetry.

    Args:
        name: Name of the event.
        timestamp:  The exact time the event occurred, measured in nanoseconds.
            If not provided, the current time will be used.
        attributes: A collection of key-value pairs representing detailed
            attributes of the event, such as the exception stack trace.
            Attributes value must be one of ``[str, int, float, bool, bytes]``
            or a sequence of these types.
    """

    name: str
    # Use current time if not provided. We need to use default factory otherwise
    # the default value will be fixed to the build time of the class.
    timestamp: int = field(default_factory=lambda: int(time.time() * 1e9))
    attributes: dict[str, AttributeValue] = field(default_factory=dict)

    @classmethod
    def from_exception(cls, exception: Exception) -> "SpanEvent":
        "Create a span event from an exception."

        stack_trace = cls._get_stacktrace(exception)
        message = str(exception)
        # OpenTelemetry defines only three attributes for an exception event, so the partial
        # completion is appended to the message rather than stored as its own attribute.
        if (partial_completion := _stringify_partial_completion(exception)) is not None:
            message = f"{message}\nPartial completion: {partial_completion}"
        return cls(
            name="exception",
            attributes={
                "exception.message": message,
                "exception.type": exception.__class__.__name__,
                "exception.stacktrace": stack_trace,
            },
        )

    @staticmethod
    def _get_stacktrace(error: BaseException) -> str:
        """Get the stacktrace of the parent error."""
        msg = repr(error)
        try:
            tb = traceback.format_exception(error)
            return "".join(tb).strip()
        except Exception:
            return msg

    def json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "attributes": json.dumps(self.attributes, cls=CustomEncoder)
            if self.attributes
            else None,
        }

    def to_otel_proto(self) -> OTelProtoSpan.Event:
        """
        Convert to OpenTelemetry protobuf event format for OTLP export.
        This is an internal method used for logging spans via OTel protocol.

        Returns:
            An OpenTelemetry protobuf Span.Event message.
        """
        otel_event = OTelProtoSpan.Event()
        otel_event.name = self.name
        otel_event.time_unix_nano = self.timestamp

        for key, value in self.attributes.items():
            attr = otel_event.attributes.add()
            attr.key = key
            _set_otel_proto_anyvalue(attr.value, value)

        return otel_event


class CustomEncoder(json.JSONEncoder):
    """
    Custom encoder to handle json serialization.
    """

    def default(self, o: object) -> str:
        try:
            # JSONEncoder.default always raises TypeError; cast to reflect the
            # effective return type of this override.
            return cast(str, super().default(o))
        except TypeError:
            # convert datetime to string format by default
            if isinstance(o, datetime):
                return o.isoformat()
            # convert object direct to string to avoid error in serialization
            return str(o)
