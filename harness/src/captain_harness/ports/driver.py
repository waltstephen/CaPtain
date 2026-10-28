"""Model invocation boundary for external agent runtimes.

The protocol owns public evidence, execution and budgets. A driver owns model
scheduling. Throwing transport exceptions back preserves host recovery/finalization.
"""
from dataclasses import dataclass
from typing import Generator, TypeVar
from .model import ModelBackend, ModelRequest, ModelResponse, CompactionRequest

@dataclass(frozen=True, slots=True)
class ModelInvocation:
    request: ModelRequest | CompactionRequest
    previous_response_id: str | None
    kind: str
    independent: bool = False

T = TypeVar("T")

def drive_protocol(protocol: Generator[ModelInvocation, ModelResponse, T], model: ModelBackend) -> T:
    try:
        event = next(protocol)
        while True:
            try:
                if isinstance(event.request, CompactionRequest):
                    response = model.compact(event.request)
                else:
                    response = model.respond(event.request,
                        previous_response_id=event.previous_response_id)
            except Exception as exc:
                # Preserve the scheduling boundary even for exceptions not wrapped
                # by a provider backend. This is metadata, never a replay request.
                try:
                    exc.model_invocation={"kind":event.kind,"independent":event.independent,
                        "previous_response_id":event.previous_response_id,
                        "verification_stage":getattr(event.request,"metadata",{}).get("verification_stage")}
                except (AttributeError,TypeError):
                    pass
                event = protocol.throw(exc)
            else:
                event = protocol.send(response)
    except StopIteration as done:
        return done.value
    finally:
        protocol.close()
