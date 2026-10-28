from __future__ import annotations

import re
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Mapping, TypeVar

from jsonschema import Draft202012Validator, SchemaError
from .schema_constraints import public_argument_constraints, inferred_literal_type

T = TypeVar("T")
_PUBLIC_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
# CAPTAIN_FAST_LOOP=1: Draft202012Validator.check_schema is a pure function of the schema; remember schemas already found valid
# (keyed by repr, which distinguishes list/tuple and key order) instead of re-checking them on every CapabilitySpec build.
_CHECKED_SCHEMAS: set[str] = set()


def _fast_schema_cache() -> bool:
    import os
    return os.environ.get("CAPTAIN_FAST_LOOP", "") == "1"


class FailureCategory(str, Enum):
    NONE = "none"
    SCHEMA_ERROR = "schema_error"
    INVALID_ARGUMENT = "invalid_argument"
    NOT_EXECUTED = "not_executed"
    CONTROLLER_REJECTED = "controller_rejected"
    TIMEOUT = "timeout"
    BACKEND_ERROR = "backend_error"
    TOOL_ERROR = "tool_error"
    IK_FAILURE = "ik_failure"
    PERCEPTION_UNCERTAIN = "perception_uncertain"
    SAFETY_REJECTED = "safety_rejected"
    STAGE_BUDGET_EXHAUSTED = "stage_budget_exhausted"


class SideEffect(str, Enum):
    READ_ONLY = "read_only"
    SENSOR_CAPTURE = "sensor_capture"
    PERCEPTION = "perception"
    COMPUTE = "compute"
    ROBOT_MOTION = "robot_motion"
    ACTUATOR = "actuator"


class ControlSemantic(str, Enum):
    TIME_ADVANCE = "time_advance"
    CARTESIAN_MOTION = "cartesian_motion"
    MOTION = "motion"
    ATTACHMENT_ATTEMPT = "attachment_attempt"
    ATTACHMENT_PROBE = "attachment_probe"
    ATTACHMENT_RELEASE = "attachment_release"
    ATTACHMENT_ASSERTION = "attachment_assertion"
    ATTACHMENT_CONTINUITY = "attachment_continuity"
    PLACEMENT_ASSERTION = "placement_assertion"
    PLACEMENT_ESTIMATE = "placement_estimate"
    SAFE_RELEASE_AUTHORIZATION = "safe_release_authorization"
    COMPLETION_ASSERTION = "completion_assertion"
    GRASP_PLANNING = "grasp_planning"


class UnknownCapabilityError(LookupError):
    pass


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    name: str
    description: str
    arguments_schema: Mapping[str, Any]
    result_schema: Mapping[str, Any]
    units: Mapping[str, str]
    coordinate_frames: Mapping[str, str]
    timeout_s: float
    side_effect: SideEffect
    argument_order: tuple[str, ...] = ()
    return_type: str = "object"
    examples: tuple[str, ...] = ()
    control_semantic: ControlSemantic | None = None
    real_robot_equivalent: str = "generic runtime capability"

    def __post_init__(self) -> None:
        if not _PUBLIC_NAME.fullmatch(self.name) or self.name.startswith("_"):
            raise ValueError(f"invalid public capability name: {self.name!r}")
        if not self.description.strip():
            raise ValueError("capability description must not be empty")
        if self.timeout_s <= 0:
            raise ValueError("capability timeout_s must be positive")
        for label, schema in (
            ("arguments_schema", self.arguments_schema),
            ("result_schema", self.result_schema),
        ):
            key = None
            if _fast_schema_cache():
                key = repr(schema)
                if key in _CHECKED_SCHEMAS:
                    continue
            try:
                Draft202012Validator.check_schema(dict(schema))
            except SchemaError as exc:
                raise ValueError(f"invalid {label}: {exc.message}") from exc
            if key is not None:
                _CHECKED_SCHEMAS.add(key)
        if self.arguments_schema.get("type") != "object":
            raise ValueError("arguments_schema must describe a JSON object")
        properties = self.arguments_schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise ValueError("arguments_schema properties must be an object")
        if len(self.argument_order) != len(set(self.argument_order)):
            raise ValueError("argument_order contains duplicate names")
        unknown_arguments = set(self.argument_order).difference(properties)
        if unknown_arguments:
            raise ValueError(
                f"argument_order names are missing from schema: {sorted(unknown_arguments)!r}"
            )
        if not self.return_type.strip():
            raise ValueError("return_type must not be empty")
        if not self.real_robot_equivalent.strip():
            raise ValueError("real_robot_equivalent must not be empty")
        if not all(isinstance(example, str) and example.strip() for example in self.examples):
            raise ValueError("examples must contain non-empty strings")
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in self.units.items()):
            raise ValueError("units must map strings to strings")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.coordinate_frames.items()
        ):
            raise ValueError("coordinate_frames must map strings to strings")


@dataclass(frozen=True, slots=True)
class CapabilityManifest:
    schema_version: str
    provider: str
    capabilities: tuple[CapabilitySpec, ...]
    usage_instructions: str = ""

    def __post_init__(self) -> None:
        if not self.schema_version.strip() or not self.provider.strip():
            raise ValueError("manifest schema_version and provider must not be empty")
        names = [spec.name for spec in self.capabilities]
        if len(names) != len(set(names)):
            raise ValueError("manifest contains duplicate capability names")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.capabilities)

    def require(self, name: str) -> CapabilitySpec:
        for spec in self.capabilities:
            if spec.name == name:
                return spec
        raise UnknownCapabilityError(f"capability {name!r} is not declared by {self.provider}")

    def render_python_interface(self) -> str:
        """Render manifest metadata as model-facing Python signatures and docs."""
        blocks: list[str] = [
            "# Every call returns CapabilityResult: .success, .executed, .failure_category, .message, .value.\n"
            "# Optional .diagnostics retains deadline scope, service attempts and measured partial execution.\n"
            "# A timed-out operation may have partially executed; inspect its receipt before retrying."
        ]
        if self.usage_instructions:
            blocks.append('\n'.join('# ' + line for line in self.usage_instructions.splitlines()))
        for spec in self.capabilities:
            properties = spec.arguments_schema.get("properties", {})
            required = set(spec.arguments_schema.get("required", ()))
            parameters = []
            optional_seen = False
            keyword_only = False
            for name in spec.argument_order:
                schema = properties[name]
                annotation = _python_annotation(schema)
                parameter = f"{name}: {annotation}"
                if name in required and optional_seen and not keyword_only:
                    parameters.append('*')
                    keyword_only = True
                if name not in required:
                    parameter += f" = {schema.get('default')!r}"
                    optional_seen = True
                parameters.append(parameter)
            signature = ", ".join(parameters)
            lines = [
                f"def {spec.name}({signature}) -> CapabilityResult[{spec.return_type}]:",
                f'    """{spec.description}',
            ]
            if spec.units:
                lines.append(f"    Units: {dict(spec.units)!r}")
            if spec.coordinate_frames:
                lines.append(f"    Frames: {dict(spec.coordinate_frames)!r}")
            lines.append("    Argument constraints: " + json.dumps(
                public_argument_constraints(spec.arguments_schema), sort_keys=True))
            if spec.examples:
                lines.append("    Examples:")
                lines.extend(f"        {example}" for example in spec.examples)
            if spec.control_semantic is not None:
                lines.append(
                    f"    Control semantic: {spec.control_semantic.value}"
                )
            lines.append(
                "    Real-robot equivalent: "
                f"{spec.real_robot_equivalent}"
            )
            lines.append(
                f"    Returns .value: {_python_schema_shape(spec.result_schema)}"
            )
            lines.extend(('    """', "    ..."))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


@dataclass(frozen=True, slots=True)
class CapabilityResult(Generic[T]):
    success: bool
    executed: bool
    failure_category: FailureCategory
    message: str
    value: T | None = None
    attempt_identity: Mapping[str, object] | None = None
    diagnostics: Mapping[str, object] | None = None
    # Adapter-issued exact attempt association; never inferred from a target label.
    attempt_evidence: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.success and not self.executed:
            raise ValueError("successful result must be executed")
        if self.success != (self.failure_category is FailureCategory.NONE):
            raise ValueError("success and failure_category disagree")
        if self.executed and self.failure_category is FailureCategory.NOT_EXECUTED:
            raise ValueError("not_executed result cannot be executed")

    @classmethod
    def ok(cls, value: T, message: str = "") -> "CapabilityResult[T]":
        return cls(True, True, FailureCategory.NONE, message, value)

    @classmethod
    def failed(
        cls,
        category: FailureCategory,
        message: str,
        *,
        executed: bool = False,
        value: T | None = None,
    ) -> "CapabilityResult[T]":
        if category is FailureCategory.NONE:
            raise ValueError("failed result requires a failure category")
        return cls(False, executed, category, message, value)

    @classmethod
    def invalid_argument(cls, message: str) -> "CapabilityResult[T]":
        return cls.failed(FailureCategory.INVALID_ARGUMENT, message)

    @classmethod
    def schema_error(cls, message: str) -> "CapabilityResult[T]":
        return cls.failed(FailureCategory.SCHEMA_ERROR, message)


def _python_annotation(schema: Mapping[str, Any]) -> str:
    any_of = schema.get("anyOf", schema.get("oneOf"))
    if isinstance(any_of, list):
        values = [
            _python_annotation(item)
            for item in any_of
            if isinstance(item, Mapping)
        ]
        return " | ".join(dict.fromkeys(values)) or "object"
    schema_type = inferred_literal_type(schema)
    if isinstance(schema_type, list):
        values = [
            _python_annotation({**schema, "type": item})
            for item in schema_type
            if isinstance(item, str)
        ]
        return " | ".join(dict.fromkeys(values)) or "object"
    if schema_type == "string":
        return "str"
    if schema_type == "boolean":
        return "bool"
    if schema_type == "integer":
        return "int"
    if schema_type == "number":
        return "float"
    if schema_type == "array":
        return f"list[{_python_annotation(schema.get('items', {}))}]"
    if schema_type == "null":
        return "None"
    return "dict[str, object]"


def _python_schema_shape(
    schema: Mapping[str, Any],
    *,
    depth: int = 0,
) -> str:
    if depth >= 4:
        return _python_annotation(schema)
    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        values = [
            _python_schema_shape(item, depth=depth + 1)
            for item in any_of
            if isinstance(item, Mapping)
        ]
        return " | ".join(dict.fromkeys(values)) or "object"
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return " | ".join(repr(value) for value in enum)
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        values = [
            _python_schema_shape({**schema, "type": item}, depth=depth)
            for item in schema_type
            if isinstance(item, str)
        ]
        return " | ".join(dict.fromkeys(values)) or "object"
    if schema_type == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping) or not properties:
            return "dict[str, object]"
        required = set(schema.get("required", ()))
        fields = []
        for name, value in properties.items():
            if not isinstance(name, str) or not isinstance(value, Mapping):
                continue
            suffix = "" if name in required else "?"
            fields.append(
                f"{name}{suffix}: {_python_schema_shape(value, depth=depth + 1)}"
            )
        return "{" + ", ".join(fields) + "}"
    if schema_type == "array":
        items = schema.get("items", {})
        item_shape = (
            _python_schema_shape(items, depth=depth + 1)
            if isinstance(items, Mapping)
            else "object"
        )
        return f"list[{item_shape}]"
    return _python_annotation(schema)
