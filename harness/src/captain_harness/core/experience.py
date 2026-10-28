from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence


_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_FORBIDDEN_ENTRY_KEYS = {
    "task",
    "task_id",
    "seed",
    "success",
    "reward",
    "predicate",
    "check_success",
    "done",
    "demonstration",
    "generated_code",
    "code",
    "policy",
    "solution",
}
_ALLOWED_ENTRY_KEYS = {
    "object_class",
    "observed_extent_m",
    "gripper_max_width_m",
    "attempts",
    "by_backend",
    "by_style",
    "width_measurable_rate",
    "notes",
}
_FORBIDDEN_NOTE_PATTERN = re.compile(
    r"\b(task|seed|reward|predicate|demonstration|generated[_ ]code|"
    r"policy|solution)\b",
    flags=re.IGNORECASE,
)


class ExperienceMode(str, Enum):
    BUILD = "build"
    COLD = "cold"
    WARM = "warm"


@dataclass(frozen=True, slots=True)
class ExperienceKey:
    environment_id: str
    environment_revision: str
    tool_revisions: Mapping[str, str]
    source_tree_sha256: str | None = None
    implementation_revision: str | None = None

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.environment_id):
            raise ValueError("environment_id must be path-safe")
        if not self.environment_revision.strip():
            raise ValueError("environment_revision must not be empty")
        if (self.source_tree_sha256 is None) == (self.implementation_revision is None):
            raise ValueError("exactly one source hash or development implementation revision is required")
        binding = self.source_tree_sha256 if self.source_tree_sha256 is not None else self.implementation_revision
        if not isinstance(binding, str) or not binding.strip():
            raise ValueError("experience implementation binding must not be empty")
        if self.implementation_revision is not None and not _SAFE_ID.fullmatch(self.implementation_revision):
            raise ValueError("implementation_revision must be an explicit path-safe revision")
        if not self.tool_revisions or any(
            not str(name).strip() or not str(revision).strip()
            for name, revision in self.tool_revisions.items()
        ):
            raise ValueError("tool_revisions must contain non-empty entries")

    def canonical(self) -> dict[str, object]:
        return {
            "environment_id": self.environment_id,
            "environment_revision": self.environment_revision,
            "tool_revisions": dict(sorted(self.tool_revisions.items())),
            **({"source_tree_sha256": self.source_tree_sha256} if self.source_tree_sha256 is not None
               else {"implementation_revision": self.implementation_revision}),
        }


@dataclass(frozen=True, slots=True)
class ExperienceSnapshot:
    mode: ExperienceMode
    key: ExperienceKey
    entries: tuple[Mapping[str, object], ...]
    store_sha256: str
    manifest_path: str | None = None
    store_path: str | None = None

    @classmethod
    def cold(cls, key: ExperienceKey) -> "ExperienceSnapshot":
        return cls(
            mode=ExperienceMode.COLD,
            key=key,
            entries=(),
            store_sha256=hashlib.sha256(b"").hexdigest(),
        )

    def entries_for_object_classes(
        self,
        object_classes: Sequence[str],
    ) -> tuple[Mapping[str, object], ...]:
        normalized = {
            str(value).strip().casefold()
            for value in object_classes
            if str(value).strip()
        }
        if not normalized:
            return ()
        return tuple(
            entry
            for entry in self.entries
            if str(entry.get("object_class", "")).strip().casefold()
            in normalized
        )

    def context_text(self) -> str:
        catalog = [_validate_entry(entry) for entry in self.entries]
        return json.dumps(
            {
                "experience_mode": self.mode.value,
                "environment_id": self.key.environment_id,
                "environment_revision": self.key.environment_revision,
                "store_sha256": self.store_sha256,
                "entries": catalog,
                "interpretation": "Tool execution successes are not task success; unknown target effects remain unknown. Unknown can mean the capability does not report target effects, including read-only tools; it is not failure. Counts are observational, not causal recommendations.",
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    def public_metadata(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            **self.key.canonical(),
            "entry_count": len(self.entries),
            "store_sha256": self.store_sha256,
            "manifest_path": self.manifest_path,
            "store_path": self.store_path,
            "read_only": self.mode is not ExperienceMode.BUILD,
        }


class EnvironmentExperienceStore:
    def __init__(
        self,
        *,
        root: Path,
        key: ExperienceKey,
        mode: ExperienceMode,
        built_from: Sequence[Mapping[str, object]] = (),
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.key = key
        self.mode = ExperienceMode(mode)
        self.built_from = tuple(dict(value) for value in built_from)
        self.directory = self.root / key.environment_id
        self.store_path = self.directory / "store.jsonl"
        self.manifest_path = self.directory / "manifest.json"
        self._entries: list[dict[str, object]] = []
        self._frozen = False

    @classmethod
    def create_build(
        cls,
        root: Path,
        key: ExperienceKey,
        *,
        built_from: Sequence[Mapping[str, object]] = (),
    ) -> "EnvironmentExperienceStore":
        store = cls(
            root=root,
            key=key,
            mode=ExperienceMode.BUILD,
            built_from=built_from,
        )
        if store.directory.exists():
            raise FileExistsError(
                f"experience store already exists: {store.directory}"
            )
        store.directory.mkdir(parents=True)
        store.store_path.touch(exist_ok=False)
        store._write_manifest(
            {
                "schema_version": "captain-experience-store/v1",
                **key.canonical(),
                "built_from": list(store.built_from),
                "frozen_at": None,
                "entry_count": 0,
                "store_sha256": None,
                "status": "building",
            }
        )
        return store

    @classmethod
    def open_frozen(
        cls,
        root: Path,
        key: ExperienceKey,
    ) -> "EnvironmentExperienceStore":
        store = cls(
            root=root,
            key=key,
            mode=ExperienceMode.WARM,
        )
        manifest = store._read_manifest()
        store._verify_key(manifest)
        if manifest.get("status") != "frozen" or not manifest.get(
            "frozen_at"
        ):
            raise ValueError("experience store is not frozen")
        raw = store.store_path.read_bytes()
        observed_sha = hashlib.sha256(raw).hexdigest()
        if manifest.get("store_sha256") != observed_sha:
            raise ValueError("experience store sha256 mismatch")
        entries = [
            json.loads(line)
            for line in raw.decode("utf-8").splitlines()
            if line.strip()
        ]
        if len(entries) != int(manifest.get("entry_count", -1)):
            raise ValueError("experience entry_count mismatch")
        for entry in entries:
            store._entries.append(_validate_entry(entry))
        store.built_from = tuple(
            dict(value)
            for value in manifest.get("built_from", ())
            if isinstance(value, Mapping)
        )
        store._frozen = True
        return store

    def append(self, entry: Mapping[str, object]) -> None:
        if self.mode is not ExperienceMode.BUILD or self._frozen:
            raise PermissionError(
                "evaluation-phase experience stores are read-only"
            )
        validated = _validate_entry(entry)
        line = json.dumps(
            validated,
            separators=(",", ":"),
            sort_keys=True,
        )
        with self.store_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        self._entries.append(validated)

    def freeze(self) -> ExperienceSnapshot:
        if self.mode is not ExperienceMode.BUILD or self._frozen:
            raise PermissionError("only an active build store can be frozen")
        raw = self.store_path.read_bytes()
        store_sha = hashlib.sha256(raw).hexdigest()
        self._write_manifest(
            {
                "schema_version": "captain-experience-store/v1",
                **self.key.canonical(),
                "built_from": list(self.built_from),
                "frozen_at": datetime.now(timezone.utc).isoformat(),
                "entry_count": len(self._entries),
                "store_sha256": store_sha,
                "status": "frozen",
            }
        )
        self._frozen = True
        return ExperienceSnapshot(
            mode=ExperienceMode.WARM,
            key=self.key,
            entries=tuple(self._entries),
            store_sha256=store_sha,
            manifest_path=str(self.manifest_path),
            store_path=str(self.store_path),
        )

    def snapshot(self) -> ExperienceSnapshot:
        if self.mode is ExperienceMode.BUILD and not self._frozen:
            raise RuntimeError("build store must be frozen before evaluation")
        raw = self.store_path.read_bytes()
        return ExperienceSnapshot(
            mode=ExperienceMode.WARM,
            key=self.key,
            entries=tuple(self._entries),
            store_sha256=hashlib.sha256(raw).hexdigest(),
            manifest_path=str(self.manifest_path),
            store_path=str(self.store_path),
        )

    def _verify_key(self, manifest: Mapping[str, object]) -> None:
        observed = {name: manifest.get(name) for name in self.key.canonical()}
        other_binding = "implementation_revision" if self.key.source_tree_sha256 is not None else "source_tree_sha256"
        if manifest.get(other_binding) is not None:
            raise ValueError("experience store key has conflicting implementation bindings")
        if observed != self.key.canonical():
            raise ValueError(
                "experience store key does not match the active environment"
            )

    def _read_manifest(self) -> Mapping[str, object]:
        payload = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        if not isinstance(payload, Mapping):
            raise ValueError("experience manifest must be an object")
        if payload.get("schema_version") != "captain-experience-store/v1":
            raise ValueError("unsupported experience manifest schema")
        return payload

    def _write_manifest(self, payload: Mapping[str, object]) -> None:
        self.manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _validate_entry(
    entry: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(entry, Mapping):
        raise ValueError("experience entry must be an object")
    _reject_forbidden_keys(entry)
    if entry.get("kind") == "tool_efficacy":
        from .tool_efficacy import validate_tool_efficacy_entry
        return validate_tool_efficacy_entry(entry)
    extra = set(entry).difference(_ALLOWED_ENTRY_KEYS)
    missing = _ALLOWED_ENTRY_KEYS.difference(entry)
    if extra or missing:
        raise ValueError(
            "experience entry fields do not match the approved schema: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    object_class = str(entry.get("object_class", "")).strip()
    if not object_class or len(object_class) > 120:
        raise ValueError("experience object_class must not be empty")
    observed_extent = _finite_positive_vector(
        entry.get("observed_extent_m"),
        "observed_extent_m",
    )
    gripper_max_width = _finite_rate_or_positive(
        entry.get("gripper_max_width_m"),
        "gripper_max_width_m",
        upper_bound=None,
        positive=True,
    )
    attempts = entry.get("attempts")
    if (
        not isinstance(attempts, int)
        or isinstance(attempts, bool)
        or attempts < 0
    ):
        raise ValueError("experience attempts must be a non-negative integer")
    by_backend_value = entry.get("by_backend")
    if not isinstance(by_backend_value, Mapping) or not by_backend_value:
        raise ValueError("experience by_backend must be a non-empty object")
    by_backend: dict[str, dict[str, float]] = {}
    for backend, metrics in by_backend_value.items():
        backend_name = str(backend).strip()
        if not backend_name or not isinstance(metrics, Mapping):
            raise ValueError("experience backend entries must be objects")
        if set(metrics) != {"candidates", "attached"}:
            raise ValueError(
                "experience backend metrics must be candidates and attached"
            )
        by_backend[backend_name] = {
            metric: _finite_rate_or_positive(
                metrics[metric],
                f"by_backend.{backend_name}.{metric}",
                upper_bound=1.0,
            )
            for metric in ("candidates", "attached")
        }
    by_style_value = entry.get("by_style")
    if not isinstance(by_style_value, Mapping) or not by_style_value:
        raise ValueError("experience by_style must be a non-empty object")
    by_style = {
        str(style).strip(): _finite_rate_or_positive(
            rate,
            f"by_style.{style}",
            upper_bound=1.0,
        )
        for style, rate in by_style_value.items()
        if str(style).strip()
    }
    if len(by_style) != len(by_style_value):
        raise ValueError("experience style names must not be empty")
    width_measurable_rate = _finite_rate_or_positive(
        entry.get("width_measurable_rate"),
        "width_measurable_rate",
        upper_bound=1.0,
    )
    notes = entry.get("notes")
    if (
        not isinstance(notes, str)
        or len(notes) > 500
        or _FORBIDDEN_NOTE_PATTERN.search(notes)
    ):
        raise ValueError(
            "experience notes must be short public tool-efficacy text"
        )
    validated = {
        "object_class": object_class,
        "observed_extent_m": observed_extent,
        "gripper_max_width_m": gripper_max_width,
        "attempts": attempts,
        "by_backend": by_backend,
        "by_style": by_style,
        "width_measurable_rate": width_measurable_rate,
        "notes": notes,
    }
    return json.loads(
        json.dumps(
            validated,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _finite_positive_vector(
    value: object,
    name: str,
) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"experience {name} must contain three values")
    try:
        parsed = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"experience {name} values must be numeric"
        ) from exc
    if not all(math.isfinite(item) and item > 0 for item in parsed):
        raise ValueError(
            f"experience {name} values must be finite and positive"
        )
    return parsed


def _finite_rate_or_positive(
    value: object,
    name: str,
    *,
    upper_bound: float | None,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"experience {name} must be numeric")
    parsed = float(value)
    lower_ok = parsed > 0 if positive else parsed >= 0
    if (
        not math.isfinite(parsed)
        or not lower_ok
        or (upper_bound is not None and parsed > upper_bound)
    ):
        raise ValueError(f"experience {name} is outside its valid range")
    return parsed


def _reject_forbidden_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().casefold()
            if normalized in _FORBIDDEN_ENTRY_KEYS:
                raise ValueError(
                    f"forbidden experience field: {key}"
                )
            _reject_forbidden_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_forbidden_keys(item)
