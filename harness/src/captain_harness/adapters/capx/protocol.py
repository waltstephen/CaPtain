from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class PublicCompletionProtocol:
    schema_version: str
    protocol_id: str
    min_on_footprint_overlap: float
    min_mark_distance_m: float
    max_mark_distance_m: float
    mark_distance_destination_ratio: float
    min_eef_clearance_m: float
    max_eef_clearance_m: float
    eef_clearance_extent_ratio: float
    min_extent_ratio: float
    max_extent_ratio: float
    completion_candidate_profile: Mapping[str, object]
    signal_reliability_high_to_low: tuple[str, ...]
    confidence_weights: Mapping[str, float]
    confidence_penalties: Mapping[str, float]
    confidence_hard_demotions: Mapping[str, float]
    confidence_necessary_conditions: tuple[str, ...]
    recommendation_threshold: float
    source_path: str
    sha256: str

    def __post_init__(self) -> None:
        if self.schema_version not in {
            "captain-public-completion/v1",
            "captain-public-completion/v2",
            "captain-public-completion/v3",
        }:
            raise ValueError(
                "unsupported public completion schema: "
                f"{self.schema_version!r}"
            )
        if not self.protocol_id.strip() or not self.sha256.strip():
            raise ValueError("completion protocol id and sha256 must not be empty")
        values = (
            self.min_on_footprint_overlap,
            self.min_mark_distance_m,
            self.max_mark_distance_m,
            self.mark_distance_destination_ratio,
            self.min_eef_clearance_m,
            self.max_eef_clearance_m,
            self.eef_clearance_extent_ratio,
            self.min_extent_ratio,
            self.max_extent_ratio,
        )
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(
                "completion protocol values must be finite and non-negative"
            )
        if not 0 <= self.min_on_footprint_overlap <= 1:
            raise ValueError("min_on_footprint_overlap must be in [0, 1]")
        if self.min_mark_distance_m > self.max_mark_distance_m:
            raise ValueError("mark distance bounds are inverted")
        if self.min_eef_clearance_m > self.max_eef_clearance_m:
            raise ValueError("EEF clearance bounds are inverted")
        if (
            self.min_extent_ratio <= 0
            or self.min_extent_ratio > self.max_extent_ratio
        ):
            raise ValueError("extent ratio bounds are invalid")
        if self.schema_version in {
            "captain-public-completion/v2",
            "captain-public-completion/v3",
        }:
            if (
                self.completion_candidate_profile.get(
                    "recommendation_only"
                )
                is not True
            ):
                raise ValueError(
                    "V2 completion candidate profile must be "
                    "recommendation-only"
                )
            required = {
                "rule_id",
                "min_xy_overlap_ratio",
                "min_mark_vertical_delta_m",
                "max_mark_vertical_delta_m",
                "max_static_displacement_m",
                "hard_support_gap_condition",
            }
            if not required.issubset(
                self.completion_candidate_profile
            ):
                raise ValueError(
                    "V2 completion candidate profile is incomplete"
                )
            if (
                self.completion_candidate_profile.get(
                    "hard_support_gap_condition"
                )
                is not False
            ):
                raise ValueError(
                    "V2 completion cannot hard-gate on bounds support gap"
                )
            if not self.signal_reliability_high_to_low:
                raise ValueError(
                    "V2 completion signal reliability order is empty"
                )
            if (
                not self.confidence_weights
                or not all(
                    math.isfinite(float(value))
                    and float(value) >= 0
                    for value in self.confidence_weights.values()
                )
                or sum(self.confidence_weights.values()) <= 0
            ):
                raise ValueError(
                    "V2 completion confidence weights are invalid"
                )
        if self.schema_version == "captain-public-completion/v3":
            expected_weights = {
                "robust_vertical_gap_m",
                "mark_vertical_delta_m",
                "xy_overlap_ratio",
                "target_static_after_release_m",
                "gripper_released",
                "centroid_inside_destination_xy",
                "target_destination_distinct",
            }
            if set(self.confidence_weights) != expected_weights:
                raise ValueError(
                    "V3 completion confidence weights do not match "
                    "the approved signal set"
                )
            if not math.isclose(
                sum(self.confidence_weights.values()),
                1.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "V3 completion confidence weights must sum to one"
                )
            if (
                not self.confidence_penalties
                or any(
                    not math.isfinite(value) or not 0 <= value <= 1
                    for value in self.confidence_penalties.values()
                )
            ):
                raise ValueError(
                    "V3 completion confidence penalties are invalid"
                )
            if any(
                not math.isfinite(value) or not 0 <= value <= 1
                for value in self.confidence_hard_demotions.values()
            ):
                raise ValueError(
                    "V3 completion hard demotions are invalid"
                )
            if "gripper_released" not in self.confidence_necessary_conditions:
                raise ValueError(
                    "V3 completion requires gripper_released as a "
                    "necessary confidence condition"
                )
            if not 0 <= self.recommendation_threshold <= 1:
                raise ValueError(
                    "V3 completion recommendation threshold must be in [0, 1]"
                )
            if (
                self.candidate_profile_number(
                    "min_robust_vertical_gap_m",
                    default=math.nan,
                )
                < 0
            ):
                raise ValueError(
                    "V3 minimum robust vertical gap must be non-negative"
                )

    @property
    def evidence_only(self) -> bool:
        return self.schema_version in {
            "captain-public-completion/v2",
            "captain-public-completion/v3",
        }

    @property
    def confidence_repair(self) -> bool:
        return self.schema_version == "captain-public-completion/v3"

    def candidate_profile_number(
        self,
        name: str,
        *,
        default: float,
    ) -> float:
        value = self.completion_candidate_profile.get(name, default)
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(
                f"completion candidate profile {name!r} is not finite"
            )
        return number

    def score_confidence(
        self,
        signals: Mapping[str, float],
        quality_flags: tuple[str, ...] | list[str],
    ) -> tuple[float, dict[str, float], float]:
        weighted = [
            (name, float(weight), float(signals[name]))
            for name, weight in self.confidence_weights.items()
            if name in signals
        ]
        weight_total = sum(weight for _, weight, _ in weighted)
        components = {
            name: (
                weight * value / weight_total
                if weight_total
                else 0.0
            )
            for name, weight, value in weighted
        }
        confidence = sum(components.values()) if weighted else 0.0
        penalty = 1.0
        if self.confidence_repair:
            for flag in dict.fromkeys(quality_flags):
                penalty *= self.confidence_penalties.get(flag, 1.0)
                penalty *= self.confidence_hard_demotions.get(flag, 1.0)
            if any(
                signals.get(name) != 1.0
                for name in self.confidence_necessary_conditions
            ):
                penalty = 0.0
            confidence *= penalty
        return (
            min(1.0, max(0.0, confidence)),
            components,
            min(1.0, max(0.0, penalty)),
        )

    def public_metadata(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "evidence_only": self.evidence_only,
            "confidence_repair": self.confidence_repair,
            "completion_candidate_profile": dict(
                self.completion_candidate_profile
            ),
            "signal_reliability_high_to_low": list(
                self.signal_reliability_high_to_low
            ),
            "confidence_weights": dict(self.confidence_weights),
            "confidence_penalties": dict(self.confidence_penalties),
            "confidence_hard_demotions": dict(
                self.confidence_hard_demotions
            ),
            "confidence_necessary_conditions": list(
                self.confidence_necessary_conditions
            ),
            "recommendation_threshold": self.recommendation_threshold,
            "source_path": self.source_path,
            "sha256": self.sha256,
        }


def default_public_completion_protocol_path() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "configs"
        / "protocols"
        / "v10_public_completion_v1.json"
    )


def load_public_completion_protocol(
    path: Path | str | None = None,
) -> PublicCompletionProtocol:
    source = (
        default_public_completion_protocol_path()
        if path is None
        else Path(path).expanduser().resolve()
    )
    raw = source.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("public completion protocol must be a JSON object")
    return PublicCompletionProtocol(
        schema_version=str(payload.get("schema_version", "")),
        protocol_id=str(payload.get("protocol_id", "")),
        min_on_footprint_overlap=float(
            payload.get("min_on_footprint_overlap", math.nan)
        ),
        min_mark_distance_m=float(
            payload.get("min_mark_distance_m", math.nan)
        ),
        max_mark_distance_m=float(
            payload.get("max_mark_distance_m", math.nan)
        ),
        mark_distance_destination_ratio=float(
            payload.get("mark_distance_destination_ratio", math.nan)
        ),
        min_eef_clearance_m=float(
            payload.get("min_eef_clearance_m", math.nan)
        ),
        max_eef_clearance_m=float(
            payload.get("max_eef_clearance_m", math.nan)
        ),
        eef_clearance_extent_ratio=float(
            payload.get("eef_clearance_extent_ratio", math.nan)
        ),
        min_extent_ratio=float(
            payload.get("min_extent_ratio", math.nan)
        ),
        max_extent_ratio=float(
            payload.get("max_extent_ratio", math.nan)
        ),
        completion_candidate_profile=(
            dict(payload["completion_candidate_profile"])
            if isinstance(
                payload.get("completion_candidate_profile"),
                Mapping,
            )
            else {}
        ),
        signal_reliability_high_to_low=tuple(
            str(value)
            for value in payload.get(
                "signal_reliability_high_to_low",
                (),
            )
        ),
        confidence_weights=(
            {
                str(name): float(value)
                for name, value in payload["confidence_weights"].items()
            }
            if isinstance(payload.get("confidence_weights"), Mapping)
            else {}
        ),
        confidence_penalties=(
            {
                str(name): float(value)
                for name, value in payload["confidence_penalties"].items()
            }
            if isinstance(payload.get("confidence_penalties"), Mapping)
            else {}
        ),
        confidence_hard_demotions=(
            {
                str(name): float(value)
                for name, value in payload[
                    "confidence_hard_demotions"
                ].items()
            }
            if isinstance(
                payload.get("confidence_hard_demotions"),
                Mapping,
            )
            else {}
        ),
        confidence_necessary_conditions=tuple(
            str(value)
            for value in payload.get(
                "confidence_necessary_conditions",
                (),
            )
        ),
        recommendation_threshold=float(
            payload.get("recommendation_threshold", 0.75)
        ),
        source_path=str(source),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
