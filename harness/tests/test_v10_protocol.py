from __future__ import annotations

import json
from pathlib import Path

import pytest

from captain_harness.adapters.capx.protocol import (
    default_public_completion_protocol_path,
    load_public_completion_protocol,
)


def test_v10_completion_protocol_is_externalized_and_hashed() -> None:
    path = default_public_completion_protocol_path()
    protocol = load_public_completion_protocol()

    assert path == Path(
        "configs/protocols/v10_public_completion_v1.json"
    ).resolve()
    assert protocol.protocol_id == "libero-public-completion-v10"
    assert protocol.min_on_footprint_overlap == 0.83
    assert protocol.max_mark_distance_m == 0.10
    assert len(protocol.sha256) == 64
    assert protocol.public_metadata()["sha256"] == protocol.sha256


def test_v10_completion_protocol_rejects_inverted_bounds(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        default_public_completion_protocol_path().read_text(encoding="utf-8")
    )
    payload["min_mark_distance_m"] = 0.2
    payload["max_mark_distance_m"] = 0.1
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="mark distance bounds"):
        load_public_completion_protocol(path)


def test_v11_completion_protocol_is_evidence_only_and_recommendation_only() -> None:
    protocol = load_public_completion_protocol(
        "configs/protocols/v11_public_completion_v1.json"
    )

    assert protocol.schema_version == "captain-public-completion/v2"
    assert protocol.protocol_id == "libero-public-completion-v11"
    assert protocol.evidence_only is True
    assert (
        protocol.completion_candidate_profile["recommendation_only"]
        is True
    )
    assert (
        protocol.completion_candidate_profile[
            "hard_support_gap_condition"
        ]
        is False
    )
    assert protocol.public_metadata()["confidence_weights"] == {
        "centroid_inside_destination_xy": 0.12,
        "eef_clearance_m": 0.06,
        "gripper_released": 0.22,
        "mark_vertical_delta_m": 0.12,
        "target_destination_distinct": 0.08,
        "target_static_after_release_m": 0.18,
        "xy_overlap_ratio": 0.22,
    }


def test_v12_completion_protocol_repairs_confidence_components() -> None:
    protocol = load_public_completion_protocol(
        "configs/protocols/v12_public_completion_v1.json"
    )

    assert protocol.schema_version == "captain-public-completion/v3"
    assert protocol.protocol_id == "libero-public-completion-v12"
    assert protocol.evidence_only is True
    assert protocol.confidence_repair is True
    assert protocol.recommendation_threshold == 0.75
    assert protocol.confidence_weights == {
        "centroid_inside_destination_xy": 0.05,
        "gripper_released": 0.1,
        "mark_vertical_delta_m": 0.2,
        "robust_vertical_gap_m": 0.3,
        "target_destination_distinct": 0.05,
        "target_static_after_release_m": 0.15,
        "xy_overlap_ratio": 0.15,
    }


def test_confidence_demotes_on_negative_gap() -> None:
    protocol = load_public_completion_protocol(
        "configs/protocols/v12_public_completion_v1.json"
    )
    signals = {name: 1.0 for name in protocol.confidence_weights}
    signals["robust_vertical_gap_m"] = 0.0

    confidence, components, penalty = protocol.score_confidence(
        signals,
        ["target_intersects_destination"],
    )

    assert components["robust_vertical_gap_m"] == 0.0
    assert penalty == 0.0
    assert confidence == 0.0


def test_quality_flags_reduce_confidence() -> None:
    protocol = load_public_completion_protocol(
        "configs/protocols/v12_public_completion_v1.json"
    )
    signals = {name: 1.0 for name in protocol.confidence_weights}

    clean, _, clean_penalty = protocol.score_confidence(signals, [])
    occluded, _, occluded_penalty = protocol.score_confidence(
        signals,
        [
            "target_underside_occluded",
            "destination_partially_covered",
        ],
    )

    assert clean == 1.0
    assert clean_penalty == 1.0
    assert occluded_penalty == pytest.approx(0.6)
    assert occluded == pytest.approx(0.6)
