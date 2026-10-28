from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from raw_request_training import InvalidRollout, build_request_samples, sampled_objective


def fixture(completions, selected=None):
    selected = list(range(len(completions))) if selected is None else selected
    p = len(selected) / len(completions)
    metadata = [{"request_index": i + 1, "prompt_length": 4, "completion_length": len(c)} for i, c in enumerate(completions)]
    return {"sample_format": "per_request_v1", "eligible_requests": metadata,
            "eligible_request_count": len(completions), "selected_request_count": len(selected),
            "full_completion_tokens": sum(map(len, completions)),
            "training_requests": [{"request_index": i + 1, "prompt_length": 4,
                "completion_length": len(completions[i]), "prompt_token_ids": [50, 60 + i, 70, 80],
                "token_ids": completions[i], "images": [], "image_hashes": [], "finish_reason": "length",
                "sampling": {"temperature": 1.0, "top_p": 1.0},
                "inclusion_probability": p, "loss_weight": 1 / p} for i in selected]}


def build(raw):
    return build_request_samples(raw, SimpleNamespace(), image_pad_id=-1, hard_max_tokens=1000)


def test_same_action_in_different_contexts_is_not_deduplicated():
    samples, total = build(fixture([[100, 101], [100, 101]]))
    assert total == 4 and len(samples) == 2
    assert samples[0]["input_ids"].tolist() != samples[1]["input_ids"].tolist()
    assert all(s["n_tokens"] == 2 for s in samples)


def test_raw_boundary_masks_generated_tokens_without_synthetic_eos():
    samples, _ = build(fixture([[248045, 74455, 198, 123]]))
    s = samples[0]
    assert s["input_ids"].tolist() == [50, 60, 70, 80, 248045, 74455, 198, 123]
    assert s["mask"].tolist() == [False] * 4 + [True] * 4


@pytest.mark.parametrize("field,value", [("sample_format", "segments"), ("full_completion_tokens", 500), ("eligible_request_count", 4)])
def test_invalid_metadata_fails_closed(field, value):
    raw = fixture([[100]])
    raw[field] = value
    with pytest.raises(InvalidRollout):
        build(raw)


def test_bad_selected_sample_never_silently_changes_denominator():
    raw = fixture([[100], [200]], [0])
    raw["training_requests"][0]["token_ids"] = []
    with pytest.raises(InvalidRollout):
        build(raw)


def test_duplicate_request_identity_rejected():
    raw = fixture([[100], [200]])
    raw["training_requests"][1] = deepcopy(raw["training_requests"][0])
    with pytest.raises(InvalidRollout):
        build(raw)


def test_missing_images_rejected():
    raw = fixture([[100]])
    raw["training_requests"][0]["image_hashes"] = ["missing"]
    with pytest.raises(InvalidRollout):
        build(raw)


def test_sampling_probability_mismatch_rejected():
    raw = fixture([[100]])
    raw["training_requests"][0]["sampling"]["temperature"] = 0.7
    with pytest.raises(InvalidRollout):
        build(raw)


@pytest.mark.parametrize("key,value", [
    ("min_tokens", 1), ("min_tokens", True), ("ignore_eos", True),
    ("structured_outputs", {"json": {}}), ("guided_regex", "[0-9]+"),
    ("response_format", {"type": "json_object"}), ("tool_choice", "required"),
    ("logits_processors", ["custom"]), ("temperature", True), ("n", True),
])
def test_probability_transforming_sampler_settings_are_rejected(key, value):
    raw = fixture([[100]])
    raw["training_requests"][0]["sampling"][key] = value
    with pytest.raises(InvalidRollout):
        build(raw)


def test_neutral_explicit_sampler_defaults_remain_valid():
    raw = fixture([[100]])
    raw["training_requests"][0]["sampling"].update(
        min_tokens=0, ignore_eos=False, response_format={"type": "text"}, tool_choice="auto")
    assert len(build(raw)[0]) == 1


def test_loss_builder_rechecks_original_image_hash():
    raw = fixture([[100]])
    raw["training_requests"][0].update(images=["data:image/png;base64,AAAA"], image_hashes=["wrong"])
    with pytest.raises(InvalidRollout, match="image payload hash differs"):
        build(raw)


def test_request_subsampling_preserves_policy_and_kl_gradients():
    full = 40
    reference = []
    sampled = []
    for indices, weight, dest in (([0, 1], 1, reference), ([0], 2, sampled), ([1], 2, sampled)):
        theta = torch.tensor([0.4, -0.7], requires_grad=True)
        loss = sum(sampled_objective(theta[0] * (10, 30)[i], theta[1] * (3, 17)[i],
                    advantage=(0.6, -0.8)[i], weight=weight, kl_beta=0.05,
                    full_completion_tokens=full) for i in indices)
        dest.append(torch.autograd.grad(loss, theta)[0])
    assert torch.allclose(reference[0], torch.stack(sampled).mean(0))


def test_zero_advantage_still_receives_reference_regularization():
    theta = torch.tensor(2.0, requires_grad=True)
    loss = sampled_objective(theta, theta.square(), advantage=0, weight=2,
                             kl_beta=0.05, full_completion_tokens=10)
    assert torch.autograd.grad(loss, theta)[0].item() == pytest.approx(0.04)
