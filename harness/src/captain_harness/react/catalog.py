"""Per-model settings mirrored from Codex CLI 0.154's model catalog, plus the V1.2 context policy.

Context policy (used by react.loop / react.context):
* context_limit        estimated total tokens at which the driver compacts BEFORE sending the next request
* context_strategy     "summary" (Pi/Codex style: programmatic facts + incremental LLM summary + recent turns),
                       "notes" (one checkpoint turn to update NOTES.md, then facts + notes + recent turns)
                       or "window" (deterministic: programmatic facts + recent turns, no LLM summary)
* keep_recent_tokens   budget of most recent whole turns kept verbatim after a compaction
* truncation_tokens    per-tool-output cap (Codex: 10k)
* spill_tool_output    when a tool output is truncated, save the full text under <workspace>/.tool_output/ and tell the model
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelProfile:
    slug: str
    template: str               # request template under react/prompts/
    tool_mode: str              # "function" (exec_command/write_stdin/view_image/apply_patch) or "code_mode" (exec JS)
    truncation_tokens: int = 10_000
    context_window: int = 272_000
    effective_context_percent: int = 95
    auto_compact_token_limit: int | None = None
    image_detail_original: bool = True
    context_strategy: str = "summary"
    keep_recent_tokens: int = 20_000
    summary_budget_tokens: int = 3_000
    spill_tool_output: bool = False

    @property
    def context_limit(self) -> int:
        """Estimated-token ceiling at which the driver compacts (Codex: auto_compact_token_limit or 90% of the window)."""
        if self.auto_compact_token_limit is not None:
            return self.auto_compact_token_limit
        return self.context_window * 9 // 10


PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile("gpt-5.5", "gpt-5.5.request.json", "function", 10_000, 1_050_000, 95, 880_000, True),
    ModelProfile("captain", "captain.request.json", "code_mode", 10_000, 900_000, 95, 840_000, True),
)


def profile_for(model: str, *, fallback: str | None = None) -> ModelProfile:
    """Longest-prefix match like Codex's catalog lookup (gpt-5.5-anything -> gpt-5.5).

    ``fallback`` names the profile to use for models outside the catalog (local students served
    through the Chat Completions backend borrow a captured template and its tool mode).
    """
    candidates = [p for p in PROFILES if model == p.slug or model.startswith(p.slug)]
    if not candidates:
        if fallback:
            return profile_for(fallback)
        raise KeyError(f"no driver profile for model {model!r}; known: {[p.slug for p in PROFILES]}")
    return max(candidates, key=lambda p: len(p.slug))


def one_turn_tokens(profile: ModelProfile, *, parallel_tools: int = 3, images: int = 2, image_tokens: int = 1_000, overhead: int = 2_000) -> int:
    """Upper bound of what one model turn can append before the next pre-request check (tool outputs + images + call text)."""
    return parallel_tools * profile.truncation_tokens + images * image_tokens + overhead


def trigger_for_hard_limit(profile: ModelProfile, hard_limit: int) -> ModelProfile:
    """Compact so that no request ever exceeds ``hard_limit`` real tokens (e.g. the training sample cap):
    trigger = hard_limit - one_turn_tokens; the estimator's calibration covers the tokenizer mismatch."""
    trigger = hard_limit - one_turn_tokens(profile)
    if trigger < 8_000:
        raise ValueError(f"hard limit {hard_limit} leaves no room for a turn (one turn ~{one_turn_tokens(profile)} tokens)")
    return replace(profile, auto_compact_token_limit=trigger)


def small_context_profile(base: ModelProfile, *, context_window: int, keep_recent_tokens: int | None = None,
                          truncation_tokens: int = 4_000, strategy: str = "window") -> ModelProfile:
    """A profile for a small-context model: compact at window - max(10%, one turn), keep ~25% of the window verbatim,
    cap tool outputs and spill them to files. The same profile must be used when producing its training data."""
    one_turn = 3 * truncation_tokens + 4_000
    limit = context_window - max(context_window // 10, one_turn)
    return replace(base, context_window=context_window, auto_compact_token_limit=limit, truncation_tokens=truncation_tokens,
                   keep_recent_tokens=keep_recent_tokens or context_window // 4, context_strategy=strategy, spill_tool_output=True)


def with_context_window(profile: ModelProfile, context_window: int, *, effective_percent: int = 90) -> ModelProfile:
    """The same template and tool mode with a different context budget (e.g. a 131k vLLM server)."""
    return ModelProfile(profile.slug, profile.template, profile.tool_mode, profile.truncation_tokens, int(context_window),
                        profile.effective_context_percent, int(context_window) * effective_percent // 100, profile.image_detail_original)


__all__ = ["ModelProfile", "PROFILES", "one_turn_tokens", "profile_for", "small_context_profile", "trigger_for_hard_limit", "with_context_window"]
