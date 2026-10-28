"""RL_KL_ON_KEPT=1: the k3 KL anchor is computed on the SAO ratio-mask kept tokens
only, rescaled per sample by sum(w_all)/sum(w_kept); masked tokens contribute nothing to the KL or its gradient, and with
every token kept the value and gradient are exactly the old implementation's."""
import math

import pytest
import torch

import train as T
from test_physics_critic import grad_of, setup_loss
from test_score_centering import FakePeft

BAND = (0.2, 0.28)
KW = dict(full_completion_tokens=3, world_size=1, score_centering=False, kl_beta=0.1)


def kl_only(s):
    return dict(s, advantage=0.0, token_advantage=torch.zeros(3))      # the loss is then beta * KL only


@pytest.fixture
def on_kept(monkeypatch):
    monkeypatch.setattr(T, "KL_ON_KEPT", True)


def samples(lp_tok):
    on = {"weight": 1.0, "advantage": 1.0, "token_advantage": torch.ones(3), "sampled_logprobs": lp_tok.clone()}
    # token 0: r = e^0.5 > 1.28 (high, masked); token 1: r = e^-0.5 < 0.8 (low, masked); token 2: r = e^0.1 kept
    off = dict(on, sampled_logprobs=lp_tok - torch.tensor([0.5, -0.5, 0.1]))
    return on, off


def test_masked_tokens_contribute_nothing_to_the_kl_or_its_gradient(on_kept):
    core, head, inputs, mask, logp, lp_tok = setup_loss()
    _, off = samples(lp_tok)
    ref_a = lambda i, m: lp_tok.detach() - torch.tensor([0.3, 0.3, 0.3])             # noqa: E731
    ref_b = lambda i, m: lp_tok.detach() - torch.tensor([9.0, -12.0, 0.3])           # masked tokens' reference changed wildly
    la, kla, _ = T.sample_loss(FakePeft(), core, head, inputs, mask, kl_only(off), ratio_band=BAND, reference=ref_a, **KW)
    lb, klb, _ = T.sample_loss(FakePeft(), core, head, inputs, mask, kl_only(off), ratio_band=BAND, reference=ref_b, **KW)
    assert torch.allclose(kla, klb) and torch.allclose(grad_of(la, head), grad_of(lb, head), atol=1e-7)
    # value: only token 2's k3, rescaled by 3 kept-weight / 1 = 3
    r2 = -0.3
    assert torch.allclose(kla, torch.tensor(3 * (math.exp(r2) - r2 - 1)), atol=1e-6)


def test_every_token_kept_reproduces_the_old_kl_exactly(on_kept, monkeypatch):
    core, head, inputs, mask, logp, lp_tok = setup_loss()
    on, _ = samples(lp_tok)
    ref = lambda i, m: lp_tok.detach() - torch.tensor([0.4, -0.2, 0.3])               # noqa: E731
    new_loss, new_kl, _ = T.sample_loss(FakePeft(), core, head, inputs, mask, on, ratio_band=BAND, reference=ref, **KW)
    g_new = grad_of(new_loss, head)
    monkeypatch.setattr(T, "KL_ON_KEPT", False)
    old_loss, old_kl, _ = T.sample_loss(FakePeft(), core, head, inputs, mask, on, ratio_band=BAND, reference=ref, **KW)
    assert torch.equal(new_kl, old_kl) and torch.equal(new_loss, old_loss) and torch.equal(g_new, grad_of(old_loss, head))


def test_token_weights_rescale_by_the_kept_weight_mass(on_kept):
    core, head, inputs, mask, logp, lp_tok = setup_loss()
    _, off = samples(lp_tok)
    w = torch.tensor([2.0, 1.0, 0.5])
    s = dict(kl_only(off), token_weights=w)
    ref = lambda i, m: lp_tok.detach() - 0.3                                           # noqa: E731
    _, kl, _ = T.sample_loss(FakePeft(), core, head, inputs, mask, s, ratio_band=BAND, reference=ref, **KW)
    k3 = math.exp(-0.3) + 0.3 - 1
    assert torch.allclose(kl, torch.tensor(0.5 * k3 * (3.5 / 0.5)), atol=1e-6)


def test_kl_stats_report_the_kl_before_and_after_the_mask(on_kept):
    core, head, inputs, mask, logp, lp_tok = setup_loss()
    _, off = samples(lp_tok)
    ref = lambda i, m: lp_tok.detach() - 0.3                                           # noqa: E731
    rows = []
    _, kl, _ = T.sample_loss(FakePeft(), core, head, inputs, mask, kl_only(off), ratio_band=BAND, reference=ref, kl_stats=rows, **KW)
    k3 = math.exp(-0.3) + 0.3 - 1
    pre, post = rows[0].tolist()
    assert math.isclose(pre, 3 * k3, rel_tol=1e-5) and math.isclose(post, float(kl), rel_tol=1e-6)


def test_no_kept_token_gives_zero_kl(on_kept):
    core, head, inputs, mask, logp, lp_tok = setup_loss()
    on, _ = samples(lp_tok)
    far = dict(kl_only(on), sampled_logprobs=lp_tok - 2.0)                             # every ratio e^2 > 1.28: all masked
    ref = lambda i, m: lp_tok.detach() - 0.3                                           # noqa: E731
    loss, kl, _ = T.sample_loss(FakePeft(), core, head, inputs, mask, far, ratio_band=BAND, reference=ref, **KW)
    assert float(kl) == 0.0 and torch.count_nonzero(grad_of(loss, head)) == 0


def test_default_is_off_and_keeps_every_token():
    assert T.KL_ON_KEPT is False or __import__("os").environ.get("RL_KL_ON_KEPT") == "1"
