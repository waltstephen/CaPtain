"""Gradient accumulation across collection rounds (one optimizer step per K rounds).

Round r of a cycle trains its samples with the trainer's usual per-round normaliser N_r (loss_normalization trajectory_mean:
the round's trajectory count, token_mean: its completion tokens), so its backward adds g_r = sum_i l_i / N_r to the gradients.
Before that backward the gradients already held are rescaled by N_{r-1} / N_r, and after the last round by N_K / sum_r N_r:

    A_1 = g_1,   A_r = A_{r-1} * N_{r-1} / N_r + g_r = (sum_{j<=r} sum_i l_i) / N_r,   final = A_K * N_K / sum_r N_r

which is exactly the gradient of one batch made of all K rounds normalised by its total N (fp rounding aside). Clipping and the
optimizer step then run once on that gradient. A round without signal (N_r = 0) contributes nothing: the caller restores the
gradients it held before that round (``stash`` / ``restore``).
"""
from __future__ import annotations

from typing import Iterable

import torch


class RoundAccumulator:
    def __init__(self, k: int) -> None:
        if type(k) is not int or k < 1:
            raise ValueError("grad_accum_rounds must be a positive integer")
        self.k = k
        self.reset()

    def reset(self) -> None:
        self.prev = 0.0      # normaliser of the last round that contributed (the scale the held gradients are at)
        self.total = 0.0     # sum of the contributing rounds' normalisers
        self.rounds = 0      # rounds seen in this cycle (contributing or not)

    def is_first(self, step: int) -> bool:
        """collection rounds are numbered from 1; a cycle is rounds (c-1)K+1 .. cK, so a checkpoint at step % K == 0 never
        holds a partial gradient"""
        return (step - 1) % self.k == 0

    def is_last(self, step: int) -> bool:
        return step % self.k == 0

    @staticmethod
    def _scale(params: Iterable[torch.Tensor], factor: float) -> None:
        if factor == 1.0:
            return
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(factor)

    def before_round(self, params: Iterable[torch.Tensor], n: float) -> None:
        """call after this round's normaliser is known and before its first backward; n <= 0 = the round adds nothing"""
        self.rounds += 1
        if n <= 0:
            return
        if self.prev > 0:
            self._scale(params, self.prev / n)
        self.prev = float(n)
        self.total += float(n)

    def finish(self, params: Iterable[torch.Tensor]) -> bool:
        """after the cycle's last backward: bring the held gradients to the 1 / sum N scale; False = nothing to step on"""
        if self.total <= 0:
            return False
        self._scale(params, self.prev / self.total)
        return True

    @staticmethod
    def stash(params: Iterable[torch.Tensor]) -> list:
        return [None if p.grad is None else p.grad.detach().clone() for p in params]

    @staticmethod
    def restore(params: Iterable[torch.Tensor], saved: list) -> None:
        with torch.no_grad():
            for p, g in zip(params, saved):
                if g is None:
                    p.grad = None
                elif p.grad is None:
                    p.grad = g
                else:
                    p.grad.copy_(g)
