from __future__ import annotations

from collections.abc import Callable


class HostOnlyFinalScorer:
    """Keep hidden audit samples outside the capability boundary."""

    def __init__(self, predicate: Callable[[], bool]) -> None:
        self.__predicate = predicate
        self.__used = False
        self.__sampled = False
        self.__success_ever = False
        self.__first_success_step: int | None = None

    def score_after_action(self, *, environment_step: int) -> bool:
        if self.__used:
            raise RuntimeError(
                "hidden audit predicate is unavailable after final scoring"
            )
        self.__sampled = True
        success = bool(self.__predicate())
        if success and not self.__success_ever:
            self.__success_ever = True
            self.__first_success_step = int(environment_step)
        return success

    def score_after_loop(self, *, loop_ended: bool) -> bool:
        if not loop_ended:
            raise RuntimeError("final predicate is forbidden before the agent loop ends")
        if self.__used:
            raise RuntimeError("final predicate may be called only once")
        self.__used = True
        success = bool(self.__predicate())
        if self.__sampled and success:
            self.__success_ever = True
        return success

    @property
    def success_ever(self) -> bool | None:
        return self.__success_ever if self.__sampled else None

    @property
    def first_success_step(self) -> int | None:
        return self.__first_success_step
