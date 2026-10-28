from __future__ import annotations

import re
import secrets
import time
from dataclasses import dataclass
from typing import Callable

_HANDLE = re.compile(
    r"^artifact://v1/(?P<episode>[A-Za-z0-9_.-]+)/(?P<kind>[A-Za-z0-9_.-]+)/(?P<token>[A-Za-z0-9_-]+)$"
)


class ArtifactHandleError(ValueError):
    pass


@dataclass(slots=True)
class _Entry:
    kind: str
    value: object
    expires_at: float | None
    stored: bool = False


class EpisodeArtifactRegistry:
    """Own episode artifacts; None TTL retains historical evidence until close.

    Handle lifetime does not establish sensor or motion-plan freshness. Those
    checks belong to the consuming capabilities and still apply to retained data.
    """

    def __init__(
        self,
        episode_id: str,
        *,
        ttl_s: float | None,
        clock: Callable[[], float] = time.monotonic,
        storage_root=None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", episode_id):
            raise ValueError("episode_id is not handle-safe")
        if ttl_s is not None and ttl_s <= 0:
            raise ValueError("artifact ttl_s must be positive")
        self.episode_id = episode_id
        self.ttl_s = ttl_s
        self._clock = clock
        self._entries: dict[str, _Entry] = {}
        self._closed = False
        self._objects = None
        if storage_root is not None:
            from captain_harness.ports.host_objects import HostObjectStore
            self._objects = HostObjectStore(storage_root)


    def put(self, kind: str, value: object) -> str:
        if self._closed:
            raise ArtifactHandleError("artifact registry is closed")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", kind):
            raise ValueError("artifact kind is not handle-safe")
        token = secrets.token_urlsafe(18)
        handle = f"artifact://v1/{self.episode_id}/{kind}/{token}"
        expires_at = None if self.ttl_s is None else self._clock() + self.ttl_s
        stored = self._objects is not None and kind in {"observation","rgb","depth","point_cloud"}
        if stored:
            self._objects.put(handle,value)
        self._entries[handle] = _Entry(kind, None if stored else value, expires_at, stored)
        return handle

    def get(self, handle: str, *, expected_kind: str | None = None) -> object:
        if self._closed:
            raise ArtifactHandleError("artifact registry is closed")
        match = _HANDLE.fullmatch(handle) if isinstance(handle, str) else None
        if match is None:
            raise ArtifactHandleError("invalid artifact handle format")
        owner = match.group("episode")
        if owner != self.episode_id:
            raise ArtifactHandleError(
                f"artifact handle belongs to episode {owner!r}, not {self.episode_id!r}"
            )
        entry = self._entries.get(handle)
        if entry is None:
            raise ArtifactHandleError("unknown artifact handle")
        if entry.expires_at is not None and self._clock() > entry.expires_at:
            del self._entries[handle]
            raise ArtifactHandleError("artifact handle expired")
        if expected_kind is not None and entry.kind != expected_kind:
            raise ArtifactHandleError(
                f"artifact has kind {entry.kind!r}; expected kind {expected_kind!r}"
            )
        return self._objects.get(handle) if entry.stored else entry.value

    def close(self) -> None:
        self._entries.clear()
        self._closed = True
