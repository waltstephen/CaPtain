"""Sensor-motion backend retaining recovery routes from unclosed descents."""
from .backend import LiberoActionBackend
from .descent_recording import DescentRecordingMixin
from .possession_monitor import PossessionMonitorMixin
from .closure_force_monitor import ClosureForceMonitorMixin


class RecordedDescentActionBackend(DescentRecordingMixin, LiberoActionBackend):
    pass


class MonitoredPossessionActionBackend(PossessionMonitorMixin, RecordedDescentActionBackend):
    """Retain recorded recovery routes and sample possession during transport."""


class ClosingMonitoredPossessionActionBackend(ClosureForceMonitorMixin, MonitoredPossessionActionBackend):
    """Also monitor closing and return measured post-closure wrist force."""
