"""Explicit contact endpoint tolerance; requests are not accuracy guarantees."""
import math


def contact_position_tolerance(value):
    if value is None:
        return .003
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not .0005 <= value <= .003:
        raise ValueError('position_tolerance_m must be finite and between 0.0005 and 0.003 m')
    return float(value)
