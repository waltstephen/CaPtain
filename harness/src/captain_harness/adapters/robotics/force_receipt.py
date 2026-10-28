"""Aggregate public sensor reads across the complete physical request boundary."""
import math
from numbers import Real


def _finite_force(value):
    if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def finalize_force_receipt(value, *, initial_force_n, final_force_n, step, sample_time_s):
    # Retain the driver's earlier summary, whose sampling phase may differ.
    fields = ("initial_force_n", "observed_force_n", "max_observed_force_n",
              "force_decreased", "force_still_over_limit")
    driver = {key: value[key] for key in fields if key in value}
    initial, final = _finite_force(initial_force_n), _finite_force(final_force_n)
    readings = [initial, final]
    readings.extend(_finite_force(driver.get(key)) for key in fields[:3])
    for sample in value.get("trajectory", ()) or ():
        if isinstance(sample, dict):
            readings.append(_finite_force(sample.get("force_n")))
    finite = [force for force in readings if force is not None]
    value.update(
        driver_force_summary=driver,
        initial_force_n=initial,
        observed_force_n=final,
        max_observed_force_n=max(finite, default=None),
        force_decreased=None if initial is None or final is None else final < initial,
        force_still_over_limit=None if final is None else final >= float(value.get("force_limit_n") or 18.),
        final_force_sample={
            "phase": "after_driver_and_target_stop",
            "step": step,
            "sample_time_s": sample_time_s,
            "force_n": final,
            "source": "public wrist force sensor",
        },
        force_summary_scope=(
            "finite sampled forces including request boundaries and driver summary; "
            "same-step sensor refreshes may differ; inter-sample peaks are not bounded"
        ),
    )
