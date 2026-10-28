"""Observe the declared EEF force sensor inside the unchanged native control cycle.

No hidden contacts, scene identities, physics/gain changes, skipped dynamics, or
extra reward/predicate reads. Crossing detection bounds reaction latency, NOT force.
"""
from contextlib import contextmanager
import math
import numpy as np


class SubstepForceMonitor:
    def __init__(self, *, initial_force, max_force, retreat, detector=None, collect_samples=False, rise_limit_n=5.):
        self.rise_limit_n=float(rise_limit_n)
        self.detector=detector
        self.collect_samples=bool(collect_samples)
        self.samples=[]
        self.initial_force=float(initial_force)
        self.reference=float(initial_force)
        self.max_force=float(max_force)
        self.retreat=bool(retreat)
        self.peak_force=float(initial_force)
        self.sample_count=0
        self.trigger=None
        self.cancellation=None
        self.sampling_interval_s=None
        self.final_force=initial_force
        from collections import deque
        self._recent=deque(maxlen=12)
        self.trigger_window=[]

    def observe(self, force, *, after_step, sample=None):
        if after_step:self.sample_count+=1
        self.final_force=force
        reason=None
        if self.collect_samples and sample is not None and sample.get("f_world") is not None:
            self.samples.append({"f_world":[float(v) for v in sample["f_world"]],
                "f_osc":[float(v) for v in sample.get("f_osc",(0.,0.,0.))],"after_step":after_step,
                **({"v_tick":[float(v) for v in sample["v_tick"]]} if sample.get("v_tick") is not None else {})})
        if force is None or not math.isfinite(force):
            reason="force_unavailable"
        else:
            self.peak_force=max(self.peak_force,force)
            if self.detector is not None and not self.retreat:
                # v5: calibrated residual detector decides; the fixed 18 N / +5 N rule is not applied.
                reason=self.detector.observe_substep(sample) if sample is not None else None
            elif self.retreat:
                if force>=self.reference+5.:reason="retreat_force_increased"
                self.reference=min(self.reference,force)
            elif force>=self.max_force or force-self.initial_force>=self.rise_limit_n:
                reason="contact_force"
        sample={"sample_index":self.sample_count,"force_n":force,
            "phase":"after_native_step" if after_step else "before_native_control"}
        self._recent.append(sample)
        if self.trigger is not None and len(self.trigger_window)<24:
            self.trigger_window.append(sample)
        if reason and self.trigger is None:
            self.trigger={"reason":reason,"force_n":force,"sample_index":self.sample_count,
                "reference_force_n":self.reference if self.retreat else self.initial_force}
            self.trigger_window=list(self._recent)
        return self.trigger

    def receipt(self):
        return {"sensor":"declared_eef_force","sampling_interval_s":self.sampling_interval_s,
            "sample_count":self.sample_count,"peak_force_n":self.peak_force,"final_force_n":self.final_force,
            "trigger":self.trigger,"trigger_window":self.trigger_window,"target_cancellation":self.cancellation,
            "scope":"native internal samples; finite reaction delay and inertial overshoot remain; no hard force cap"}


@contextmanager
def active_force_monitor(backend, monitor):
    old=getattr(backend,"_contact_substep_monitor",None)
    backend._contact_substep_monitor=monitor
    try:yield
    finally:backend._contact_substep_monitor=old


def monitored_step(env, action, monitor, *, read_force, cancel, read_sample=None):
    """Temporarily instrument native hooks; preserve env.step bookkeeping/dynamics."""
    step_env=env
    seen=set()
    while not callable(getattr(env,"_pre_action",None)) and id(env) not in seen:
        seen.add(id(env))
        nested=getattr(env,"env",None)
        if nested is None:break
        env=nested
    if not callable(getattr(env,"_pre_action",None)) or not callable(getattr(env,"_update_observables",None)):
        raise RuntimeError("native substep force hooks unavailable")
    # LIBERO wrappers delegate attributes; patch the method owner that actually
    # runs the native loop, not a shadow attribute on the outer wrapper.
    env=getattr(env._pre_action,"__self__",env)
    pre,update=env._pre_action,env._update_observables
    # Remove temporary instance methods afterwards when originals were class methods.
    had_pre="_pre_action" in vars(env)
    had_update="_update_observables" in vars(env)
    monitor.sampling_interval_s=float(env.model_timestep)
    def pre_action(values,policy_step=False):
        monitor.observe(read_force(),after_step=False,sample=read_sample() if read_sample is not None else None)
        if monitor.trigger is not None:
            if monitor.cancellation is None:
                monitor.cancellation=cancel()
                if not isinstance(monitor.cancellation,dict) or monitor.cancellation.get("status")!="cancelled":
                    raise RuntimeError("substep target cancellation unconfirmed")
                policy_step=True
            # Zero delta arm goal and neutral gripper integration. Existing holding
            # target is preserved. Native controller still computes each torque.
            values=np.zeros_like(values)
        return pre(values,policy_step=policy_step)
    def update_observables(*args,**kwargs):
        result=update(*args,**kwargs)
        monitor.observe(read_force(),after_step=True,sample=read_sample() if read_sample is not None else None)
        return result
    env._pre_action=pre_action
    env._update_observables=update_observables
    try:
        result=step_env.step(action)
        # A trigger on the final substep also cancels the target before returning.
        if monitor.trigger is not None and monitor.cancellation is None:
            monitor.cancellation=cancel()
            if not isinstance(monitor.cancellation,dict) or monitor.cancellation.get("status")!="cancelled":
                raise RuntimeError("substep target cancellation unconfirmed")
        return result
    finally:
        if had_pre:env._pre_action=pre
        else:del env._pre_action
        if had_update:env._update_observables=update
        else:del env._update_observables
