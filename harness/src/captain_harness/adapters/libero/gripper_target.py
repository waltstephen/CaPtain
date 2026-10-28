"""Explicit position-servo command replacement from public finger encoders.

This bridge deliberately writes only the robot driver's target accumulator and
its actuator command buffer. Dynamics state and model parameters are read-only.
(Design notes are not part of this release.)
"""
import xml.etree.ElementTree as ET
import numpy as np


def replace_gripper_target(env, measured_joints, *, max_open_fraction_correction=None):
    gripper=env.robots[0].gripper
    if (type(gripper).__module__!='robosuite.models.grippers.panda_gripper'
            or type(gripper).__name__!='PandaGripper'):
        raise ValueError('unsupported incremental gripper command codec')
    names=list(gripper.actuators);joints=list(gripper.joints)
    if len(names)!=2 or len(joints)!=2 or len(set(names))!=2 or len(set(joints))!=2:
        raise ValueError('unsupported gripper actuator/joint count')
    raw=np.asarray(measured_joints)
    measured=np.asarray(measured_joints,dtype=float)
    if raw.dtype.kind=='b' or measured.shape!=(2,) or not np.isfinite(measured).all():
        raise ValueError('expected two finite measured finger joint positions')
    model=env.sim.model
    indices=np.asarray([model.actuator_name2id(name) for name in names],dtype=int)
    root=ET.fromstring(gripper.get_xml())
    position_targets=[];declared_ranges=[];gains=[]
    for name in names:
        items=[e for e in root.findall('./actuator/*') if e.get('name')==name]
        if len(items)!=1 or items[0].tag!='position':
            raise ValueError('gripper requires declared position actuators')
        element=items[0]
        joint=element.get('joint')
        if joint not in joints or element.get('gear','1')!='1':
            raise ValueError('gripper requires unit direct joint transmission')
        position_targets.append(float(measured[joints.index(joint)]))
        declared_ranges.append([float(x) for x in element.get('ctrlrange','').split()])
        gains.append(float(element.get('kp','nan')))
    bounds=np.asarray(declared_ranges,dtype=float)
    target=np.asarray(position_targets,dtype=float)
    if (bounds.shape!=(2,2) or not np.isfinite(bounds).all()
            or not np.all(bounds[:,1]>bounds[:,0]) or not np.isfinite(gains).all()
            or not np.all(np.asarray(gains)>0)):
        raise ValueError('invalid declared position actuator calibration')
    projection_fraction=0.
    target_basis='exact_measured_joint_target'
    if max_open_fraction_correction is not None:
        if (isinstance(max_open_fraction_correction,bool)
                or not isinstance(max_open_fraction_correction,(int,float))
                or not np.isfinite(max_open_fraction_correction)
                or not 0 < max_open_fraction_correction <= .25):
            raise ValueError('invalid explicit release correction budget')
    if np.any(target<bounds[:,0]) or np.any(target>bounds[:,1]):
        if max_open_fraction_correction is None:
            raise ValueError('measured position outside command range; target not clipped')
        projected=np.clip(target,bounds[:,0],bounds[:,1])
        # The verified Panda codec opens the first finger positively and the
        # second negatively. Reorder by each actuator's declared joint binding.
        opening_sign=np.asarray([1. if joints.index(
            next(e for e in root.findall('./actuator/*') if e.get('name')==name).get('joint'))==0
            else -1. for name in names])
        correction=projected-target
        if np.any(correction*opening_sign < 0.):
            raise ValueError('release target projection would command closing')
        projection_fraction=float(np.max(np.abs(correction)/(bounds[:,1]-bounds[:,0])))
        if projection_fraction >= max_open_fraction_correction:
            raise ValueError('release target projection exhausts opening budget')
        target=projected
        target_basis='opening_only_command_range_projection'
    expected_joint_ids=[model.joint_name2id(e.get('joint')) for name in names
        for e in root.findall('./actuator/*') if e.get('name')==name]
    expected_gear=np.zeros((2,6));expected_gear[:,0]=1.
    expected_gain=np.zeros((2,10));expected_gain[:,0]=gains
    expected_bias=np.zeros((2,10));expected_bias[:,1]=-np.asarray(gains)
    comparisons=[(model.actuator_ctrlrange[indices],bounds),
        (model.actuator_trntype[indices],np.zeros(2)),
        (model.actuator_trnid[indices,0],expected_joint_ids),
        (model.actuator_gear[indices],expected_gear),
        (model.actuator_dyntype[indices],np.zeros(2)),
        (model.actuator_gaintype[indices],np.zeros(2)),
        (model.actuator_biastype[indices],np.ones(2)),
        (model.actuator_gainprm[indices],expected_gain),
        (model.actuator_biasprm[indices],expected_bias)]
    if any(np.shape(a)!=np.shape(b) or not np.allclose(a,b,rtol=0,atol=1e-12)
           for a,b in comparisons):
        raise ValueError('compiled gripper calibration disagrees with declared position servo')
    bias=bounds.mean(axis=1);weight=(bounds[:,1]-bounds[:,0])*.5
    normalized=(target-bias)/weight
    previous_normalized=np.asarray(gripper.current_action,dtype=float).copy()
    previous_target=np.asarray(env.sim.data.ctrl[indices],dtype=float).copy()
    if (previous_normalized.shape!=(2,) or not np.isfinite(previous_normalized).all()
            or not np.isfinite(previous_target).all()):
        raise ValueError('invalid existing gripper command state')
    receipt={'status':'replaced','joint_names':joints,'actuator_names':names,
        'measured_joint_positions_m':measured.tolist(),
        'previous_target_m':previous_target.tolist(),'target_m':target.tolist(),
        'previous_normalized_target':previous_normalized.tolist(),
        'normalized_target':normalized.tolist(),
        'basis':'public finger encoders and verified robot position actuator calibration',
        'physics_steps':0}
    if max_open_fraction_correction is not None:
        receipt.update(target_basis=target_basis, opening_projection_fraction=projection_fraction)
    # Explicit driver command operation, after every validation above. The
    # native format_action(0) will retain this target on every physics substep.
    gripper.current_action=normalized.copy()
    env.sim.data.ctrl[indices]=target
    return receipt


from contextlib import contextmanager

@contextmanager
def bounded_opening_commands(gripper, max_open_fraction_increase):
    """Limit every native servo substep after a verified measured-target reset.

    This bounds commanded position; dynamics can still overshoot between public
    samples. The caller retains force, aperture and arm-drift guards.
    """
    if (type(gripper).__module__!='robosuite.models.grippers.panda_gripper'
            or type(gripper).__name__!='PandaGripper'):
        raise ValueError('unsupported incremental gripper command codec')
    if (isinstance(max_open_fraction_increase,bool)
            or not isinstance(max_open_fraction_increase,(int,float))
            or not np.isfinite(max_open_fraction_increase)
            or not 0 < max_open_fraction_increase <= .25):
        raise ValueError('invalid release command budget')
    start=np.asarray(gripper.current_action,dtype=float).copy()
    if start.shape!=(2,) or not np.isfinite(start).all() or np.any(np.abs(start)>1.):
        raise ValueError('invalid verified gripper command state')
    limit=np.clip(start+2.*max_open_fraction_increase*np.array([1.,-1.]),-1.,1.)
    low=np.minimum(start,limit);high=np.maximum(start,limit)
    original=gripper.format_action
    owned='format_action' in gripper.__dict__
    owned_value=gripper.__dict__.get('format_action')
    def limited(action):
        requested=original(action)
        bounded=np.clip(np.asarray(requested,dtype=float),low,high)
        gripper.current_action=bounded.copy()
        return bounded
    gripper.format_action=limited
    try:
        yield {'maximum_open_fraction_increase':max_open_fraction_increase,
            'initial_normalized_target':start.tolist(),'limit_normalized_target':limit.tolist(),
            'scope':'native command range bound; physical overshoot remains sampled'}
    finally:
        if owned:gripper.format_action=owned_value
        else:del gripper.format_action
