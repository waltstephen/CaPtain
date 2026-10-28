"""Direct commands expose geometry without choosing grasps; receipts survive failures."""
from types import SimpleNamespace
from dataclasses import dataclass
import numpy as np
import pytest
from captain_harness.adapters.capx.direct_programming import direct_manifest, transform_pose, DirectProgramming, PERCEPTION
from captain_harness.adapters.capx.manifest import build_capx_tool_manifest, ToolApiLevel
from captain_harness.ports.broker import ValidatedCapabilityBroker
from captain_harness.ports.capability import CapabilityResult, FailureCategory

def manifest():
    return direct_manifest(build_capx_tool_manifest(ToolApiLevel("S3"), joint_count=7,
        service_aligned=True, dehardcoded=True, v15_protocol=True))

def test_public_surface_and_mutually_exclusive_commands():
    m=manifest()
    assert m.schema_version == "captain-capabilities/r30-direct-v8"
    assert {s.name for s in m.capabilities} <= PERCEPTION | {"move_to","hold","recover","release","preview_motion","measure_image_points","transform_pose","robot_geometry","check_motion","inspect_contact_geometry","inspect_release_geometry","compare_object_observations","measure_object_geometry"}
    called=[]
    broker=ValidatedCapabilityBroker(m,lambda name,args: (called.append((name,args)) or CapabilityResult.ok({})))
    for args in ({}, {"position":[0,0,0]}, {"gripper":"close","position":[0,0,0],"quaternion":[1,0,0,0]},
                 {"gripper":"open","max_speed_m_s":.05}, {"position":[0,0,0],"quaternion":[1,0,0,0],"extra":1}):
        assert not broker.invoke("move_to",args).success
    assert not called
    assert broker.invoke("move_to",{"gripper":"close"}).success
    assert broker.invoke("move_to",{"position":[0,0,0],"quaternion":[1,0,0,0]}).success

def test_geometry_local_axes_and_explicit_pivot():
    q=[np.sqrt(.5),0,0,np.sqrt(.5)]
    r=transform_pose([1,0,0],q,translation=[1,0,0],frame="end_effector").value
    assert np.allclose(r["position"],[1,1,0])
    r=transform_pose([1,0,0],[1,0,0,0],rotation_vector=[0,0,np.pi/2],pivot=[0,0,0]).value
    assert np.allclose(r["position"],[0,1,0])
    with pytest.raises(ValueError): transform_pose([0,0,0],[0,0,0,0])

@dataclass
class State:
    position: tuple=(0.,0.,0.)
    end_effector_position_m: tuple=(0.,0.,0.)
    end_effector_quaternion_wxyz: tuple=(1.,0.,0.,0.)
    joint_positions_rad: tuple=(0.,)*7
    gripper_open_fraction: float=1.

class Robot:
    _control_frequency=20
    def _eef_force_norm(self): return 0.
    simulation_steps=0
    _gripper_command_fraction=1.
    def get_robot_state(self): return CapabilityResult.ok(State())
    def close_gripper(self):
        self.simulation_steps+=3
        return CapabilityResult.ok({"motion_precision":{"error_m":.002}})
    def hold_position(self,**args):
        self.simulation_steps+=2
        raise RuntimeError("camera disconnected after movement")
    def open_gripper(self):
        self.simulation_steps+=1
        return CapabilityResult.ok({})

def fake_tools(robot):
    model=SimpleNamespace(limits=np.array([[-3.,3.]]*7),collision=lambda *a:{'collision':False})
    return SimpleNamespace(robot=robot,_motion_model=lambda:model)

def test_physical_failure_preserves_attempt_actual_progress_and_precision():
    d=DirectProgramming(fake_tools(Robot()))
    a=d.invoke("move_to",{"gripper":"close"})
    assert a.executed and a.value["attempt_id"]
    assert a.attempt_identity["attempt_role"] == "origin"
    assert a.value["motion_precision"]["error_m"] == .002
    b=d.invoke("hold",{"duration_s":.1})
    assert b.executed and not b.success
    assert b.value["attempt_id"] == a.value["attempt_id"]
    assert b.attempt_evidence["attempt_reference"] == a.attempt_identity["attempt_reference"]
    assert b.attempt_evidence["status"] == "unknown"
    assert b.value["start_step"] == 3 and b.value["end_step"] == 5
    assert b.value["object_retention"] == "not_verified"

def test_direct_path_uses_real_ik_signature_and_collision_guard():
    from captain_harness.adapters.capx.motion_tools import pose_matrix
    @dataclass
    class PoseState:
        end_effector_position_m: tuple=(0.,0.,1.)
        end_effector_quaternion_wxyz: tuple=(1.,0.,0.,0.)
        joint_positions_rad: tuple=(0.,)*7
        gripper_open_fraction: float=1.
    class MotionRobot(Robot):
        _cartesian_max_steps=480
        def get_robot_state(self):return CapabilityResult.ok(PoseState())
        def _execute_contact_sweep(self,path,**options):
            assert self._motion_guard()
            assert options["precision_reference"] == "absolute_pose_goal"
            assert options["position_tolerance_m"] == .003
            assert options["orientation_tolerance_rad"] == .03
            self.simulation_steps+=1
            return CapabilityResult.ok({"position_residual_m":.001})
    class Model:
        calls=0
        self_checks=0
        limits=np.array([[-3.,3.]]*7)
        last_ik={}
        def collision(self,joints,aperture,points):
            self.self_checks+=1
            assert len(points)==0
            return {"collision":False,"observed_collisions":{},"self_collisions":[]}
        def solve_ik(self,pose,seed,*,aperture,position_tolerance,orientation_tolerance):
            self.calls+=1
            assert position_tolerance==.001 and orientation_tolerance==.01
            return np.zeros(7)
    robot=MotionRobot();model=Model()
    scene=SimpleNamespace(joints=[0.]*7,aperture=1.)
    calls=[]
    def collision(q):calls.append(q);return {"collision":False}
    tool=SimpleNamespace(robot=robot,observe=lambda:CapabilityResult.ok({"observation_handle":"obs"}),
        build_scene=lambda handle:CapabilityResult.ok({"scene_handle":"scene"}),
        _scene=lambda handle,**kwargs:scene,_motion_model=lambda:model,
        _scene_collision_checker=lambda *a,**kw:(collision,{}),
        _tracking_reserved_checker=lambda *a,**kw:(collision,{}))
    result=DirectProgramming(tool).invoke("move_to",{"position":[0,0,1.02],"quaternion":[1,0,0,0]})
    assert result.success, result.message
    assert model.calls>1 and model.self_checks>model.calls and not calls
    assert result.value["end_step"]==1
    assert robot._motion_guard is None

def test_geometry_uses_actual_pad_reference_and_excludes_legacy_ik_offset():
    class CalibratedRobot(Robot):
        def gripper_contact_envelope(self):
            return {"pad_min_local_m":[-.01,-.04,-.012],"pad_max_local_m":[.01,.04,.004],
                    "basis":"hardware kinematics","origin_frame":"end_effector","axes_frame":"end_effector",
                    "units":"meter","rotation_end_effector_from_axes":np.eye(3).tolist(),"closing_axis_local":[0,1,0]}
    tool=SimpleNamespace(robot=CalibratedRobot(),gripper_profile=SimpleNamespace(public=lambda:{"max_width_m":.08}),
                         _tcp_offset_local_m=[0,0,-.1])
    r=DirectProgramming(tool).invoke("robot_geometry",{})
    assert r.success
    assert "tcp_offset_local_m" not in r.value
    assert np.allclose(r.value["eef_to_pad_center_local_m"],[0,0,-.004])
    assert "gripper0_eef" in r.value["eef_reference"]

def test_post_action_measurement_exception_keeps_executed_receipt_and_attempt():
    class LostStateRobot(Robot):
        reads=0
        def get_robot_state(self):
            self.reads+=1
            if self.simulation_steps>0:raise RuntimeError("state stream failed")
            return CapabilityResult.ok(State())
    d=DirectProgramming(fake_tools(LostStateRobot()))
    r=d.invoke("move_to",{"gripper":"close"})
    assert r.executed and not r.success
    assert r.value["driver_success"] is False
    assert r.value["end_step"]==3 and r.value["robot_after"] is None
    assert "state stream failed" in r.value["robot_after_error"]
    assert r.attempt_identity["attempt_role"]=="origin"


def test_v4_manifest_rejects_ambiguous_handles_and_gripper_separation():
    m=manifest()
    calls=[]
    broker=ValidatedCapabilityBroker(m,lambda name,args:(calls.append(args) or CapabilityResult.ok({})))
    a="artifact://v1/a";b="artifact://v1/b"
    assert not broker.invoke("move_to",{"position":[0,0,0],"quaternion":[1,0,0,0],
        "target_detection_id":a,"target_detection_ids":[b]}).success
    assert not broker.invoke("move_to",{"gripper":"close","separate_from_contact":True}).success
    assert not broker.invoke("move_to",{"gripper":"close","carrying_detection_ids":[a]}).success
    assert not calls
    assert broker.invoke("check_motion",{"position":[0,0,0],"quaternion":[1,0,0,0],
        "target_detection_ids":[a,b],"separate_from_contact":True}).success
    move=next(spec for spec in m.capabilities if spec.name=="move_to")
    assert not {"target_detection_id","target_detection_ids","carrying_detection_id","carrying_detection_ids",
                "gripper_empty","separate_from_contact"} & set(move.arguments_schema["properties"])


def test_failed_optional_geometry_does_not_gate_physical_pose_or_change_it():
    from test_contact_separation import model as make_model
    from captain_harness.adapters.capx.motion_tools import MotionToolMixin, quaternion
    @dataclass
    class PoseState:
        end_effector_position_m: tuple
        end_effector_quaternion_wxyz: tuple
        joint_positions_rad: tuple
        gripper_open_fraction: float=1.
    hardware=make_model()
    class SimulatedDriver:
        simulation_steps=0
        _control_frequency=20
        _gripper_command_fraction=1.
        _cartesian_max_steps=480
        q=np.zeros(2)
        def get_robot_state(self):
            pose=hardware.eef_pose(self.q)
            return CapabilityResult.ok(PoseState(tuple(pose[:3,3]),tuple(quaternion(pose)),tuple(self.q)))
        def _execute_contact_sweep(self,path,**kwargs):
            for pose in path:
                self.q=np.array(pose[:2,3])
                assert self._motion_guard()
                self.simulation_steps+=1
            return CapabilityResult.ok({"position_residual_m":0.})
    robot=SimulatedDriver()
    scene=SimpleNamespace(joints=[0.,0.],aperture=1.,points=np.array([[-.021,0,0]]),margin_m=.01)
    class Tools(MotionToolMixin):
        def __init__(self):self.robot=robot
        def _motion_model(self):return hardware
        def observe(self):return CapabilityResult.ok({"observation_handle":"obs"})
        def build_scene(self,handle):return CapabilityResult.ok({"scene_handle":"scene"})
        def _scene(self,handle,**kwargs):return scene
    direct=DirectProgramming(Tools())
    args={"position":[.02,0,0],"quaternion":[1,0,0,0]}
    queried=direct.invoke("check_motion",args)
    assert queried.success
    assert not queried.value["geometry_check_passed"]
    assert queried.value["assessment_message"]=="observed_start_collision"
    assert not queried.value["physical_action_executed"] and robot.simulation_steps==0
    accepted=direct.invoke("move_to",args)
    assert accepted.success,accepted.message
    assert accepted.executed and accepted.value["end_step"]>0
    assert accepted.value["requested"]["position"]==args["position"]
    assert accepted.value["object_retention"]=="not_verified"
    np.testing.assert_allclose(robot.q,[.02,0],atol=1e-10)

def test_minimal_physical_path_retains_self_collision_and_force_stops():
    @dataclass
    class PoseState:
        end_effector_position_m: tuple=(0.,0.,.1)
        end_effector_quaternion_wxyz: tuple=(1.,0.,0.,0.)
        joint_positions_rad: tuple=(0.,)*7
        gripper_open_fraction: float=.3
    class Hardware:
        limits=np.array([[-3.,3.]]*7)
        last_ik={}
        blocked=False
        def collision(self,*args):
            return {"collision":self.blocked,"observed_collisions":{},
                    "self_collisions":[["arm","hand"]] if self.blocked else []}
        def solve_ik(self,pose,seed,**kwargs):return np.zeros(7)
    hardware=Hardware()
    class Driver(Robot):
        simulation_steps=0
        _gripper_command_fraction=0.
        _cartesian_max_steps=480
        mode="force"
        invoked=0
        def get_robot_state(self):return CapabilityResult.ok(PoseState())
        def _execute_contact_sweep(self,path,**options):
            self.invoked+=1
            assert options["max_force_n"]==18.
            assert options["position_tolerance_m"]==.003
            if self.mode=="force":
                return CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED,"initial_force_limit",executed=False)
            self.simulation_steps+=1;hardware.blocked=True
            assert not self._motion_guard()
            return CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED,"external_motion_guard",executed=True)
    driver=Driver()
    # No perception/selection/carrying tools exist in this fixture.
    direct=DirectProgramming(SimpleNamespace(robot=driver,_motion_model=lambda:hardware))
    args={"position":[0,0,.12],"quaternion":[1,0,0,0]}
    hardware.blocked=True
    collision=direct.invoke("move_to",args)
    assert not collision.executed and driver.invoked==0
    assert collision.message=="robot_start_self_collision"
    hardware.blocked=False
    force=direct.invoke("move_to",args)
    assert force.message=="initial_force_limit" and not force.executed
    assert driver.invoked==1
    driver.mode="guard"
    interrupted=direct.invoke("move_to",args)
    assert interrupted.executed and not interrupted.success
    assert interrupted.value["end_step"]==1
    assert interrupted.value["execution_stop"]["reason"]=="robot_self_collision_during_execution"
    assert interrupted.value["requested"]==args
    assert driver._motion_guard is None


def test_contact_mode_and_load_budget_are_accepted_only_with_pose_commands():
    m=manifest()
    calls=[]
    broker=ValidatedCapabilityBroker(m,lambda name,args: (calls.append((name,args)) or CapabilityResult.ok({})))
    assert broker.invoke("move_to",{"position":[0,0,0],"quaternion":[1,0,0,0],"contact_mode":"manipulation","load_budget_n":30}).success
    assert broker.invoke("move_to",{"position":[0,0,0],"quaternion":[1,0,0,0],"contact_mode":"guarded"}).success
    assert not broker.invoke("move_to",{"gripper":"close","contact_mode":"manipulation"}).success
    assert not broker.invoke("move_to",{"position":[0,0,0],"quaternion":[1,0,0,0],"contact_mode":"soft"}).success
    assert not broker.invoke("move_to",{"position":[0,0,0],"quaternion":[1,0,0,0],"load_budget_n":100}).success
    assert len(calls)==2
