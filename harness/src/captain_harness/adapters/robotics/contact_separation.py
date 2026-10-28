"""Validate explicitly requested separation from existing measured clearance.

This module chooses no motion. It checks the supplied configurations against
the same detached robot and public samples. No environment/contact truth.
"""
from collections.abc import Mapping
import numpy as np
from scipy.spatial import cKDTree
from captain_harness.ports.deadline import check_deadline


def geometry_clearance(geometry, local):
    """Signed outside distance / normalized convex halfspace clearance."""
    if geometry.primitive == "sphere":
        return np.linalg.norm(local, axis=1) - geometry.size[0]
    if geometry.primitive == "capsule":
        return np.hypot(np.linalg.norm(local[:, :2], axis=1),
                        np.maximum(np.abs(local[:, 2])-geometry.size[1], 0.)) - geometry.size[0]
    if geometry.primitive == "cylinder":
        radial=np.linalg.norm(local[:, :2],axis=1)-geometry.size[0]
        axial=np.abs(local[:,2])-geometry.size[1]
        return np.hypot(np.maximum(radial,0.),np.maximum(axial,0.)) + np.minimum(np.maximum(radial,axial),0.)
    output=np.empty(len(local))
    for begin in range(0,len(local),256):
        check_deadline()
        block=local[begin:begin+256]
        values=np.full(len(block),-np.inf)
        for first in range(0,len(geometry.planes),64):
            faces=geometry.planes[first:first+64]
            values=np.maximum(values,np.max(np.einsum("ij,kj->ik",block,faces[:,:3],optimize=False)+faces[:,3],axis=1))
        output[begin:begin+len(block)]=values
    return output


class ContactSeparationChecker:
    """Allow only original gripper/payload clearance contacts to separate.

    Hardware geometry penetration, new scene contacts and robot self/arm
    collision remain failures. An observed payload is a partial convex proxy;
    its original overlap may be within the existing public scene uncertainty
    margin. It must not be mistaken for known physical penetration. After a sample clears its margin it cannot re-enter.
    Execution permits the existing3mm tracking allowance only while the sample
    stays outside physical geometry; planning permits numerical slack only.
    """
    numerical_slack_m=1e-5

    def __init__(self, model, joints, aperture, points, margin_m, *,
                 target_mask=None, contact_margin_m=.003, reserve_padding=None,
                 execution_clearance_slack_m=.003):
        self.model=model;self.aperture=float(aperture)
        self.points=np.asarray(points,dtype=float).reshape(-1,3).copy()
        if not np.isfinite(self.points).all():
            raise ValueError("separation needs finite public scene points")
        self.margin_m=float(margin_m);self.contact_margin_m=float(contact_margin_m)
        self.target_mask=np.zeros(len(self.points),dtype=bool) if target_mask is None else np.asarray(target_mask,dtype=bool)
        if self.target_mask.shape!=(len(self.points),):
            raise ValueError("target membership must match scene points")
        self.reserve=dict(reserve_padding or {})
        self.execution_slack=float(execution_clearance_slack_m)
        if min(self.margin_m,self.contact_margin_m,self.execution_slack)<0:
            raise ValueError("clearance margins must be nonnegative")
        self.tree=cKDTree(self.points)
        self.geometries=[g for g in model.geometries if g.collidable and g.moving]
        self.eligible={g.name for g in self.geometries if g.name.startswith(model.gripper_prefix)
                       or g.name==model.carried_geometry_name}
        self.baseline={}
        self.released={"planning":{},"execution":{}}
        model.set_state(joints,aperture)
        self.initial_self=model.collision(joints,aperture,np.empty((0,3)))["self_collisions"]
        self.initial_unsafe={}
        for geometry in self.geometries:
            indices,distances,limits=self._samples(geometry,self.reserve.get(geometry.name,0.))
            involved=distances<=limits
            if geometry.name not in self.eligible:
                if involved.any():self.initial_unsafe[geometry.name]=int(involved.sum())
                continue
            floor=(-self.margin_m if geometry.name==model.carried_geometry_name else -self.numerical_slack_m)
            penetration=distances < floor
            if penetration.any():self.initial_unsafe[geometry.name]=int(penetration.sum())
            initial=np.full(len(self.points),np.nan)
            initial[indices[involved]]=distances[involved]
            self.baseline[geometry.name]=initial
            for stage in self.released:
                self.released[stage][geometry.name]=np.zeros(len(self.points),dtype=bool)
        self.initial_sample_pairs=sum(int(np.isfinite(x).sum()) for x in self.baseline.values())

    def _samples(self, geometry, padding):
        center,rotation=self.model.geom_pose(geometry)
        maximum=max(self.margin_m,self.contact_margin_m)+padding
        indices=np.asarray(self.tree.query_ball_point(center,geometry.radius+2.*maximum),dtype=int)
        if not len(indices):return indices,np.empty(0),np.empty(0)
        limits=np.full(len(indices),self.margin_m+padding)
        if geometry.name.startswith(self.model.gripper_prefix):
            limits[self.target_mask[indices]]=self.contact_margin_m+padding
        distance=geometry_clearance(geometry,(self.points[indices]-center)@rotation)
        return indices,distance,limits

    def check(self, joints, extra_margin=0., first_hit_only=False, *, stage="planning"):
        if stage not in {"initial","planning","execution"}:raise ValueError("unknown separation stage")
        if isinstance(extra_margin,Mapping):
            if set(extra_margin)!={g.name for g in self.geometries}:
                raise ValueError("padding must name every moving collision geometry")
            padding={k:float(v) for k,v in extra_margin.items()}
        else:
            padding={g.name:float(extra_margin) for g in self.geometries}
        if any(not np.isfinite(v) or v<0 for v in padding.values()):raise ValueError("invalid geometry padding")
        if self.initial_unsafe or self.initial_self:
            return {"collision":True,"observed_collisions":dict(self.initial_unsafe),
                    "self_collisions":self.initial_self,"separation":{"status":"unsafe_initial_geometry"}}
        # Empty measured cloud still invokes all original detached self contacts.
        self_collision=self.model.collision(joints,self.aperture,np.empty((0,3)))
        if self_collision["collision"]:
            return {**self_collision,"separation":{"status":"robot_self_collision"}}
        hits={};reasons={};allowed_count=0;new_released={}
        tolerance=self.execution_slack if stage=="execution" else self.numerical_slack_m
        for geometry in self.geometries:
            check_deadline()
            indices,distances,limits=self._samples(geometry,padding[geometry.name])
            involved=distances<=limits
            contact_ids=indices[involved];current=distances[involved]
            if geometry.name not in self.eligible:
                if len(contact_ids):hits[geometry.name]=len(contact_ids);reasons[geometry.name]="arm_contact"
                continue
            initial=self.baseline[geometry.name][contact_ids]
            known=np.isfinite(initial)
            floor=(-self.margin_m if geometry.name==self.model.carried_geometry_name else -self.numerical_slack_m)
            physical_clear=current>=floor
            initial_clear=initial>=floor
            no_deepening=current>=initial-tolerance
            already_released=(np.zeros(len(contact_ids),dtype=bool) if stage=="initial"
                              else self.released[stage][geometry.name][contact_ids])
            allowed=known & physical_clear & initial_clear & no_deepening & ~already_released
            if (~allowed).any():
                hits[geometry.name]=int((~allowed).sum())
                reasons[geometry.name]={
                    "new_contacts":int((~known).sum()),
                    "geometry_overlap_beyond_bound":int((~physical_clear).sum()),
                    "initial_geometry_overlap_beyond_bound":int((known & ~initial_clear).sum()),
                    "increased_clearance_overlap":int((known & ~no_deepening).sum()),
                    "reentered_contacts":int(already_released.sum())}
            allowed_count+=int(allowed.sum())
            if stage!="initial":
                released=self.released[stage][geometry.name].copy()
                still_involved=np.zeros(len(self.points),dtype=bool);still_involved[contact_ids]=True
                released[np.isfinite(self.baseline[geometry.name]) & ~still_involved]=True
                new_released[geometry.name]=released
        collision=bool(hits)
        if not collision and stage!="initial":
            self.released[stage].update(new_released)
        return {"collision":collision,"observed_collisions":hits,"self_collisions":[],
                "separation":{"status":"blocked" if collision else "within_separation_constraints",
                    "initial_sample_pairs":self.initial_sample_pairs,"allowed_sample_pairs":allowed_count,
                    "reasons":reasons,"stage":stage,
                    "payload_proxy_uncertainty_m":self.margin_m,
                    "scope":"hardware penetration stays blocked; payload proxy overlap is bounded by the existing public scene uncertainty, not a physical-contact verdict; no retention or unseen-space certificate"}}
