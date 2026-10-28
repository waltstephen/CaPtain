"""Sampled, monotonic opening from existing self-contact; no arm escape planning."""
import numpy as np

def penetration_map(hit):
    if hit.get("observed_collisions"):
        raise ValueError("scene collision cannot be admitted as hardware separation")
    result={}
    for row in hit.get("self_collision_details", ()):
        pair=tuple(sorted(row["geometries"]))
        depth=float(row["penetration_m"])
        if not np.isfinite(depth) or depth < 0:raise ValueError("invalid collision depth")
        result[pair]=max(result.get(pair,0.),depth)
    if hit.get("collision") and not result:
        raise ValueError("collision lacks measured pair depths")
    return result

class MonotonicCollisionRelease:
    def __init__(self, model, joints, aperture, increase):
        self.model=model
        self.aperture=float(aperture)
        self.initial=penetration_map(model.collision(joints,aperture,np.empty((0,3))))
        if not self.initial:raise ValueError("no collision to separate")
        best=dict(self.initial)
        end=min(1.,self.aperture+increase)
        if end<=self.aperture:raise ValueError("no opening range")
        self.preview=[]
        for a in np.linspace(self.aperture,end,17):
            now=penetration_map(model.collision(joints,float(a),np.empty((0,3))))
            self._check(best,now)
            best={pair:min(depth,now.get(pair,0.)) for pair,depth in best.items()}
            self.preview.append({"aperture":float(a),"max_penetration_m":max(now.values(),default=0.)})
        if best and not any(best[p]<self.initial[p]-1e-6 for p in best):
            raise ValueError("opening does not predict reduced collision")
        self.best=dict(self.initial)
        self.remaining=dict(self.initial)

    @staticmethod
    def _check(best, now):
        if any(pair not in best for pair in now):raise ValueError("new self-collision pair")
        if any(depth>best[pair]+1e-6 for pair,depth in now.items()):
            raise ValueError("self-collision worsened or reentered")

    def observe(self, hit, aperture):
        if not np.isfinite(aperture) or aperture<self.aperture-.001:
            raise ValueError("opening reversed")
        now=penetration_map(hit)
        self._check(self.best,now)
        self.best={pair:min(depth,now.get(pair,0.)) for pair,depth in self.best.items()}
        self.aperture=max(self.aperture,float(aperture))
        self.remaining=now

    def receipt(self):
        return {"predicted_opening_samples":self.preview,
            "remaining_pairs":[{"geometries":list(p),"penetration_m":d} for p,d in self.remaining.items()],
            "scope":"sampled detached hardware opening plus live pair/force/pose checks; no task success claim"}
