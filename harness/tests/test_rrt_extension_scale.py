import numpy as np
from captain_harness.adapters.robotics.planning import _rrt_candidate


def test_ompl_growth_edges_obey_existing_joint_sample_scale():
    edges=[];start=np.array([-.02,-.02]);goal=np.array([.04,.03]);step=.005
    def sweep(a,b):edges.append((a.copy(),b.copy()));return True
    path=_rrt_candidate(start,goal,np.array([[-.04,.06],[-.04,.06]]),lambda q:True,
        timeout_s=1.,max_joint_step=step,edge_valid=sweep)
    assert path is not None and edges
    np.testing.assert_array_equal(path[0],start)
    np.testing.assert_array_equal(path[-1],goal)
    assert max(np.linalg.norm(b-a) for a,b in edges)<=step+1e-12
