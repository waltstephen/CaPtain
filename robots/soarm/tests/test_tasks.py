"""Task generator invariants: unique references, layouts inside the reachable workspace, no goal satisfied at the start
(physics settle), deterministic per seed."""
import os

import numpy as np
import pytest

from soarm.families import FAMILIES
from soarm.objects import describe
from soarm.tasks import R_MAX, R_MIN, sample_task

SEEDS = range(6)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_sampling_is_deterministic_and_inside_workspace(family):
    for seed in SEEDS:
        a, b = sample_task(family, seed), sample_task(family, seed)
        assert a.to_json() == b.to_json()
        names = {o.name for o in a.objects}
        assert set(a.layout) == names
        for name, p in a.layout.items():
            r = float(np.hypot(*p.xy))
            assert R_MIN - 0.06 <= r <= R_MAX + 0.06, (name, r)
        for _, args, _ in a.atoms:
            assert all(arg in names for arg in args if isinstance(arg, str) and "_" in arg)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_referenced_objects_have_unique_descriptions(family):
    for seed in SEEDS:
        t = sample_task(family, seed)
        for role, name in t.roles.items():
            for n in name.split(","):
                spec = next((o for o in t.objects if o.name == n), None)
                if spec is not None and family not in ("sort_by_color", "sort_by_shape", "gather", "put_all_in_container",
                                                       "line_up", "clear_region"):
                    assert describe(spec, t.objects) is not None, (role, n)


needs_menagerie = pytest.mark.skipif(not os.environ.get("SOARM_MENAGERIE"), reason="SOARM_MENAGERIE not set")


@needs_menagerie
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_physics_start_has_no_goal_satisfied(family):
    from soarm.sim import SoarmSim
    for seed in range(3):
        sim = SoarmSim(family, seed)
        assert not any(sim.atom_values()), (family, seed)
        assert not sim.held()
