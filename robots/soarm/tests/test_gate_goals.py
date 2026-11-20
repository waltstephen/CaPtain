"""soarm success atoms -> completion gate goals, through the adapter's own goals_from_bddl conversion."""
from __future__ import annotations

from types import SimpleNamespace

from captain_harness.adapters.capx.completion_gate import goals_from_bddl

from soarm.episode import _gate_goals


def goals(atoms):
    raw = _gate_goals(SimpleNamespace(atoms=atoms))
    return None if raw is None else goals_from_bddl(raw)


def test_containment_atoms_map_to_gate_goals():
    for atom in ("inside", "in_region", "placed_at"):
        (goal,) = goals([(atom, ("cube_0", "bowl_0"), {})])
        assert (goal["target"], goal["destination"], goal["kind"]) == ("cube_0", "bowl_0", "containment")


def test_object_referenced_atoms_are_left_to_the_final_predicate():
    assert goals([("on_top", ("cube_0", "cube_1"), {})]) is None
    assert goals([("relation", ("cube_0", "cube_1", "left"), {})]) is None
    assert goals([("inside", ("cube_0", "bowl_0"), {}), ("on_top", ("cube_1", "cube_2"), {})]) is None


def test_unsupported_atom_disables_the_gate():
    assert goals([("lifted", ("cube_0",), {})]) is None
    assert goals([("inside", ("cube_0", "bowl_0"), {}), ("grouped", ("cube_0",), {})]) is None
