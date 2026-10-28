"""Session -> policy-server routing of policy_proxy (--route static | least-loaded | request-rr)."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import policy_proxy as PROXY  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_proxy_state():
    saved, routes = copy.deepcopy(PROXY.STATE), dict(PROXY.ROUTES)
    yield
    PROXY.STATE.clear(); PROXY.STATE.update(saved)
    PROXY.ROUTES.clear(); PROXY.ROUTES.update(routes)


def fresh(route: str, n: int = 4, group_size: int = 1):
    P = PROXY
    P.ROUTES.clear()
    for k in ("inflight", "route_seen", "rr"):
        P.STATE.pop(k, None)
    P.STATE.update({"upstreams": [f"http://u{i}" for i in range(n)], "group_size": group_size, "route": route,
                    "route_idle_s": 1800.0})
    return P


def test_static_default_is_the_old_candidate_spread():
    P = fresh("static", n=4, group_size=16)
    assert P.upstream_for("s3g2k5_x") == "http://u" + str((3 + 2 * 16 + 5) % 4)
    assert P.upstream_for("s3g2k5_x") == P.upstream_for("s3g2k5_x")          # sticky
    assert P.upstream_for("other") == P.upstream_for("other")                # hashed, sticky


def test_least_loaded_balances_sessions_and_stays_sticky():
    P = fresh("least-loaded", n=4)
    # 8 sessions whose static choice is all the same server (same round/group/candidate residue)
    sids = [f"s{4 * i}g0k0_e{i}" for i in range(8)]
    assert len({P._static_upstream(s, P.STATE["upstreams"]) for s in sids}) == 1
    first = [P.upstream_for(s) for s in sids]
    counts = {u: first.count(u) for u in P.STATE["upstreams"]}
    assert set(counts.values()) == {2}                                       # 8 sessions -> 2 per server
    assert [P.upstream_for(s) for s in sids] == first                         # pinned


def test_least_loaded_releases_on_delete():
    import asyncio
    P = fresh("least-loaded", n=2)
    a, b = P.upstream_for("s0g0k0_a"), P.upstream_for("s0g0k0_b")
    assert a != b
    asyncio.run(P.delete("s0g0k0_a"))
    assert P.upstream_for("s0g0k0_c") == a                                    # the freed server takes the next session


def test_least_loaded_breaks_ties_by_inflight():
    P = fresh("least-loaded", n=3)
    P.STATE["inflight"] = {"http://u0": 3, "http://u1": 0, "http://u2": 1}
    assert P.upstream_for("s9g9k9_new") == "http://u1"


def test_request_rr_has_no_affinity():
    P = fresh("request-rr", n=3)
    assert [P.upstream_for("s0g0k0_x") for _ in range(4)] == ["http://u0", "http://u1", "http://u2", "http://u0"]
