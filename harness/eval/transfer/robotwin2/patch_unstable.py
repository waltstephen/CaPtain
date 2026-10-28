#!/usr/bin/env python3
"""Second RoboTwin patch (after `git apply robotwin.patch`): ROBOTWIN_ALLOW_UNSTABLE=1 keeps a scene whose objects moved
during RoboTwin's settling check instead of raising UnStableError (the moved objects are recorded in
self.unstable_objects). The harness sets it for the official task files (`allow_unstable: true`): those seeds already
passed RoboTwin's own expert check with the official embodiment, and the dual-Franka scene is built from the same layout.

    python3 eval/transfer/robotwin2/patch_unstable.py $ROBOTWIN_ROOT
"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "envs" / "_base_task.py"
s = p.read_text()
old = """        is_stable, unstable_list = self.check_stable()
        if not is_stable:
            raise UnStableError("""
new = """        is_stable, unstable_list = self.check_stable()
        self.unstable_objects = [] if is_stable else list(unstable_list)
        # CaPtain: ROBOTWIN_ALLOW_UNSTABLE=1 keeps a scene whose objects moved during settling (recorded in
        # self.unstable_objects)
        if not is_stable and os.environ.get("ROBOTWIN_ALLOW_UNSTABLE") != "1":
            raise UnStableError("""
if "ROBOTWIN_ALLOW_UNSTABLE" in s:
    print("already patched")
else:
    assert old in s, "unexpected RoboTwin revision (expected ea8b211)"
    p.write_text(s.replace(old, new))
    print("patched", p)
