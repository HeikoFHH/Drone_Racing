"""The committed demo website: its recordings must be consistent (stop branches continue the main run and end landed)."""

import json
import re
from pathlib import Path

HTML = Path(__file__).resolve().parent.parent / "docs" / "sign_demo" / "index.html"


def load():
    m = re.search(r"const D=(\{.*\});const \$=", HTML.read_text(), re.S)
    return json.loads(m.group(1))


def test_demo_site_has_both_policies_and_stop_branches_that_continue_the_flight():
    D = load()
    assert set(D["policies"]) == {"approach", "out_and_back"} and abs(D["sign_w"] - 22.5 * 0.0254) < 1e-9
    assert len(D["runs"]) >= 4
    for key, run in D["runs"].items():
        assert run["phase"] == "LANDED" and run["branches"], key
        assert abs(run["result"]["final_range_m"] - 0.5) < 0.1, key                         # stops 0.5 m in front
        rows = run["rows"]
        for b in run["branches"]:
            earlier = [r for r in rows if r[0] < b["t"] - 1e-6]
            if not earlier:
                continue                                                                          # pressed during take-off
            prev, first = earlier[-1], b["rows"][0]
            if first[3] < 0.05:
                continue                                                                          # pressed during the final touchdown
            assert abs(prev[1] - first[1]) < 0.15 and abs(prev[3] - first[3]) < 0.15, (key, b["t"])   # same place when STOP is pressed
            assert b["phase"] in ("LANDED", "ABORTED") and b["rows"][-1][3] < 0.05, (key, b["t"])      # and it ends on the ground
    ob = D["runs"]["out_and_back_8"]
    assert abs(ob["result"]["out_distance_m"] - ob["result"]["returned_m"]) < 0.4
