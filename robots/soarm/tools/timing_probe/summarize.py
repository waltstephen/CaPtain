"""Summarise SOARM_TIMING_LOG files: calls, mean / p50 / p90 / max ms and total seconds per wrapped function.

    python tools/timing_probe/summarize.py <timing.jsonl> [...]
"""
import collections
import json
import sys


def main() -> None:
    calls = collections.defaultdict(list)
    for path in sys.argv[1:]:
        for line in open(path):
            row = json.loads(line)
            calls[row["fn"]].append(row["s"])
    for fn, v in sorted(calls.items(), key=lambda kv: -sum(kv[1])):
        v = sorted(v)
        print(f"{fn:60s} n={len(v):5d} mean {1000 * sum(v) / len(v):7.1f} ms  p50 {1000 * v[len(v) // 2]:7.1f}  "
              f"p90 {1000 * v[int(0.9 * (len(v) - 1))]:7.1f}  max {1000 * v[-1]:8.1f}  total {sum(v):7.1f} s")


if __name__ == "__main__":
    main()
