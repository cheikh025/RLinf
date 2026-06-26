"""Parse LIBERO eval logs into per-(task, trial) success, and diff arms vs K=1.

Two jobs, one script:

1. **Extract** (any eval log -> JSON): scrape the per-episode lines RLinf already
   emits (``rlinf/envs/libero/utils.py``:
   ``[libero eval] task_id=T, trial_id=R, success=True|False``) into a reusable
   results file: per-(task, trial) success, per-task rates, the **failure set
   F**, the **failing task ids** (paste into ``env.eval.task_id_filter``), and the
   flat **reset_state_ids** (for an exact ``reset_state_id_filter`` replay).

2. **Diff** (an arm's log vs the K=1 baseline JSON): on the SAME (task, trial)
   cells, report **recovery** (K=1 failed -> arm solved) and **regression**
   (K=1 solved -> arm failed). This is the failure-replay metric.

No project code is touched; this only reads logs. See
``dreamzero_robometer_progress_failure_replay_README.md`` for the full plan.

Examples
--------
    # 1. turn the K=1 run into the baseline + failure set
    python dreamzero_checks/parse_libero_eval.py k1/eval_embodiment.log \
        --out k1_failures.json

    # 2. score a best-of-K arm against that baseline
    python dreamzero_checks/parse_libero_eval.py exec-idm-progress/eval_embodiment.log \
        --baseline k1_failures.json --out exec-idm-progress_vs_k1.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Matches the line RLinf logs per completed eval episode (utils.py:264), after
# ANSI / rank-prefix stripping. trial_id is the per-task init-state index.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LINE = re.compile(
    r"\[libero eval\]\s*task_id=(\d+)\s*,\s*trial_id=(\d+)\s*,\s*success=(True|False)"
)


def parse_log(path: Path) -> dict[tuple[int, int], bool]:
    """Log -> {(task_id, trial_id): success}. First occurrence wins (eval counts
    each cell once); conflicting duplicates are warned, not silently merged."""
    results: dict[tuple[int, int], bool] = {}
    conflicts = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = _ANSI.sub("", raw)
            m = _LINE.search(line)
            if not m:
                continue
            key = (int(m.group(1)), int(m.group(2)))
            val = m.group(3) == "True"
            if key in results:
                if results[key] != val:
                    conflicts += 1
                continue
            results[key] = val
    if conflicts:
        print(f"  WARNING: {conflicts} (task,trial) cells had conflicting "
              f"success values; kept the first.", file=sys.stderr)
    if not results:
        print("  WARNING: no '[libero eval] task_id=..' lines found. Is this a "
              "finished LIBERO eval log?", file=sys.stderr)
    return results


def flat_reset_state_id(task_id: int, trial_id: int, trials_per_task: int) -> int:
    """Flat reset_state_id for uniform-trial suites: task*T + trial (LIBERO T=50).
    For non-uniform suites pass the right --trials-per-task or use task_id_filter."""
    return task_id * trials_per_task + trial_id


def summarize(results: dict[tuple[int, int], bool], trials_per_task: int) -> dict:
    per_task: dict[int, dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0})
    for (tid, _trial), ok in results.items():
        per_task[tid]["total"] += 1
        per_task[tid]["success"] += int(ok)
    failures = sorted(k for k, ok in results.items() if not ok)
    total = len(results)
    n_fail = len(failures)
    return {
        "n_episodes": total,
        "n_success": total - n_fail,
        "n_failures": n_fail,
        "success_rate": round((total - n_fail) / total, 4) if total else None,
        "trials_per_task": trials_per_task,
        "per_task": {str(t): per_task[t] for t in sorted(per_task)},
        "failing_task_ids": sorted({t for t, _ in failures}),
        "failures": [
            {"task_id": t, "trial_id": r,
             "reset_state_id": flat_reset_state_id(t, r, trials_per_task)}
            for (t, r) in failures
        ],
        # ready-to-paste replay selectors:
        "reset_state_ids": [flat_reset_state_id(t, r, trials_per_task) for (t, r) in failures],
        # full per-cell results (for downstream diffing), as [task, trial, success]:
        "episodes": [[t, r, results[(t, r)]] for (t, r) in sorted(results)],
    }


def load_baseline(path: Path) -> dict[tuple[int, int], bool]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(int(t), int(r)): bool(s) for t, r, s in data["episodes"]}


def diff(baseline: dict[tuple[int, int], bool],
         arm: dict[tuple[int, int], bool]) -> dict:
    """Recovery / regression of `arm` vs `baseline` on shared (task,trial) cells."""
    shared = sorted(set(baseline) & set(arm))
    base_fail = [k for k in shared if not baseline[k]]
    base_ok = [k for k in shared if baseline[k]]
    recovered = [k for k in base_fail if arm[k]]
    regressed = [k for k in base_ok if not arm[k]]

    def rate(num, den):
        return round(num / den, 4) if den else None

    return {
        "n_shared_cells": len(shared),
        "baseline_failures": len(base_fail),
        "baseline_successes": len(base_ok),
        "recovered": len(recovered),
        "recovery_rate": rate(len(recovered), len(base_fail)),
        "regressed": len(regressed),
        "regression_rate": rate(len(regressed), len(base_ok)),
        "net_solved_delta": len(recovered) - len(regressed),
        "recovered_cells": [{"task_id": t, "trial_id": r} for (t, r) in recovered],
        "regressed_cells": [{"task_id": t, "trial_id": r} for (t, r) in regressed],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("log", help="A LIBERO eval_embodiment.log to parse.")
    ap.add_argument("--out", default=None, help="Write the JSON result here.")
    ap.add_argument("--baseline", default=None,
                    help="A K=1 results JSON; report recovery/regression vs it.")
    ap.add_argument("--trials-per-task", type=int, default=50,
                    help="Trials per task for flat reset_state_id (LIBERO=50).")
    args = ap.parse_args()

    results = parse_log(Path(args.log))
    out = summarize(results, args.trials_per_task)
    print(f"parsed {args.log}")
    print(f"  episodes={out['n_episodes']}  success={out['n_success']}  "
          f"failures={out['n_failures']}  success_rate={out['success_rate']}")
    if out["failing_task_ids"]:
        print(f"  failing_task_ids ({len(out['failing_task_ids'])}): "
              f"{out['failing_task_ids']}")
        print(f"  -> coarse replay: +env.eval.task_id_filter="
              f"[{','.join(map(str, out['failing_task_ids']))}]")
        print(f"  -> exact replay : {len(out['reset_state_ids'])} reset_state_ids "
              f"(see JSON 'reset_state_ids')")

    if args.baseline:
        base = load_baseline(Path(args.baseline))
        d = diff(base, results)
        out["diff_vs_baseline"] = {"baseline": str(args.baseline), **d}
        print(f"\nvs baseline {args.baseline} (on {d['n_shared_cells']} shared cells):")
        print(f"  RECOVERY:   {d['recovered']}/{d['baseline_failures']} "
              f"failures solved  (rate {d['recovery_rate']})")
        print(f"  REGRESSION: {d['regressed']}/{d['baseline_successes']} "
              f"successes broken (rate {d['regression_rate']})")
        print(f"  NET solved delta: {d['net_solved_delta']:+d}")

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
