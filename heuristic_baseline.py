"""
Heuristic Baseline for SRE Incident Response OpenEnv.

Rule-based agent — no LLM, no API key required.
Scores are fully deterministic and reproducible.
Run this to verify grader correctness and establish a performance floor.

Strategy:
  Score each service by anomaly = 0.3*cpu + 0.3*mem + 0.4*err_rate*100
  Pick the most anomalous service as root cause.
  Infer failure type and action from dominant metric.

Usage:
  python heuristic_baseline.py
Output:
  Results table + heuristic_baseline_results.json
"""

import json
import sys


sys.path.insert(0, ".")

from models import SREAction
from scenarios import TASK_IDS
from server.sre_environment import SREEnvironment

N_SEEDS = 5


def heuristic_diagnose(obs_dict: dict) -> SREAction:
    metrics   = obs_dict.get("metrics", [])
    dep_graph = obs_dict.get("dependency_graph", {})

    anomaly = {}
    for m in metrics:
        svc = m["service"]
        cpu = m.get("cpu_percent", 0) or 0
        mem = m.get("memory_percent", 0) or 0
        err = m.get("error_rate", 0) or 0
        anomaly[svc] = cpu * 0.3 + mem * 0.3 + err * 100 * 0.4

    if anomaly:
        worst = max(anomaly, key=anomaly.get)
    else:
        worst = list(dep_graph.keys())[0] if dep_graph else "unknown"

    wm  = next((m for m in metrics if m["service"] == worst), {})
    cpu = wm.get("cpu_percent", 0) or 0
    mem = wm.get("memory_percent", 0) or 0
    err = wm.get("error_rate", 0) or 0

    if mem > 85:
        rct, act = "memory_leak",   "restart_service"
    elif cpu > 85:
        rct, act = "cpu_spike",     "scale_out"
    elif err > 0.3:
        rct, act = "cascade_failure","rollback"
    else:
        rct, act = "cascade_failure","noop"

    return SREAction(
        action_type        = "diagnose",
        root_cause_service = worst,
        root_cause_type    = rct,
        recommended_action = act,
        confidence         = 0.5,
        reasoning          = f"Heuristic: {worst} anomaly={anomaly.get(worst, 0):.1f}",
    )


def run():
    env     = SREEnvironment()
    results = {}

    print(f"\n{'='*70}")
    print("  SRE Incident Response — Heuristic Baseline (no LLM)")
    print(f"  {N_SEEDS} seeds per task | variant rotation tested")
    print(f"{'='*70}\n")

    for task_id in TASK_IDS:
        scores = []
        for seed_idx in range(N_SEEDS):
            seed = seed_idx * 13 + 7
            obs  = env.reset(task_id=task_id, seed=seed)
            a    = heuristic_diagnose(obs.model_dump() if hasattr(obs, 'model_dump') else obs.__dict__)
            o    = env.step(a)
            scores.append(round(float(o.reward or 0.0), 4))

        mean = sum(scores) / len(scores)
        results[task_id] = {
            "mean":   round(mean, 4),
            "min":    round(min(scores), 4),
            "max":    round(max(scores), 4),
            "scores": scores,
        }
        print(
            f"  {task_id:<35} "
            f"mean={mean:.4f}  "
            f"min={min(scores):.4f}  "
            f"max={max(scores):.4f}  "
            f"scores={scores}"
        )

    overall = sum(r["mean"] for r in results.values()) / len(results)
    print(f"\n  {'OVERALL':<35} mean={overall:.4f}")
    print(f"\n{'='*70}\n")

    with open("heuristic_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved → heuristic_baseline_results.json")


if __name__ == "__main__":
    run()
