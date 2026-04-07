---
title: SRE Incident Response
emoji: 🚨
colorFrom: red
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
---

# 🚨 SRE Incident Response — OpenEnv

> A real-world RL environment where an AI agent acts as an on-call SRE engineer. The agent receives production alerts, logs, and service metrics — then investigates and diagnoses the root cause of live incidents.

**This models a task that production engineers do every day. Wrong diagnosis = extended outage.**

---

## What Makes This Environment Distinctive

### 1. Partial Observability (hard tasks)
Hard tasks withhold critical evidence from the initial observation. The deploy log, secret rotation event, or memory growth history that confirms the root cause mechanism only surfaces via a targeted `run_diagnostic` query. Without it, an agent can still score up to 0.70 by correctly reasoning from visible signals — but investigation is rewarded with the full 1.00. This models real on-call work: an experienced SRE can make a confident call from context alone, but a thorough SRE who checks the deploy history is doing the right thing.

### 2. Scenario Family Rotation (18 unique configurations)
Each of the 6 tasks has 3 variants selected by `seed % 3`. Same topology, different root cause service each rotation. An RL agent trained on 1000 seeds cannot memorize the answer — it must learn the reasoning pattern.

| Task | Variant 0 | Variant 1 | Variant 2 |
|------|-----------|-----------|-----------|
| `easy_memory_leak` | payment-service | auth-service | worker-service |
| `easy_cpu_spike` | image-processor | report-generator | ml-training-job |
| `medium_db_contention` | postgres-primary | message-broker | redis-cluster |
| `medium_network` | service-mesh | load-balancer | dns-resolver |
| `hard_bad_deploy` | config-service | secret-manager | feature-flag-service |
| `hard_cascade_oom` | ml-inference | stream-processor | log-aggregator |

### 3. Time-Pressure Penalty (models real MTTR pressure)
```
final_score = raw_score × max(0.70, 1.0 - 0.06 × max(0, diag_steps - 1))
```
First diagnostic: free. Second: −6%. Third: −12%. Etc. An agent learns to commit when confident and investigate only when necessary — exactly the on-call tradeoff.

### 4. Distinct Reasoning Skills Per Difficulty
- **Easy**: Tests log reading (heap growth timeline) and alert correlation (CPU + queue depth)
- **Medium**: Tests dependency graph tracing and temporal reasoning about infrastructure events
- **Hard**: Tests investigation under partial observability — rewards agents that find hidden evidence

### 5. Clean Red Herrings
Red herrings in hard tasks do not announce themselves. A postgres replication lag log appears during an incident window. A certificate renewal happened an hour earlier. A previous deploy completed successfully. The agent must reason about temporal correlation and dependency direction to dismiss them — not read a label.

---

## Tasks

| Task | Difficulty | What It Tests |
|------|-----------|--------------|
| `easy_memory_leak` | Easy | Log reading: follow heap growth timeline |
| `easy_cpu_spike` | Easy | Alert correlation: CPU alert + batch job log |
| `medium_db_contention` | Medium | Dependency graph: find shared upstream resource |
| `medium_network` | Medium | Temporal reasoning: infrastructure event correlation |
| `hard_bad_deploy` | Hard | Partial observability: config/secret layer failure |
| `hard_cascade_oom` | Hard | Partial observability: upstream memory leak cascade |

---

## Episode Structure

```
reset(task_id, seed)          → initial observation
step(run_diagnostic)          → deeper telemetry for one service (optional, costs time)
step(run_diagnostic)          → repeat as needed
step(diagnose)                → final answer → score → done
```

Max 7 steps per episode. Hard tasks: score × 0.70 without the hidden diagnostic, × 1.00 with it.

---

## Observation Space

```python
@dataclass
class SREObservation(Observation):
    alerts:            List[Alert]           # active alerts: severity/service/metric/value/threshold
    logs:              List[LogEntry]        # recent logs across all services
    metrics:           List[ServiceMetric]   # cpu/mem/error_rate/latency/rps per service
    dependency_graph:  Dict[str, List[str]]  # service → downstream dependencies
    task_description:  str
    steps_remaining:   int
    diagnostic_result: Optional[Dict]        # populated after run_diagnostic
    action_feedback:   str
    done:              bool
    reward:            float
```

---

## Action Space

### Run Diagnostic
```json
{"action_type": "run_diagnostic", "query_service": "<service>", "query_type": "recent_logs|metrics_history|connections"}
```

### Diagnose (ends episode)
```json
{
  "action_type": "diagnose",
  "root_cause_service": "<service from dependency_graph>",
  "root_cause_type": "memory_leak|cpu_spike|db_connection_exhaustion|lock_contention|cascade_failure|bad_deploy|network_partition",
  "recommended_action": "rollback|restart_service|scale_out|investigate_db|block_traffic|noop",
  "confidence": 0.0-1.0,
  "reasoning": "<chain of thought>"
}
```

---

## Reward Function

| Component | Weight | Notes |
|-----------|--------|-------|
| `root_cause_service` | 0.45 | Binary |
| `root_cause_type` | 0.30 | Partial credit (0.12) for `db_connection_exhaustion` ↔ `lock_contention` |
| `recommended_action` | 0.25 | Partial credit (0.10) for safer alternatives |
| Time penalty | modifier | `× max(0.70, 1.0 - 0.06*(diag_steps-1))` — first diagnostic free |
| Observability curve | modifier | Hard tasks: `raw_score × 0.70` without hidden diagnostic, `× 1.00` with it |

No confidence calibration — removed to prevent circular reward hacking incentives.

**Observability curve design**: A smart agent that correctly deduces the root cause from the initial observation (dependency graph + log correlation) earns 70% of the raw score. An agent that *also* runs the hidden diagnostic — finding the precise failure mechanism — earns 100%. Correct reasoning without investigation is rewarded (0.70×), not penalised. Investigation is incentivised (1.00×), not mandated.

**Example scores on `hard_bad_deploy`**:
- Wrong service, no diagnostic: 0.00
- Correct service only, no diagnostic: 0.45 × 0.70 = 0.315
- Perfect answer, no diagnostic: 1.00 × 0.70 = 0.70
- Perfect answer, with hidden diagnostic: 1.00 × 1.00 = 1.00

---

## Baseline Scores

**Heuristic agent** (rule-based, no LLM): picks most anomalous service by `0.3×cpu + 0.3×mem + 0.4×err`. 5 seeds per task.

| Task | Heuristic Mean | Min | Max | Notes |
|------|---------------|-----|-----|-------|
| `easy_memory_leak` | 1.0000 | 1.00 | 1.00 | Alert points directly to root cause |
| `easy_cpu_spike` | 1.0000 | 1.00 | 1.00 | Alert points directly to root cause |
| `medium_db_contention` | 0.4500 | 0.45 | 0.45 | Root cause not in alerts — must trace dependency graph |
| `medium_network` | 0.1500 | 0.00 | 0.25 | Infrastructure services have healthy metrics |
| `hard_bad_deploy` | 0.1750 | 0.175 | 0.175 | Root cause is silent — no alerts, zero error rate |
| `hard_cascade_oom` | 0.1050 | 0.00 | 0.175 | Root cause crashed (all-zero metrics, no logs in initial obs) |
| **Overall** | **0.4800** | | | |

`hard_bad_deploy` scores 0.175: the heuristic picks the loudest alerting service, not the quiet upstream config-layer root cause. It pattern-matches the correct type and action, earning partial credit multiplied by the 0.70 observability floor. `hard_cascade_oom` scores 0.105: the root cause has crashed (all-zero metrics, no error logs in initial observation). The heuristic sees nothing anomalous about a silent service and guesses wrong. Variants where the heuristic randomly picks the correct action score 0.175; others score 0. An LLM that traces dependency graphs, identifies the silent upstream service, and runs the hidden diagnostic will score 1.00 on both hard tasks.

---

## Quickstart

```python
from server.sre_environment import SREEnvironment
from models import SREAction

env = SREEnvironment()
obs = env.reset(task_id="hard_bad_deploy", seed=42)

# Run a targeted diagnostic
obs = env.step(SREAction(
    action_type="run_diagnostic",
    query_service="config-service",
    query_type="recent_logs"
))
print(obs.diagnostic_result["data"])  # reveals the hidden deploy log

# Diagnose
obs = env.step(SREAction(
    action_type="diagnose",
    root_cause_service="config-service",
    root_cause_type="bad_deploy",
    recommended_action="rollback",
    confidence=0.92,
    reasoning="Deploy v2.4.1 renamed config keys. Auth and payment fail on lookup."
))
print(obs.reward)     # 1.0
print(obs.metadata)   # full breakdown
```

---

## HTTP API

```bash
# Start episode
curl -X POST https://your-space.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard_bad_deploy", "seed": 42}'

# Run diagnostic
curl -X POST https://your-space.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "action_type": "run_diagnostic", "query_service": "config-service", "query_type": "recent_logs"}'

# Diagnose
curl -X POST https://your-space.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "action_type": "diagnose", "root_cause_service": "config-service", "root_cause_type": "bad_deploy", "recommended_action": "rollback", "confidence": 0.9, "reasoning": "..."}'
```

---

## Project Structure

```
sre-incident-response/
├── __init__.py                  # pip-installable package exports
├── pyproject.toml               # setuptools metadata
├── models.py                    # SREAction, SREObservation, SREState
├── scenarios.py                 # 6 families × 3 variants, partial observability engine
├── grader.py                    # deterministic grader: time penalty, partial obs cap
├── server/
│   ├── __init__.py
│   ├── sre_environment.py       # multi-step Environment: tracks diag_steps, queried_hidden
│   └── app.py                   # FastAPI server: TTL sessions, PYTHONPATH-safe
├── inference.py                 # agent loop: [START][STEP][END], rotating seeds
├── heuristic_baseline.py        # reproducible no-LLM baseline
├── openenv.yaml                 # spec manifest
├── Dockerfile                   # PYTHONPATH set, HEALTHCHECK included
├── requirements.txt
└── README.md
```

---

## Notes for Evaluators

**Graders are 100% deterministic.** No LLM-as-judge. Run `python heuristic_baseline.py` to reproduce the baseline scores exactly.

**Hard tasks require investigation.** An agent that diagnoses `hard_bad_deploy` without querying `config-service` (or `secret-manager`/`feature-flag-service` in other variants) scores max 0.42. The hidden evidence is the only way to identify the precise failure mechanism with confidence.

**Time pressure is real.** Three extra diagnostic queries cost 12% of final score. An agent that investigates efficiently beats one that queries everything.

**18 distinct episode configurations.** 6 tasks × 3 variants. An agent trained on all seeds cannot memorize answers — it learns patterns.

**Sessions are TTL-isolated.** Each `/reset` creates a new session. Sessions expire after 30 minutes of inactivity. Concurrent judge evaluations are fully isolated.
