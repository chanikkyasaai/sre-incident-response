"""
SRE Incident Response — Inference Script
inference.py — must be at root directory
"""

import json, os, sys, signal
import requests
from openai import OpenAI

# ── Config — exactly as specified ─────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"

# Our live environment
ENV_BASE_URL = (
    os.environ.get("OPENENV_BASE_URL", "https://chane35-sre-incident-response.hf.space")
    .replace("wss://", "https://").replace("ws://", "http://").rstrip("/")
)

TASKS = [
    "easy_memory_leak",
    "easy_cpu_spike",
    "medium_db_contention",
    "medium_network",
    "hard_bad_deploy",
    "hard_cascade_oom",
]

SEEDS = [42, 55, 71]
EPISODE_TIMEOUT = 60
_MAX_STEPS = 7

_VALID_TYPES   = "memory_leak|cpu_spike|db_connection_exhaustion|lock_contention|cascade_failure|bad_deploy|network_partition"
_VALID_ACTIONS = "rollback|restart_service|scale_out|investigate_db|block_traffic|noop"


# ── Logging — exact format from sample ────────────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── HTTP client ────────────────────────────────────────────────────
class SREEnvClient:
    def __init__(self, base_url):
        self._url = base_url
        self._sid = None

    def reset(self, task_id, seed):
        r = requests.post(f"{self._url}/reset",
            json={"task_id": task_id, "seed": seed}, timeout=30)
        r.raise_for_status()
        d = r.json()
        self._sid = d["session_id"]
        return d["observation"]

    def step(self, action):
        r = requests.post(f"{self._url}/step",
            json={"session_id": self._sid, **action}, timeout=30)
        r.raise_for_status()
        return r.json()

    def close(self):
        pass


# ── Prompt ─────────────────────────────────────────────────────────
def build_prompt(obs):
    alerts  = "\n".join(
        f"  [{a['severity']}] {a['service']}: {a['message']}"
        for a in obs.get("alerts", []))
    logs    = "\n".join(
        f"  [{e['timestamp']}] [{e['level']}] {e['service']}: {e['message']}"
        for e in obs.get("logs", []))
    metrics = "\n".join(
        f"  {m['service']}: cpu={m['cpu_percent']}% mem={m['memory_percent']}% err={m['error_rate']:.1%} p99={m['latency_p99_ms']}ms"
        for m in obs.get("metrics", []))
    deps    = json.dumps(obs.get("dependency_graph", {}), indent=2)
    services = sorted(obs.get("dependency_graph", {}).keys())
    diag = ""
    d = obs.get("diagnostic_result")
    if d:
        diag = f"\nDIAGNOSTIC [{d['service']} / {d['type']}]:\n  {d['data']}"

    return f"""You are an SRE engineer diagnosing a production incident.
TASK: {obs.get("task_description", "")}
STEPS REMAINING: {obs.get("steps_remaining", 0)}

=== ALERTS ===
{alerts}

=== LOGS ===
{logs}

=== METRICS ===
{metrics}

=== DEPENDENCY GRAPH ===
{deps}
{diag}

REASONING:
1. Which services are alerting? These are victims, not root causes.
2. Trace upstream dependencies — find the common ancestor.
3. If metrics show cpu=0/mem=0/rps=0 on a service — it has CRASHED SILENTLY.
4. Check logs for deploy/config change events before the incident.
5. If failure type is ambiguous, run run_diagnostic on the suspect service.

Respond with ONLY a JSON object (no markdown):
Option A: {{"action_type": "run_diagnostic", "query_service": "<service>", "query_type": "recent_logs|metrics_history|connections"}}
Option B: {{"action_type": "diagnose", "root_cause_service": "<one of: {services}>", "root_cause_type": "<one of: {_VALID_TYPES}>", "recommended_action": "<one of: {_VALID_ACTIONS}>", "confidence": 0.0-1.0, "reasoning": "<text>"}}"""


# ── Episode ────────────────────────────────────────────────────────
def run_episode(client, task_id, seed):
    env = SREEnvClient(base_url=ENV_BASE_URL)
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        obs  = env.reset(task_id=task_id, seed=seed)
    except Exception as e:
        obs = {
            "task_description": f"Diagnose incident: {task_id}",
            "alerts": [{"service": "api-gateway", "severity": "P1", "message": "High error rate", "metric": "error_rate", "value": 0.45, "threshold": 0.05}],
            "logs": [{"timestamp": "T+00:00", "level": "ERROR", "service": "api-gateway", "message": "Upstream not responding"}],
            "metrics": [{"service": "api-gateway", "cpu_percent": 80, "memory_percent": 70, "error_rate": 0.45, "latency_p99_ms": 3000, "requests_per_second": 100}],
            "dependency_graph": {"api-gateway": ["backend"], "backend": []},
            "steps_remaining": 3, "diagnostic_result": None, "action_feedback": "", "done": False,
        }

    done = obs.get("done", False)
    log_start(task=task_id, env="sre-incident-response", model=MODEL_NAME)

    for step in range(1, _MAX_STEPS + 1):
        if done:
            break

        action_dict = None
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": build_prompt(obs)}],
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
            action_dict = json.loads(raw)
        except Exception as e:
            print(f"[DEBUG] LLM error: {e}", file=sys.stderr, flush=True)

        if action_dict is None:
            svcs = sorted(obs.get("dependency_graph", {}).keys())
            action_dict = {"action_type": "diagnose",
                "root_cause_service": svcs[0] if svcs else "unknown",
                "root_cause_type": "cascade_failure",
                "recommended_action": "noop", "confidence": 0.05, "reasoning": "fallback"}

        reward = 0.0
        error  = None
        try:
            result = env.step(action_dict)
            obs    = result["observation"]
            done   = result.get("done", False)
            reward = float(result.get("reward") or 0.0)
        except Exception as e:
            error = str(e)[:100]
            done  = True

        steps_taken = step
        rewards.append(reward)
        log_step(step=step, action=action_dict.get("action_type","diagnose"),
                 reward=reward, done=done, error=error)
        if done:
            score = reward
            break

    success = score >= 0.5
    try: env.close()
    except: pass
    # Score must be strictly between 0 and 1 (exclusive) — not 0.0 or 1.0
    score = max(0.001, min(0.999, score))
    rewards_clamped = [max(0.001, min(0.999, r)) if r > 0 else 0.001 for r in rewards]
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards_clamped)
    return score


# ── Main ───────────────────────────────────────────────────────────
def main():
    import signal

    def _timeout(signum, frame):
        raise TimeoutError("episode timeout")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    all_scores = []

    for task_id in TASKS:
        for seed in SEEDS:
            print(f"[DEBUG] task={task_id} seed={seed}", file=sys.stderr, flush=True)
            try:
                signal.signal(signal.SIGALRM, _timeout)
                signal.alarm(EPISODE_TIMEOUT)
            except AttributeError:
                pass
            try:
                s = run_episode(client, task_id, seed)
                all_scores.append(s)
            except Exception as e:
                print(f"[DEBUG] episode error: {e}", file=sys.stderr, flush=True)
                log_start(task=task_id, env="sre-incident-response", model=MODEL_NAME)
                log_end(success=False, steps=0, score=0.0, rewards=[])
                all_scores.append(0.0)
            finally:
                try: signal.alarm(0)
                except: pass

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"[DEBUG] avg_score={avg:.4f}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
