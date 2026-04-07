"""
SRE Incident Response — Environment

Key behaviours:
- Tracks which diagnostic queries were run (for time-pressure penalty)
- Tracks whether the hidden_diagnostic service+type was queried (for partial obs curve)
- Passes both to the grader at diagnosis time
- MAX_STEPS = 7: enough room to run 2 diagnostics and diagnose
- Invalid diagnoses do NOT consume a step
"""

import uuid
from typing import Any, Dict, Optional

from models import SREAction, SREObservation, SREState, VALID_ROOT_CAUSE_TYPES, VALID_ACTIONS
from scenarios import get_scenario, TASK_IDS
from grader import grade

MAX_STEPS        = 7
DIAGNOSTIC_STEPS_FREE = 1   # first diagnostic costs nothing in penalty


class SREEnvironment:

    def __init__(self):
        self._episode_id:      Optional[str]       = None
        self._task_id:         Optional[str]       = None
        self._scenario:        Optional[Dict]      = None
        self._step_count:      int                 = 0
        self._diag_steps:      int                 = 0   # diagnostic steps taken
        self._done:            bool                = False
        self._queried:         set                 = set()   # "svc:type" pairs
        self._queried_hidden:  bool                = False
        self._cumulative_reward: float             = 0.0

    # ── OpenEnv interface ──────────────────────────────────────────

    def reset(self, task_id: str = "easy_memory_leak", seed: int = 42) -> SREObservation:
        if task_id not in TASK_IDS:
            raise ValueError(f"Unknown task '{task_id}'. Valid: {TASK_IDS}")
        self._episode_id       = str(uuid.uuid4())[:8]
        self._task_id          = task_id
        self._scenario         = get_scenario(task_id, seed)
        self._step_count       = 0
        self._diag_steps       = 0
        self._done             = False
        self._queried          = set()
        self._queried_hidden   = False
        self._cumulative_reward = 0.0
        return self._build_obs(feedback="Episode started. Analyse the incident signals.")

    def step(self, action: SREAction) -> SREObservation:
        if self._done:
            raise RuntimeError("Episode complete. Call reset() to start a new episode.")
        if self._scenario is None:
            raise RuntimeError("Must call reset() before step().")

        if action.action_type == "run_diagnostic":
            return self._handle_diagnostic(action)
        elif action.action_type == "diagnose":
            return self._handle_diagnosis(action)
        else:
            # Unknown action type — don't burn a step
            obs = self._build_obs(feedback=(
                f"Unknown action_type '{action.action_type}'. "
                "Use 'run_diagnostic' or 'diagnose'."
            ))
            obs.reward = 0.0
            obs.done   = False
            return obs

    def state(self) -> SREState:
        return SREState(
            episode_id      = self._episode_id,
            step_count      = self._step_count,
            task_id         = self._task_id or "",
            difficulty      = self._scenario["difficulty"] if self._scenario else "",
            is_diagnosed    = self._done,
            current_score   = round(self._cumulative_reward, 4),
            max_steps       = MAX_STEPS,
            steps_used      = self._step_count,
        )

    def close(self):
        pass

    # ── Internal handlers ─────────────────────────────────────────

    def _handle_diagnostic(self, action: SREAction) -> SREObservation:
        dep_graph = self._scenario["dependency_graph"]
        diag_data = self._scenario.get("diagnostic_data", {})

        # Validate
        if not action.query_service:
            obs = self._build_obs(feedback="run_diagnostic requires query_service.")
            obs.reward, obs.done = 0.0, False
            return obs
        if action.query_service not in dep_graph:
            obs = self._build_obs(feedback=(
                f"Unknown service '{action.query_service}'. "
                f"Services: {sorted(dep_graph.keys())}"
            ))
            obs.reward, obs.done = 0.0, False
            return obs
        if action.query_type not in ("recent_logs", "metrics_history", "connections"):
            obs = self._build_obs(feedback="query_type must be: recent_logs | metrics_history | connections")
            obs.reward, obs.done = 0.0, False
            return obs

        # Consume a step
        self._step_count += 1
        self._diag_steps += 1

        key = f"{action.query_service}:{action.query_type}"
        self._queried.add(key)

        # Check if this query reveals the hidden evidence
        hidden = self._scenario.get("hidden_diagnostic")
        if (hidden
                and action.query_service == hidden["service"]
                and action.query_type    == hidden["query_type"]):
            self._queried_hidden = True

        svc_data    = diag_data.get(action.query_service, {})
        result_text = svc_data.get(
            action.query_type,
            f"No detailed {action.query_type} data available for {action.query_service}."
        )

        diag_result = {
            "service": action.query_service,
            "type":    action.query_type,
            "data":    result_text,
        }

        # Check max steps exhausted after this diagnostic
        if self._step_count >= MAX_STEPS and not self._done:
            self._done = True
            obs = self._build_obs(
                diagnostic_result=diag_result,
                feedback="MAX STEPS REACHED — episode ending. No diagnosis submitted."
            )
            obs.reward, obs.done = 0.0, True
            return obs

        obs = self._build_obs(
            diagnostic_result=diag_result,
            feedback=f"Diagnostic result for {action.query_service} / {action.query_type}:"
        )
        obs.reward, obs.done = 0.0, False
        return obs

    def _handle_diagnosis(self, action: SREAction) -> SREObservation:
        # Validate fields
        errors = self._validate(action)
        if errors:
            obs = self._build_obs(feedback=f"Invalid diagnosis: {errors}. Fix and resubmit.")
            obs.reward, obs.done = 0.0, False
            # Do NOT increment step count for invalid submissions
            return obs

        self._step_count += 1
        self._done = True

        result = grade(
            action         = action,
            scenario       = self._scenario,
            diag_steps     = self._diag_steps,
            queried_hidden = self._queried_hidden,
        )
        self._cumulative_reward = min(1.0, self._cumulative_reward + result.score)

        obs = self._build_obs(feedback=result.feedback)
        obs.reward   = round(result.score, 4)
        obs.done     = True
        obs.metadata = {
            "grade_breakdown":   result.breakdown,
            "raw_score":         result.raw_score,
            "time_penalty":      result.time_penalty,
            "diag_steps":        self._diag_steps,
            "queried_hidden":    self._queried_hidden,
            "correct_service":   result.correct_service,
            "correct_type":      result.correct_type,
            "correct_action":    result.correct_action,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "steps_used":        self._step_count,
        }
        return obs

    def _validate(self, action: SREAction) -> str:
        dep_graph = self._scenario["dependency_graph"]
        errors    = []

        if not action.root_cause_service:
            errors.append("root_cause_service required")
        elif action.root_cause_service not in dep_graph:
            errors.append(
                f"root_cause_service '{action.root_cause_service}' not in graph. "
                f"Valid: {sorted(dep_graph.keys())}"
            )

        if not action.root_cause_type:
            errors.append("root_cause_type required")
        elif action.root_cause_type not in VALID_ROOT_CAUSE_TYPES:
            errors.append(f"root_cause_type must be one of: {sorted(VALID_ROOT_CAUSE_TYPES)}")

        if not action.recommended_action:
            errors.append("recommended_action required")
        elif action.recommended_action not in VALID_ACTIONS:
            errors.append(f"recommended_action must be one of: {sorted(VALID_ACTIONS)}")

        if not (0.0 <= action.confidence <= 1.0):
            errors.append("confidence must be in [0.0, 1.0]")

        return "; ".join(errors)

    def _build_obs(
        self,
        diagnostic_result: Optional[Dict[str, Any]] = None,
        feedback: str = "",
    ) -> SREObservation:
        s = self._scenario
        return SREObservation(
            done              = self._done,
            reward            = None,
            alerts            = s["alerts"],
            logs              = s["logs"],
            metrics           = s["metrics"],
            dependency_graph  = s["dependency_graph"],
            task_id           = self._task_id or "",
            task_description  = s["task_description"] if "task_description" in s else s.get("task_description",""),
            time_elapsed_seconds = self._step_count * 30,
            steps_remaining   = max(0, MAX_STEPS - self._step_count),
            diagnostic_result = diagnostic_result,
            action_feedback   = feedback,
            available_actions = [
                "ACTION A — run_diagnostic (gather more info, costs 1 step):",
                "  {action_type: 'run_diagnostic', query_service: '<n>', query_type: 'recent_logs|metrics_history|connections'}",
                "ACTION B — diagnose (submit final answer, ends episode):",
                "  {action_type: 'diagnose', root_cause_service: '<n>', root_cause_type: '<type>',",
                "   recommended_action: '<action>', confidence: 0.0-1.0, reasoning: '<text>'}",
                f"  Valid root_cause_types: {sorted(VALID_ROOT_CAUSE_TYPES)}",
                f"  Valid recommended_actions: {sorted(VALID_ACTIONS)}",
                f"  Services: {sorted(s['dependency_graph'].keys())}",
                "NOTE: Run diagnostics to gather deeper telemetry before diagnosing.",
            ],
        )
