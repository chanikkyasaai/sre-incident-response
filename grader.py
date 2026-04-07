"""
SRE Incident Response — Grader

Design principles:
1. NO confidence calibration — removes circular reward hacking incentive
2. TIME-PRESSURE penalty — applies to ALL tasks uniformly:
     excess     = max(0, diag_steps - 1)          # first diagnostic free
     time_mult  = max(0.70, 1.0 - 0.06 * excess)  # ranges 1.0 → 0.70
     final      = raw_score * effective_mult
   For hard tasks without hidden diagnostic: effective_mult capped at 0.70
   This means wrong investigation = no investigation (same floor)
   Investigation is always rewarded: found-hidden score ≥ no-investigation score
3. PARTIAL CREDIT only for genuinely close failure type pairs:
   db_connection_exhaustion ↔ lock_contention (both are DB resource starvation)
   No other families — bad_deploy and cascade_failure are NOT similar.
4. PARTIAL CREDIT for actions that are strictly safer alternatives:
   rollback is acceptable when investigate_db is correct (investigate before rollback)
5. PARTIAL OBSERVABILITY CURVE (hard tasks only):
   Without hidden diagnostic: raw_score × 0.70  (rewards correct reasoning)
   With    hidden diagnostic: raw_score × 1.00  (rewards thorough investigation)
"""

from dataclasses import dataclass, field
from typing import Dict

from models import SREAction, VALID_ROOT_CAUSE_TYPES, VALID_ACTIONS

# Partial credit families — ONLY genuinely close failure modes
_TYPE_FAMILIES = [
    frozenset({"db_connection_exhaustion", "lock_contention"}),  # DB resource starvation
]

# Partial credit for actions — strictly safer alternative
_ACCEPTABLE_ACTION_PAIRS = {
    ("rollback",    "investigate_db"),   # rollback is safer than investigate — acceptable
    ("investigate_db", "rollback"),      # investigate before committing to rollback
}


@dataclass
class GradeResult:
    score:           float
    raw_score:       float                      # before time penalty
    breakdown:       Dict[str, float] = field(default_factory=dict)
    time_penalty:    float = 0.0
    feedback:        str   = ""
    correct_service: str   = ""
    correct_type:    str   = ""
    correct_action:  str   = ""


def grade(
    action:         SREAction,
    scenario:       dict,
    diag_steps:     int  = 0,
    queried_hidden: bool = False,
) -> GradeResult:
    """
    Score a final diagnosis.

    Parameters:
        action          — the SREAction with action_type="diagnose"
        scenario        — the scenario dict from get_scenario()
        diag_steps      — number of run_diagnostic steps taken this episode
        queried_hidden  — True if agent queried the hidden_diagnostic service+type

    Returns GradeResult with score in [0.0, 1.0].
    """
    correct_service = scenario["root_cause_service"]
    correct_type    = scenario["root_cause_type"]
    correct_action  = scenario["correct_action"]
    is_hard         = scenario.get("difficulty", "") == "hard"
    has_hidden      = "hidden_diagnostic" in scenario
    breakdown       = {}

    # ── 1. Root cause service (0.0–0.45) ──────────────────────────
    if action.root_cause_service == correct_service:
        breakdown["root_cause_service"] = 0.45
    else:
        breakdown["root_cause_service"] = 0.0

    # ── 2. Root cause type (0.0–0.30) ─────────────────────────────
    if action.root_cause_type == correct_type:
        breakdown["root_cause_type"] = 0.30
    elif _is_close_type(action.root_cause_type, correct_type):
        breakdown["root_cause_type"] = 0.12
    else:
        breakdown["root_cause_type"] = 0.0

    # ── 3. Recommended action (0.0–0.25) ──────────────────────────
    if action.recommended_action == correct_action:
        breakdown["recommended_action"] = 0.25
    elif _is_acceptable_action(action.recommended_action, correct_action):
        breakdown["recommended_action"] = 0.10
    else:
        breakdown["recommended_action"] = 0.0

    raw_score = sum(breakdown.values())
    raw_score = round(min(1.0, max(0.0, raw_score)), 4)

    # ── 4+5. Time penalty + Partial observability (unified) ──────────
    #
    # TIME PENALTY applies to ALL tasks:
    #   excess     = max(0, diag_steps - 1)      first diagnostic is free
    #   time_mult  = max(0.70, 1.0 - 0.06*excess) ranges [1.0 → 0.70]
    #
    # OBSERVABILITY cap for hard tasks (partial observability curve):
    #   If hard task AND hidden diagnostic NOT queried:
    #     cap time_mult at 0.70  →  investigation score floor = no-investigation
    #   If hard task AND hidden diagnostic WAS queried:
    #     full time_mult applies  →  investigation rewarded
    #   Easy/medium tasks: no cap, time_mult applies directly
    #
    # Outcomes:
    #   Easy/medium, 0 diags:     raw * 1.00 = raw
    #   Easy/medium, 1 diag:      raw * 1.00 = raw  (free)
    #   Easy/medium, 5 diags:     raw * 0.76
    #   Hard, no diag, 0 steps:   raw * min(0.70, 1.00) = raw * 0.70
    #   Hard, no diag, 5 steps:   raw * min(0.70, 0.76) = raw * 0.70  (floor, same as no-invest)
    #   Hard, found, 1 diag:      raw * max(0.70, 1.00) = raw * 1.00
    #   Hard, found, 4 excess:    raw * max(0.70, 0.76) = raw * 0.76
    excess_steps   = max(0, diag_steps - 1)
    time_mult      = round(max(0.70, 1.0 - 0.06 * excess_steps), 4)

    if is_hard and has_hidden:
        effective_mult = time_mult if queried_hidden else min(0.70, time_mult)
    else:
        effective_mult = time_mult

    time_penalty   = round(1.0 - effective_mult, 4)  # actual reduction applied
    final_score    = round(min(1.0, raw_score * effective_mult), 4)

    # ── Feedback ──────────────────────────────────────────────────
    svc_ok   = breakdown["root_cause_service"] > 0
    typ_ok   = breakdown["root_cause_type"]    >= 0.30
    act_ok   = breakdown["recommended_action"] >= 0.25
    all_ok   = svc_ok and typ_ok and act_ok

    if all_ok:
        # Correct diagnosis — score may be reduced by time or obs penalty
        inv_note = ""
        if is_hard and has_hidden and not queried_hidden:
            inv_note = " Score at 0.70× — run hidden diagnostic for full 1.00×."
        elif time_penalty > 0:
            inv_note = f" Score reduced {time_penalty:.0%} by {diag_steps} diagnostic steps."
        feedback = (
            f"Correct diagnosis: {correct_service} ({correct_type}) → {correct_action}."
            + inv_note
        )
    elif svc_ok and (typ_ok or act_ok):
        feedback = (
            f"Right service ({correct_service}). "
            f"Type: {'correct' if typ_ok else f'wrong — was {correct_type}'}. "
            f"Action: {'correct' if act_ok else f'wrong — was {correct_action}'}."
        )
    elif svc_ok:
        feedback = (
            f"Correct service ({correct_service}) but wrong type and action. "
            f"Type: {correct_type}. Action: {correct_action}."
        )
    elif is_hard and has_hidden and not queried_hidden:
        feedback = (
            f"Wrong service. Run run_diagnostic to surface hidden evidence. "
            f"Root cause: {correct_service} ({correct_type}). "
            f"Correct action: {correct_action}."
        )
    else:
        feedback = (
            f"Wrong service. Root cause: {correct_service} ({correct_type}). "
            f"Action: {correct_action}. Trace dependency graph upstream."
        )

    return GradeResult(
        score=final_score,
        raw_score=raw_score,
        breakdown=breakdown,
        time_penalty=time_penalty,
        feedback=feedback,
        correct_service=correct_service,
        correct_type=correct_type,
        correct_action=correct_action,
    )


def _is_close_type(predicted: str, correct: str) -> bool:
    for family in _TYPE_FAMILIES:
        if predicted in family and correct in family and predicted != correct:
            return True
    return False


def _is_acceptable_action(predicted: str, correct: str) -> bool:
    return (predicted, correct) in _ACCEPTABLE_ACTION_PAIRS
