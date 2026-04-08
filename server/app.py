"""
FastAPI server for SRE Incident Response OpenEnv.

Endpoints:
  GET  /health              — HF Spaces ping, returns 200
  POST /reset               — Start episode, returns session_id + observation
  POST /step                — Take action, returns observation + reward + done
  GET  /state               — Episode metadata
  GET  /                    — Info

Session management:
  TTL-based expiry (30 min inactive) — not count-based FIFO.
  Concurrent requests to different sessions never share state.
"""

import time
import uuid
from dataclasses import asdict
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from server.sre_environment import SREEnvironment
from models import SREAction
from scenarios import TASK_IDS, EASY_TASKS, MEDIUM_TASKS, HARD_TASKS

app = FastAPI(
    title       = "SRE Incident Response — OpenEnv",
    description = (
        "Real-world SRE incident triage. "
        "Agent diagnoses production outages via multi-step investigation. "
        "6 task families × 3 variants each = 18 unique episode configurations."
    ),
    version = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Session store with TTL + thread safety ────────────────────────

import threading

SESSION_TTL_SECONDS = 1800  # 30 minutes

class _Session:
    def __init__(self):
        self.env        = SREEnvironment()
        self.last_touch = time.time()

    def touch(self):
        self.last_touch = time.time()

    @property
    def expired(self) -> bool:
        return (time.time() - self.last_touch) > SESSION_TTL_SECONDS


_sessions: Dict[str, _Session] = {}
_sessions_lock = threading.Lock()


def _evict_expired():
    """Remove sessions idle longer than SESSION_TTL_SECONDS. Call under lock."""
    expired = [sid for sid, s in _sessions.items() if s.expired]
    for sid in expired:
        del _sessions[sid]


def _get_session(session_id: str) -> _Session:
    with _sessions_lock:
        _evict_expired()
        if session_id not in _sessions:
            raise HTTPException(404, f"Session '{session_id}' not found. Call /reset first.")
        s = _sessions[session_id]
        s.touch()
        return s


# ── Request/response models ────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id:    str           = Field("easy_memory_leak", description=f"One of: {TASK_IDS}")
    seed:       int           = Field(42, description="Seed for reproducible episodes. seed%3 selects scenario variant.")
    session_id: Optional[str] = Field(None, description="Reuse an existing session. Auto-generated if omitted.")


class StepRequest(BaseModel):
    session_id:         str            = Field(..., description="Session ID from /reset")
    action_type:        str            = Field(..., description="'run_diagnostic' or 'diagnose'")
    # run_diagnostic
    query_service:      Optional[str]  = None
    query_type:         Optional[str]  = None
    # diagnose
    root_cause_service: Optional[str]  = None
    root_cause_type:    Optional[str]  = None
    recommended_action: Optional[str]  = None
    confidence:         float          = Field(0.5, ge=0.0, le=1.0)
    reasoning:          str            = ""


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/health")
def health():
    """HuggingFace Spaces automated ping. Must return 200."""
    _evict_expired()
    return {"status": "ok", "active_sessions": len(_sessions), "tasks": len(TASK_IDS)}


@app.get("/")
def root():
    return {
        "name":    "SRE Incident Response — OpenEnv",
        "version": "1.0.0",
        "tasks":   {"easy": EASY_TASKS, "medium": MEDIUM_TASKS, "hard": HARD_TASKS},
        "variants_per_task": 3,
        "episode_flow": [
            "POST /reset  → session_id + initial observation",
            "POST /step (run_diagnostic) → deeper telemetry, optional",
            "POST /step (diagnose)       → score, done=True",
            "GET  /state → episode metadata",
        ],
        "scoring": {
            "root_cause_service":  0.45,
            "root_cause_type":     0.30,
            "recommended_action":  0.25,
            "time_pressure":       "score *= max(0.70, 1 - 0.06*(diag_steps-1))",
            "partial_observability": "hard tasks cap score without querying hidden evidence",
        },
    }


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()
    if req.task_id not in TASK_IDS:
        raise HTTPException(400, f"Unknown task_id. Valid: {TASK_IDS}")

    with _sessions_lock:
        _evict_expired()
        session_id = req.session_id or str(uuid.uuid4())[:12]
        session    = _Session()
        _sessions[session_id] = session

    try:
        obs = session.env.reset(task_id=req.task_id, seed=req.seed)
        return {"session_id": session_id, "observation": obs.model_dump()}
    except Exception as e:
        with _sessions_lock:
            _sessions.pop(session_id, None)
        raise HTTPException(500, str(e))


@app.post("/step")
def step(req: StepRequest):
    session = _get_session(req.session_id)
    action  = SREAction(
        action_type        = req.action_type,
        query_service      = req.query_service,
        query_type         = req.query_type,
        root_cause_service = req.root_cause_service,
        root_cause_type    = req.root_cause_type,
        recommended_action = req.recommended_action,
        confidence         = req.confidence,
        reasoning          = req.reasoning,
    )
    try:
        obs = session.env.step(action)
        return {"observation": obs.model_dump(), "done": obs.done, "reward": obs.reward}
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/state")
def state(session_id: str = Query(..., description="Session ID from /reset")):
    session = _get_session(session_id)
    return session.env.state().model_dump()


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
