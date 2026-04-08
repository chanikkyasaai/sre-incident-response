"""
SRE Incident Response — Typed Models (Pydantic BaseModel as required by OpenEnv spec)
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


# ── OpenEnv base classes ───────────────────────────────────────────

class Action(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Observation(BaseModel):
    done:     bool                          = False
    reward:   Optional[Union[bool, int, float]] = None
    metadata: Dict[str, Any]               = Field(default_factory=dict)

class State(BaseModel):
    episode_id: Optional[str] = None
    step_count: int            = 0


# ── Domain primitives ─────────────────────────────────────────────

class Alert(BaseModel):
    model_config = ConfigDict(frozen=False)
    service:   str
    severity:  str
    message:   str
    metric:    Optional[str]   = None
    value:     Optional[float] = None
    threshold: Optional[float] = None

class LogEntry(BaseModel):
    timestamp: str
    level:     str
    service:   str
    message:   str

class ServiceMetric(BaseModel):
    model_config = ConfigDict(frozen=False)
    service:             str
    cpu_percent:         float
    memory_percent:      float
    error_rate:          float
    latency_p99_ms:      float
    requests_per_second: float


# ── Valid enum values ─────────────────────────────────────────────

VALID_ROOT_CAUSE_TYPES = frozenset([
    "memory_leak", "cpu_spike", "db_connection_exhaustion",
    "lock_contention", "cascade_failure", "bad_deploy", "network_partition",
])

VALID_ACTIONS = frozenset([
    "rollback", "restart_service", "scale_out",
    "investigate_db", "block_traffic", "noop",
])


# ── SRE-specific models ───────────────────────────────────────────

class SREAction(Action):
    action_type:         str            = "diagnose"
    query_service:       Optional[str]  = None
    query_type:          Optional[str]  = None
    root_cause_service:  Optional[str]  = None
    root_cause_type:     Optional[str]  = None
    recommended_action:  Optional[str]  = None
    confidence:          float          = 0.5
    reasoning:           str            = ""


class SREObservation(Observation):
    alerts:            List[Alert]          = Field(default_factory=list)
    logs:              List[LogEntry]       = Field(default_factory=list)
    metrics:           List[ServiceMetric]  = Field(default_factory=list)
    dependency_graph:  Dict[str, List[str]] = Field(default_factory=dict)
    task_id:               str = ""
    task_description:      str = ""
    time_elapsed_seconds:  int = 0
    steps_remaining:       int = 0
    diagnostic_result:     Optional[Dict[str, Any]] = None
    action_feedback:       str = ""
    available_actions:     List[str] = Field(default_factory=list)


class SREState(State):
    task_id:       str   = ""
    difficulty:    str   = ""
    is_diagnosed:  bool  = False
    current_score: float = 0.0
    max_steps:     int   = 0
    steps_used:    int   = 0
