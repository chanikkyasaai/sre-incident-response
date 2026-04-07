"""
SRE Incident Response — typed models.
Dataclass-based, matching OpenEnv Environment base class interface.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# ── OpenEnv base classes ───────────────────────────────────────────

@dataclass
class Action:
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Observation:
    done:     bool                         = False
    reward:   Union[bool, int, float, None] = None
    metadata: Dict[str, Any]               = field(default_factory=dict)

@dataclass
class State:
    episode_id: Optional[str] = None
    step_count: int            = 0


# ── Domain primitives ─────────────────────────────────────────────

@dataclass
class Alert:
    service:   str
    severity:  str            # P1 | P2 | P3
    message:   str
    metric:    Optional[str]   = None
    value:     Optional[float] = None
    threshold: Optional[float] = None

@dataclass
class LogEntry:
    timestamp: str
    level:     str            # ERROR | WARN | INFO
    service:   str
    message:   str

@dataclass
class ServiceMetric:
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

@dataclass
class SREAction(Action):
    """
    Two action modes:
      run_diagnostic — query one service for deeper telemetry (costs 1 step)
      diagnose       — submit final root-cause diagnosis (ends episode)
    """
    action_type: str = "diagnose"   # "run_diagnostic" | "diagnose"

    # run_diagnostic fields
    query_service: Optional[str] = None   # service name from dependency_graph
    query_type:    Optional[str] = None   # "recent_logs" | "metrics_history" | "connections"

    # diagnose fields
    root_cause_service:  Optional[str]   = None
    root_cause_type:     Optional[str]   = None
    recommended_action:  Optional[str]   = None
    confidence:          float           = 0.5
    reasoning:           str             = ""


@dataclass
class SREObservation(Observation):
    # Core signals — always present
    alerts:           List[Alert]              = field(default_factory=list)
    logs:             List[LogEntry]           = field(default_factory=list)
    metrics:          List[ServiceMetric]      = field(default_factory=list)
    dependency_graph: Dict[str, List[str]]     = field(default_factory=dict)

    # Episode context
    task_id:              str = ""
    task_description:     str = ""
    time_elapsed_seconds: int = 0
    steps_remaining:      int = 0

    # Populated after run_diagnostic
    diagnostic_result: Optional[Dict[str, Any]] = None
    action_feedback:   str = ""

    available_actions: List[str] = field(default_factory=list)


@dataclass
class SREState(State):
    task_id:         str   = ""
    difficulty:      str   = ""
    is_diagnosed:    bool  = False
    current_score:   float = 0.0
    max_steps:       int   = 0
    steps_used:      int   = 0
