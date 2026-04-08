"""
SRE Incident Response — Typed Models
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class Alert:
    service:   str
    severity:  str
    message:   str
    metric:    Optional[str]   = None
    value:     Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class LogEntry:
    timestamp: str
    level:     str
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


VALID_ROOT_CAUSE_TYPES = frozenset([
    "memory_leak", "cpu_spike", "db_connection_exhaustion",
    "lock_contention", "cascade_failure", "bad_deploy", "network_partition",
])

VALID_ACTIONS = frozenset([
    "rollback", "restart_service", "scale_out",
    "investigate_db", "block_traffic", "noop",
])


@dataclass
class SREAction:
    action_type:         str            = "diagnose"
    query_service:       Optional[str]  = None
    query_type:          Optional[str]  = None
    root_cause_service:  Optional[str]  = None
    root_cause_type:     Optional[str]  = None
    recommended_action:  Optional[str]  = None
    confidence:          float          = 0.5
    reasoning:           str            = ""


@dataclass
class SREObservation:
    done:                  bool                          = False
    reward:                Optional[Union[bool,int,float]] = None
    metadata:              Dict[str, Any]                = field(default_factory=dict)
    alerts:                List[Alert]                   = field(default_factory=list)
    logs:                  List[LogEntry]                = field(default_factory=list)
    metrics:               List[ServiceMetric]           = field(default_factory=list)
    dependency_graph:      Dict[str, List[str]]          = field(default_factory=dict)
    task_id:               str = ""
    task_description:      str = ""
    time_elapsed_seconds:  int = 0
    steps_remaining:       int = 0
    diagnostic_result:     Optional[Dict[str, Any]]      = None
    action_feedback:       str = ""
    available_actions:     List[str]                     = field(default_factory=list)

    def model_dump(self):
        """Compatibility method for Pydantic-style serialization."""
        from dataclasses import asdict
        return asdict(self)


@dataclass
class SREState:
    task_id:       str   = ""
    difficulty:    str   = ""
    is_diagnosed:  bool  = False
    current_score: float = 0.0
    max_steps:     int   = 0
    steps_used:    int   = 0
    episode_id:    Optional[str] = None
    step_count:    int   = 0

    def model_dump(self):
        from dataclasses import asdict
        return asdict(self)
