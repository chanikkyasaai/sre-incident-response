"""
SRE Incident Response — Scenario Engine

KEY DESIGN DECISIONS (addressing all critique points):

1. PARTIAL OBSERVABILITY (mandatory for hard tasks)
   Hard tasks withhold the critical evidence from the initial observation.
   It lives in diagnostic_data and can only be found via run_diagnostic().
   An agent that skips diagnostics cannot score above PARTIAL_OBS_CAP on hard tasks.

2. SCENARIO FAMILIES (rotating root causes)
   Each task family has 3 variants. seed % 3 selects which variant is active.
   Same topology, different root cause each rotation.
   An RL agent cannot memorize the answer — it must learn the reasoning.

3. CLEAN RED HERRINGS
   Red herrings do NOT announce themselves.
   They look like legitimate signals. The agent must reason about
   temporal correlation, dependency direction, and metric consistency
   to dismiss them — not read a label.

4. DISTINCT EASY TASKS
   easy_memory_leak: tests LOG READING (follow the heap growth timeline)
   easy_cpu_spike:   tests ALERT CORRELATION (CPU alert + queue depth)
   Different reasoning skills required.

5. TIME-PRESSURE MECHANIC
   Final score is penalised for excess diagnostic steps:
     score *= max(0.70, 1.0 - 0.06 * max(0, diag_steps - 1))
   Diagnose correctly in 1 step: full score.
   Run 4 diagnostics then diagnose: 82% of score.
   Real on-call engineers face MTTR pressure. This models it.
"""

import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from models import Alert, LogEntry, ServiceMetric

# Agents cannot score above this on hard tasks without running diagnostics
# Used by grader: soft curve multiplier — correct w/o diagnostic gets 0.70 * raw_score
# With correct service (0.45) + correct type (0.30) + correct action (0.25) * 0.70 = 0.70
PARTIAL_OBS_CAP = 0.70   # floor multiplier when hidden diagnostic not queried


# ─────────────────────────────────────────────────────────────────
# Scenario family definition
# Each family has 3 variants (rotated by seed % 3).
# Variants share topology but rotate the root cause service.
# ─────────────────────────────────────────────────────────────────

def _easy_memory_leak_variants() -> List[Dict]:
    """
    Easy task 1: Tests LOG READING.
    A single service has a growing heap. Logs show the progression.
    Initial observation contains full evidence — no diagnostics needed.
    Red herring: a downstream service shows elevated error rate.
    """
    base = [
        # Variant 0: payment-service
        {
            "root_cause_service": "payment-service",
            "root_cause_type":    "memory_leak",
            "correct_action":     "restart_service",
            "alerts": [
                Alert("payment-service", "P2", "Memory usage critical",       "memory_percent", 93.8, 85.0),
                Alert("api-gateway",     "P3", "Upstream error rate elevated", "error_rate",      0.04, 0.02),
            ],
            "logs": [
                LogEntry("T+00:00", "WARN",  "payment-service", "JVM heap: 1.1GB / 2GB"),
                LogEntry("T+01:00", "WARN",  "payment-service", "JVM heap: 1.5GB / 2GB — GC pressure increasing"),
                LogEntry("T+02:00", "WARN",  "payment-service", "JVM heap: 1.9GB / 2GB — GC running continuously"),
                LogEntry("T+03:00", "ERROR", "payment-service", "OutOfMemoryError: Java heap space"),
                LogEntry("T+03:01", "ERROR", "payment-service", "GC overhead limit exceeded"),
                LogEntry("T+03:05", "WARN",  "api-gateway",     "503 upstream: payment-service not responding"),
                # Red herring: looks like auth is involved but it isn't
                LogEntry("T+02:45", "INFO",  "auth-service",    "Token cache eviction: 2400 entries pruned (scheduled)"),
            ],
            "metrics": [
                ServiceMetric("payment-service",  46.0, 93.8, 0.14, 410.0, 135.0),
                ServiceMetric("api-gateway",      28.0, 36.0, 0.04,  88.0, 480.0),
                ServiceMetric("auth-service",     20.0, 34.0, 0.00,  38.0, 520.0),
                ServiceMetric("postgres-primary", 52.0, 58.0, 0.00,  14.0, 175.0),
            ],
            "dependency_graph": {
                "api-gateway":      ["payment-service", "auth-service"],
                "payment-service":  ["postgres-primary"],
                "auth-service":     ["postgres-primary"],
                "postgres-primary": [],
            },
        },
        # Variant 1: auth-service leaks
        {
            "root_cause_service": "auth-service",
            "root_cause_type":    "memory_leak",
            "correct_action":     "restart_service",
            "alerts": [
                Alert("auth-service",    "P2", "Memory usage critical",       "memory_percent", 91.4, 85.0),
                Alert("api-gateway",     "P3", "Authentication latency high",  "latency_p99_ms", 1200.0, 300.0),
            ],
            "logs": [
                LogEntry("T+00:00", "WARN",  "auth-service",    "Session store: 800MB / 1GB"),
                LogEntry("T+01:30", "WARN",  "auth-service",    "Session store: 1.1GB — not evicting expired sessions"),
                LogEntry("T+02:30", "ERROR", "auth-service",    "OutOfMemoryError: unable to allocate session object"),
                LogEntry("T+02:35", "WARN",  "api-gateway",     "Auth timeout — requests queuing"),
                # Red herring: payment looks suspicious but is fine
                LogEntry("T+02:00", "INFO",  "payment-service", "Stripe webhook received: payment_intent.succeeded"),
            ],
            "metrics": [
                ServiceMetric("auth-service",     55.0, 91.4, 0.18, 1200.0, 200.0),
                ServiceMetric("api-gateway",      35.0, 38.0, 0.06,  420.0, 490.0),
                ServiceMetric("payment-service",  22.0, 40.0, 0.01,   85.0, 180.0),
                ServiceMetric("postgres-primary", 48.0, 55.0, 0.00,   11.0, 170.0),
            ],
            "dependency_graph": {
                "api-gateway":      ["payment-service", "auth-service"],
                "payment-service":  ["postgres-primary"],
                "auth-service":     ["postgres-primary"],
                "postgres-primary": [],
            },
        },
        # Variant 2: worker-service leaks
        {
            "root_cause_service": "worker-service",
            "root_cause_type":    "memory_leak",
            "correct_action":     "restart_service",
            "alerts": [
                Alert("worker-service",  "P2", "Memory usage critical",    "memory_percent", 95.1, 85.0),
                Alert("job-queue",       "P3", "Job processing latency",    "latency_p99_ms", 8200.0, 1000.0),
            ],
            "logs": [
                LogEntry("T+00:00", "WARN",  "worker-service", "Heap: 3.1GB / 4GB — large object allocation"),
                LogEntry("T+01:00", "WARN",  "worker-service", "Heap: 3.7GB / 4GB — GC unable to reclaim"),
                LogEntry("T+02:00", "ERROR", "worker-service", "OutOfMemoryError: GC overhead limit exceeded"),
                LogEntry("T+02:05", "WARN",  "job-queue",      "Worker not polling — jobs backing up"),
                # Red herring: redis looks related
                LogEntry("T+01:30", "INFO",  "redis-cache",    "Memory: 2.1GB / 8GB — within normal bounds"),
            ],
            "metrics": [
                ServiceMetric("worker-service",  44.0, 95.1, 0.22, 8200.0,  60.0),
                ServiceMetric("job-queue",       18.0, 32.0, 0.08, 4100.0, 140.0),
                ServiceMetric("redis-cache",     12.0, 26.0, 0.00,    2.0, 900.0),
                ServiceMetric("postgres-primary",50.0, 55.0, 0.00,   10.0, 200.0),
            ],
            "dependency_graph": {
                "job-queue":        ["worker-service", "redis-cache"],
                "worker-service":   ["postgres-primary", "redis-cache"],
                "redis-cache":      [],
                "postgres-primary": [],
            },
        },
    ]
    return base


def _easy_cpu_spike_variants() -> List[Dict]:
    """
    Easy task 2: Tests ALERT + LOG CORRELATION.
    A batch process saturates CPU. Distinct from memory leak:
    requires correlating the batch-job log entry with the CPU alert.
    Initial observation contains full evidence.
    """
    base = [
        # Variant 0: image-processor
        {
            "root_cause_service": "image-processor",
            "root_cause_type":    "cpu_spike",
            "correct_action":     "scale_out",
            "alerts": [
                Alert("image-processor", "P2", "CPU usage critical",        "cpu_percent",    97.0, 80.0),
                Alert("upload-service",  "P3", "Processing latency elevated","latency_p99_ms", 4800.0, 1000.0),
            ],
            "logs": [
                LogEntry("T+00:00", "INFO",  "image-processor", "Batch triggered: compress 9200 images for CDN"),
                LogEntry("T+00:10", "WARN",  "image-processor", "All 16 worker threads active"),
                LogEntry("T+01:00", "WARN",  "image-processor", "Job queue depth: 3100 — falling behind"),
                LogEntry("T+02:00", "ERROR", "upload-service",  "Timeout on image-processor after 30s"),
                # Red herring: postgres has elevated writes
                LogEntry("T+01:30", "INFO",  "postgres-primary","Checkpoint: 6200 buffers written (elevated — high write load)"),
            ],
            "metrics": [
                ServiceMetric("image-processor",  97.0, 54.0, 0.06, 9100.0,  18.0),
                ServiceMetric("upload-service",   32.0, 38.0, 0.09, 4800.0, 110.0),
                ServiceMetric("postgres-primary", 68.0, 55.0, 0.00,   18.0, 220.0),
            ],
            "dependency_graph": {
                "upload-service":   ["image-processor", "postgres-primary"],
                "image-processor":  [],
                "postgres-primary": [],
            },
        },
        # Variant 1: report-generator
        {
            "root_cause_service": "report-generator",
            "root_cause_type":    "cpu_spike",
            "correct_action":     "scale_out",
            "alerts": [
                Alert("report-generator", "P2", "CPU usage critical",        "cpu_percent",    96.0, 80.0),
                Alert("analytics-api",    "P3", "Report API latency elevated","latency_p99_ms", 6200.0, 500.0),
            ],
            "logs": [
                LogEntry("T+00:00", "INFO",  "report-generator", "Monthly report job started: 140 reports"),
                LogEntry("T+00:30", "WARN",  "report-generator", "CPU-bound aggregation: 96% utilisation"),
                LogEntry("T+01:30", "WARN",  "report-generator", "Report queue: 88 pending"),
                LogEntry("T+02:30", "ERROR", "analytics-api",    "Timeout waiting for report-generator: 60s"),
                # Red herring: cache miss
                LogEntry("T+01:00", "INFO",  "redis-cache",      "Cache miss rate: 18% (elevated — cold start after deploy)"),
            ],
            "metrics": [
                ServiceMetric("report-generator", 96.0, 60.0, 0.04, 6200.0, 12.0),
                ServiceMetric("analytics-api",    30.0, 42.0, 0.07, 3100.0, 80.0),
                ServiceMetric("redis-cache",      15.0, 30.0, 0.00,    3.0, 600.0),
            ],
            "dependency_graph": {
                "analytics-api":    ["report-generator", "redis-cache"],
                "report-generator": [],
                "redis-cache":      [],
            },
        },
        # Variant 2: ml-training-job
        {
            "root_cause_service": "ml-training-job",
            "root_cause_type":    "cpu_spike",
            "correct_action":     "scale_out",
            "alerts": [
                Alert("ml-training-job", "P2", "CPU usage critical",         "cpu_percent",    98.0, 80.0),
                Alert("model-api",       "P3", "Model serving latency high",  "latency_p99_ms", 3400.0, 500.0),
            ],
            "logs": [
                LogEntry("T+00:00", "INFO",  "ml-training-job", "Retraining started: 2.1M samples, 40 epochs"),
                LogEntry("T+00:20", "WARN",  "ml-training-job", "CPU: 98% — all cores allocated to training"),
                LogEntry("T+01:00", "WARN",  "model-api",       "Inference requests queuing behind training job"),
                LogEntry("T+02:00", "ERROR", "model-api",       "Request timeout: training job monopolising CPU"),
                # Red herring: disk IO
                LogEntry("T+00:45", "INFO",  "ml-training-job", "Checkpoint saved: 1.8GB written to disk"),
            ],
            "metrics": [
                ServiceMetric("ml-training-job", 98.0, 72.0, 0.00, 2100.0,  5.0),
                ServiceMetric("model-api",       55.0, 48.0, 0.11, 3400.0, 90.0),
                ServiceMetric("postgres-primary",40.0, 50.0, 0.00,   12.0, 150.0),
            ],
            "dependency_graph": {
                "model-api":        ["ml-training-job", "postgres-primary"],
                "ml-training-job":  [],
                "postgres-primary": [],
            },
        },
    ]
    return base


def _medium_db_contention_variants() -> List[Dict]:
    """
    Medium task 1: Tests DEPENDENCY GRAPH TRACING.
    Multiple downstream services alert simultaneously.
    Root cause is in a shared dependency.
    Initial observation has enough to solve — but a graph intersection
    analysis is required. One service is a deliberate decoy.
    Diagnostic data exists but is not needed to score full marks.
    """
    base = [
        # Variant 0: postgres primary
        {
            "root_cause_service": "postgres-primary",
            "root_cause_type":    "lock_contention",
            "correct_action":     "investigate_db",
            "alerts": [
                Alert("order-service",     "P1", "Error rate 21%",          "error_rate",     0.21, 0.05),
                Alert("inventory-service", "P2", "Latency p99 elevated",     "latency_p99_ms", 2800.0, 500.0),
                Alert("redis-cache",       "P3", "Cache hit rate degraded",  "cache_hit_rate", 0.88, 0.92),
            ],
            "logs": [
                LogEntry("T+00:01", "ERROR", "order-service",     "QueryTimeoutException after 30s on DB call"),
                LogEntry("T+00:02", "ERROR", "inventory-service", "Connection pool timeout after 30s — all connections busy"),
                LogEntry("T+00:04", "INFO",  "order-service",     "Retry attempt 2/3 — same timeout"),
                LogEntry("T+00:08", "INFO",  "order-service",     "Retry attempt 3/3 failed — request dropped"),
                # Red herrings
                LogEntry("T-05:00", "INFO",  "redis-cache",       "Scheduled key expiry: 11400 entries removed"),
                LogEntry("T-02:00", "INFO",  "order-service",     "Traffic: 2.1x normal order volume (flash sale)"),
            ],
            "metrics": [
                ServiceMetric("order-service",     56.0, 60.0, 0.21, 3100.0, 190.0),
                ServiceMetric("inventory-service", 44.0, 50.0, 0.11, 2800.0, 170.0),
                ServiceMetric("postgres-primary",  89.0, 70.0, 0.00,    0.0,   0.0),
                ServiceMetric("redis-cache",       13.0, 28.0, 0.00,    2.0, 820.0),
            ],
            "dependency_graph": {
                "api-gateway":       ["order-service", "inventory-service"],
                "order-service":     ["postgres-primary", "redis-cache"],
                "inventory-service": ["postgres-primary", "redis-cache"],
                "postgres-primary":  [],
                "redis-cache":       [],
            },
        },
        # Variant 1: shared message broker
        {
            "root_cause_service": "message-broker",
            "root_cause_type":    "db_connection_exhaustion",
            "correct_action":     "investigate_db",
            "alerts": [
                Alert("notification-svc", "P1", "Message delivery failures 34%", "delivery_fail_rate", 0.34, 0.05),
                Alert("event-processor",  "P2", "Event processing latency high",  "latency_p99_ms",     4200.0, 500.0),
            ],
            "logs": [
                LogEntry("T+00:01", "ERROR", "notification-svc",  "AMQP connection refused: pool exhausted"),
                LogEntry("T+00:02", "ERROR", "event-processor",   "Cannot acquire broker connection after 10s"),
                LogEntry("T+00:04", "WARN",  "message-broker",    "Connection pool: 99/100 — refusing new connections"),
                LogEntry("T+00:05", "WARN",  "message-broker",    "Long-running consumer detected: PID 8821 open 2h"),
                LogEntry("T+00:07", "INFO",  "notification-svc",  "Retry 2/3"),
                # Red herring: scheduled maintenance window started
                LogEntry("T-10:00", "INFO",  "monitoring",        "Maintenance window opened for network switch upgrade"),
            ],
            "metrics": [
                ServiceMetric("notification-svc", 38.0, 44.0, 0.34, 3200.0, 160.0),
                ServiceMetric("event-processor",  42.0, 48.0, 0.18, 4200.0, 140.0),
                ServiceMetric("message-broker",   71.0, 65.0, 0.00,    0.0,   0.0),
                ServiceMetric("monitoring",        8.0, 20.0, 0.00,    5.0,  10.0),
            ],
            "dependency_graph": {
                "api-gateway":       ["notification-svc", "event-processor"],
                "notification-svc":  ["message-broker"],
                "event-processor":   ["message-broker"],
                "message-broker":    [],
                "monitoring":        [],
            },
        },
        # Variant 2: shared cache becoming a bottleneck
        {
            "root_cause_service": "redis-cluster",
            "root_cause_type":    "db_connection_exhaustion",
            "correct_action":     "investigate_db",
            "alerts": [
                Alert("session-service", "P1", "Auth failure rate 28%", "error_rate", 0.28, 0.05),
                Alert("cart-service",    "P2", "Cart save failures 19%", "error_rate", 0.19, 0.05),
            ],
            "logs": [
                LogEntry("T+00:01", "ERROR", "session-service",  "Redis MAXCLIENTS reached — connection refused"),
                LogEntry("T+00:02", "ERROR", "cart-service",     "Redis connection timeout after 5s"),
                LogEntry("T+00:04", "WARN",  "redis-cluster",    "Connected clients: 498/500"),
                LogEntry("T+00:05", "WARN",  "redis-cluster",    "Client with idle time >3600s detected: 80 connections"),
                LogEntry("T+00:07", "INFO",  "session-service",  "Retrying with new connection"),
                # Red herring: recent deploy with no issues
                LogEntry("T-30:00", "INFO",  "cart-service",     "Deploy v3.1.0 completed — all health checks passed"),
            ],
            "metrics": [
                ServiceMetric("session-service", 40.0, 45.0, 0.28, 2400.0, 200.0),
                ServiceMetric("cart-service",    35.0, 42.0, 0.19, 1800.0, 180.0),
                ServiceMetric("redis-cluster",   60.0, 72.0, 0.00,    0.0,   0.0),
            ],
            "dependency_graph": {
                "api-gateway":     ["session-service", "cart-service"],
                "session-service": ["redis-cluster"],
                "cart-service":    ["redis-cluster"],
                "redis-cluster":   [],
            },
        },
    ]
    return base


def _medium_network_variants() -> List[Dict]:
    """
    Medium task 2: Tests TEMPORAL REASONING.
    Intermittent failures caused by a infrastructure-level event.
    The pattern (some succeed, some fail, correlated with a specific event)
    is the key signal. Requires reading log timestamps carefully.
    """
    base = [
        # Variant 0: service mesh reload
        {
            "root_cause_service": "service-mesh",
            "root_cause_type":    "network_partition",
            "correct_action":     "rollback",
            "alerts": [
                Alert("checkout-service", "P1", "Error rate 34%",             "error_rate",   0.34, 0.05),
                Alert("payment-service",  "P2", "Connection refused errors",   "conn_refused", 120.0, 0.0),
            ],
            "logs": [
                LogEntry("T-00:10", "INFO",  "service-mesh",    "Config reload triggered by operator"),
                LogEntry("T+00:01", "ERROR", "checkout-service","ECONNREFUSED payment-service:8080"),
                LogEntry("T+00:02", "ERROR", "checkout-service","ECONNREFUSED payment-service:8080"),
                LogEntry("T+00:02", "INFO",  "checkout-service","Request succeeded on retry — intermittent"),
                LogEntry("T+00:04", "WARN",  "service-mesh",    "iptables flush in progress — 28s window"),
                LogEntry("T+00:32", "INFO",  "service-mesh",    "Config reload complete — routing restored"),
                LogEntry("T+00:33", "INFO",  "checkout-service","Error rate returning to baseline"),
                # Red herring: unrelated DB activity
                LogEntry("T-05:00", "INFO",  "postgres-primary","Scheduled ANALYZE on public.orders complete"),
            ],
            "metrics": [
                ServiceMetric("checkout-service", 38.0, 44.0, 0.34, 1200.0, 290.0),
                ServiceMetric("payment-service",  35.0, 40.0, 0.18,  800.0, 240.0),
                ServiceMetric("service-mesh",      4.0,  8.0, 0.00,    0.0,   0.0),
                ServiceMetric("postgres-primary", 28.0, 52.0, 0.00,   14.0, 190.0),
            ],
            "dependency_graph": {
                "api-gateway":      ["checkout-service"],
                "checkout-service": ["payment-service", "service-mesh"],
                "payment-service":  ["service-mesh", "postgres-primary"],
                "service-mesh":     [],
                "postgres-primary": [],
            },
        },
        # Variant 1: load balancer misconfiguration
        {
            "root_cause_service": "load-balancer",
            "root_cause_type":    "network_partition",
            "correct_action":     "rollback",
            "alerts": [
                Alert("user-service",   "P1", "Error rate 29%",            "error_rate",   0.29, 0.05),
                Alert("search-service", "P2", "Connection timeout errors",  "conn_refused", 88.0, 0.0),
            ],
            "logs": [
                LogEntry("T-00:05", "INFO",  "load-balancer",  "Health check config updated: interval 10s→60s"),
                LogEntry("T+00:01", "ERROR", "user-service",   "Connection timeout to search-service:9200"),
                LogEntry("T+00:02", "INFO",  "user-service",   "Retry succeeded — intermittent failure"),
                LogEntry("T+00:03", "ERROR", "search-service", "Upstream timeout: load-balancer health check failing"),
                LogEntry("T+00:05", "WARN",  "load-balancer",  "Unhealthy backend removed — 1 of 3 nodes gone"),
                LogEntry("T+00:06", "WARN",  "load-balancer",  "Traffic concentration on 2 nodes — asymmetric load"),
                # Red herring: app deploy
                LogEntry("T-45:00", "INFO",  "user-service",   "Deploy v2.8.1 complete — canary 10% traffic"),
            ],
            "metrics": [
                ServiceMetric("user-service",   44.0, 48.0, 0.29, 2200.0, 280.0),
                ServiceMetric("search-service", 78.0, 55.0, 0.14, 1800.0, 240.0),
                ServiceMetric("load-balancer",   3.0,  6.0, 0.00,    0.0,   0.0),
            ],
            "dependency_graph": {
                "api-gateway":    ["user-service", "search-service"],
                "user-service":   ["search-service", "load-balancer"],
                "search-service": ["load-balancer"],
                "load-balancer":  [],
            },
        },
        # Variant 2: DNS resolver failure
        {
            "root_cause_service": "dns-resolver",
            "root_cause_type":    "network_partition",
            "correct_action":     "rollback",
            "alerts": [
                Alert("recommendation-svc", "P1", "External API failures 41%", "error_rate",   0.41, 0.05),
                Alert("product-service",    "P2", "Vendor API timeouts",        "conn_refused", 95.0, 0.0),
            ],
            "logs": [
                LogEntry("T-00:08", "INFO",  "dns-resolver",      "Primary DNS config updated — new forwarders"),
                LogEntry("T+00:01", "ERROR", "recommendation-svc","DNS resolution failed: vendor-api.external.com"),
                LogEntry("T+00:02", "INFO",  "recommendation-svc","Retry succeeded using cached record — intermittent"),
                LogEntry("T+00:03", "ERROR", "product-service",   "getaddrinfo ENOTFOUND catalog.supplier.io"),
                LogEntry("T+00:05", "WARN",  "dns-resolver",      "Upstream forwarder 10.0.0.1: response timeout"),
                LogEntry("T+00:07", "WARN",  "dns-resolver",      "Falling back to secondary — higher latency"),
                # Red herring: certificate renewal
                LogEntry("T-60:00", "INFO",  "cert-manager",      "TLS certificate renewed: vendor-api.external.com"),
            ],
            "metrics": [
                ServiceMetric("recommendation-svc", 42.0, 46.0, 0.41, 3400.0, 160.0),
                ServiceMetric("product-service",    38.0, 44.0, 0.28, 2800.0, 200.0),
                ServiceMetric("dns-resolver",        2.0,  4.0, 0.48,  820.0,  28.0),
                ServiceMetric("cert-manager",        1.0,  3.0, 0.00,    0.0,   0.0),
            ],
            "dependency_graph": {
                "api-gateway":        ["recommendation-svc", "product-service"],
                "recommendation-svc": ["dns-resolver"],
                "product-service":    ["dns-resolver"],
                "dns-resolver":       [],
                "cert-manager":       [],
            },
        },
    ]
    return base


def _hard_bad_deploy_variants() -> List[Dict]:
    """
    Hard task 1: PARTIAL OBSERVABILITY — config-layer failure.

    CRITICAL DESIGN:
    - The initial observation OMITS the deploy log and config key mapping.
    - The agent sees: mass P1 alerts, services failing on config lookups,
      a quiet service with no alerts and healthy metrics.
    - Without running run_diagnostic("config-service", "recent_logs"),
      the agent cannot know WHICH service caused the failure or WHY.
    - Agents that diagnose without querying score max PARTIAL_OBS_CAP.
    - The deploy log is only in diagnostic_data["config-service"]["recent_logs"].

    Three variants rotate which config-layer service is the culprit.
    """
    base = [
        # Variant 0: config-service schema migration
        {
            "root_cause_service": "config-service",
            "root_cause_type":    "bad_deploy",
            "correct_action":     "rollback",
            "alerts": [
                Alert("frontend",        "P1", "5xx error rate 47%",              "error_rate",       0.47, 0.01),
                Alert("api-gateway",     "P1", "Request success rate 59%",         "success_rate",     0.59, 0.99),
                Alert("auth-service",    "P1", "Token validation failures spiking","validation_fails", 880.0, 10.0),
                Alert("payment-service", "P2", "Elevated error rate",              "error_rate",       0.32, 0.05),
            ],
            "logs": [
                # NO deploy log here — it's hidden in diagnostic_data
                LogEntry("T+00:00", "ERROR", "auth-service",    "KeyError: config key not found — using fallback"),
                LogEntry("T+00:00", "WARN",  "auth-service",    "JWT secret fallback active — existing sessions invalidated"),
                LogEntry("T+00:01", "ERROR", "payment-service", "Configuration error: payment processor init failed"),
                LogEntry("T+00:02", "WARN",  "api-gateway",     "Auth token rejected — 401 returned to client"),
                LogEntry("T+00:03", "ERROR", "frontend",        "401 flood — users being logged out"),
                # Red herrings — unlabeled, look plausible
                LogEntry("T-02:00", "INFO",  "postgres-replica","Replication lag 1.8s — 48 transactions behind"),
                LogEntry("T-01:00", "INFO",  "cdn",             "Edge cache invalidated for /api/v2/ — routine"),
                LogEntry("T-00:30", "WARN",  "redis-cache",     "Memory: 5.2GB / 8GB — approaching warning threshold"),
            ],
            "metrics": [
                ServiceMetric("config-service",   3.0,  16.0, 0.00,   9.0,   6.0),   # Quiet. This is the answer.
                ServiceMetric("auth-service",    74.0,  46.0, 0.88,  90.0, 860.0),
                ServiceMetric("frontend",        36.0,  40.0, 0.47, 190.0, 1100.0),
                ServiceMetric("api-gateway",     60.0,  55.0, 0.40, 220.0, 1050.0),
                ServiceMetric("payment-service", 42.0,  50.0, 0.32, 410.0,  280.0),
                ServiceMetric("redis-cache",     22.0,  65.0, 0.00,   3.0,  900.0),
                ServiceMetric("postgres-replica",23.0,  60.0, 0.00,  42.0,  380.0),
            ],
            "dependency_graph": {
                "frontend":          ["api-gateway"],
                "api-gateway":       ["auth-service", "payment-service"],
                "auth-service":      ["config-service", "redis-cache"],
                "payment-service":   ["config-service", "postgres-primary"],
                "config-service":    [],
                "redis-cache":       [],
                "postgres-primary":  ["postgres-replica"],
                "postgres-replica":  [],
                "cdn":               [],
            },
            # HIDDEN evidence — only accessible via run_diagnostic
            "hidden_diagnostic": {
                "service":    "config-service",
                "query_type": "recent_logs",
                "reveals":    "Deploy v2.4.1 at T-01:30: schema migration renamed 'auth.jwt_secret'→'auth.jwt.secret' and 'payment.stripe_key'→'payment.stripe.api_key'. Services using old key names fail silently on lookup.",
            },
        },
        # Variant 1: secret-manager rotation gone wrong
        {
            "root_cause_service": "secret-manager",
            "root_cause_type":    "bad_deploy",
            "correct_action":     "investigate_db",
            "alerts": [
                Alert("order-service",   "P1", "Database connection failures 55%","error_rate",       0.55, 0.05),
                Alert("api-gateway",     "P1", "Backend error rate elevated",      "error_rate",       0.38, 0.01),
                Alert("auth-service",    "P2", "LDAP bind failures",               "error_rate",       0.24, 0.05),
                Alert("analytics-svc",   "P2", "Data pipeline errors",             "error_rate",       0.19, 0.05),
            ],
            "logs": [
                # NO secret rotation log — hidden in diagnostic_data
                LogEntry("T+00:00", "ERROR", "order-service",   "PG auth failed: password authentication failed for user 'app'"),
                LogEntry("T+00:01", "ERROR", "auth-service",    "LDAP bind error: invalid credentials"),
                LogEntry("T+00:02", "ERROR", "analytics-svc",   "DB connection refused: authentication failure"),
                LogEntry("T+00:03", "WARN",  "api-gateway",     "Multiple backend services returning 500"),
                # Red herrings
                LogEntry("T-05:00", "INFO",  "order-service",   "Traffic spike: 2x normal order volume"),
                LogEntry("T-03:00", "INFO",  "postgres-primary","Checkpoint: pg_wal size 2.1GB"),
            ],
            "metrics": [
                ServiceMetric("secret-manager",   2.0,  12.0, 0.00,   6.0,   4.0),  # Quiet. This is the answer.
                ServiceMetric("order-service",    58.0,  55.0, 0.55, 2800.0, 170.0),
                ServiceMetric("auth-service",     45.0,  48.0, 0.24, 1400.0, 200.0),
                ServiceMetric("analytics-svc",    38.0,  44.0, 0.19, 1800.0, 140.0),
                ServiceMetric("api-gateway",      62.0,  54.0, 0.38,  420.0, 900.0),
                ServiceMetric("postgres-primary", 55.0,  60.0, 0.00,   22.0, 380.0),
            ],
            "dependency_graph": {
                "api-gateway":       ["order-service", "auth-service", "analytics-svc"],
                "order-service":     ["secret-manager", "postgres-primary"],
                "auth-service":      ["secret-manager"],
                "analytics-svc":     ["secret-manager", "postgres-primary"],
                "secret-manager":    [],
                "postgres-primary":  [],
            },
            "hidden_diagnostic": {
                "service":    "secret-manager",
                "query_type": "recent_logs",
                "reveals":    "Automated secret rotation at T-00:02: rotated DB passwords and LDAP bind credentials. New secrets in vault but services not restarted — cached old credentials. Action: investigate_db to confirm secret state, then restart affected services with new credentials.",
            },
        },
        # Variant 2: feature-flag service bad rollout
        {
            "root_cause_service": "feature-flag-service",
            "root_cause_type":    "bad_deploy",
            "correct_action":     "rollback",
            "alerts": [
                Alert("checkout-service", "P1", "Checkout failure rate 44%",     "error_rate", 0.44, 0.05),
                Alert("pricing-service",  "P1", "Price calculation errors 38%",  "error_rate", 0.38, 0.05),
                Alert("api-gateway",      "P2", "Elevated 500 error rate",        "error_rate", 0.28, 0.01),
            ],
            "logs": [
                # NO flag rollout log — hidden in diagnostic_data
                LogEntry("T+00:00", "ERROR", "checkout-service", "FeatureFlagException: flag 'new_pricing_v2' returned null"),
                LogEntry("T+00:01", "ERROR", "pricing-service",  "NullPointerException in pricing engine — flag value missing"),
                LogEntry("T+00:02", "WARN",  "api-gateway",      "Elevated 500s from checkout and pricing backends"),
                # Red herrings
                LogEntry("T-10:00", "INFO",  "checkout-service", "A/B test 'checkout_flow_v3': 50% traffic split active"),
                LogEntry("T-02:00", "INFO",  "postgres-primary", "Autovacuum: ANALYZE public.products complete"),
            ],
            "metrics": [
                ServiceMetric("feature-flag-service", 2.0,  10.0, 0.00,   5.0,  3.0),  # Quiet. This is the answer.
                ServiceMetric("checkout-service",    50.0,  52.0, 0.44, 2200.0, 220.0),
                ServiceMetric("pricing-service",     44.0,  48.0, 0.38, 1800.0, 200.0),
                ServiceMetric("api-gateway",         58.0,  52.0, 0.28,  350.0, 880.0),
                ServiceMetric("postgres-primary",    45.0,  55.0, 0.00,   18.0, 340.0),
            ],
            "dependency_graph": {
                "api-gateway":         ["checkout-service", "pricing-service"],
                "checkout-service":    ["feature-flag-service", "pricing-service", "postgres-primary"],
                "pricing-service":     ["feature-flag-service", "postgres-primary"],
                "feature-flag-service":[],
                "postgres-primary":    [],
            },
            "hidden_diagnostic": {
                "service":    "feature-flag-service",
                "query_type": "recent_logs",
                "reveals":    "SDK update v4.0 deployed at T-00:05: breaking change — flag evaluation now returns null instead of default value when flag key missing. Services expecting boolean defaults now throw NullPointerException.",
            },
        },
    ]
    return base


def _hard_cascade_oom_variants() -> List[Dict]:
    """
    Hard task 2: PARTIAL OBSERVABILITY — upstream silent failure.

    CRITICAL DESIGN:
    - Initial observation shows ONLY downstream alerts — root cause service has NO alert.
    - The root cause service is silent in both alerts AND metrics (memory not shown).
    - Logs show OOM from root cause but also an ambiguous recent config change.
    - The initial observation is genuinely ambiguous between OOM and config-related failure.
    - Diagnostic data (metrics_history) proves it is a leak: steady growth, not a spike.
    - WITHOUT the hidden diagnostic, agent cannot distinguish memory_leak from bad_deploy.
    - WITH the diagnostic, the growth curve makes the diagnosis unambiguous.
    """
    base = [
        # Variant 0: ml-inference OOM — rollback model version is the correct fix
        {
            "root_cause_service": "ml-inference-service",
            "root_cause_type":    "memory_leak",
            "correct_action":     "rollback",
            "alerts": [
                Alert("checkout-service",   "P1", "Error rate 38%",          "error_rate",     0.38, 0.05),
                Alert("recommendation-svc", "P1", "Service unavailable",      "availability",   0.00, 0.99),
                Alert("api-gateway",        "P2", "Latency p99 3400ms",       "latency_p99_ms", 3400.0, 500.0),
            ],
            "logs": [
                # Root cause service logs are MINIMAL — only downstream effects visible
                LogEntry("T+00:01", "ERROR", "recommendation-svc",   "Upstream ml-inference-service not responding"),
                LogEntry("T+00:02", "ERROR", "checkout-service",     "Degraded mode: recommendations unavailable"),
                # Red herrings — model deploy 8 min ago looks suspicious
                LogEntry("T-08:00", "INFO",  "ml-inference-service", "Model rec-v3.2 deployment started"),
                LogEntry("T-07:55", "INFO",  "ml-inference-service", "Model rec-v3.2 ready — switching traffic"),
                LogEntry("T-15:00", "INFO",  "api-gateway",          "Rate limit config updated: burst 500→1000 rps"),
            ],
            "metrics": [
                ServiceMetric("ml-inference-service",  0.0,  0.0, 0.00,    0.0,   0.0),
                ServiceMetric("recommendation-svc",    0.0,  0.0, 1.00,    0.0,   0.0),
                ServiceMetric("checkout-service",     48.0, 50.0, 0.38, 1800.0, 310.0),
                ServiceMetric("api-gateway",          44.0, 47.0, 0.12, 3400.0, 800.0),
                ServiceMetric("postgres-primary",     66.0, 58.0, 0.00,   52.0, 400.0),
            ],
            "dependency_graph": {
                "api-gateway":          ["checkout-service", "recommendation-svc"],
                "checkout-service":     ["recommendation-svc", "postgres-primary"],
                "recommendation-svc":   ["ml-inference-service"],
                "ml-inference-service": [],
                "postgres-primary":     [],
            },
            "hidden_diagnostic": {
                "service":    "ml-inference-service",
                "query_type": "metrics_history",
                "reveals":    "Memory grew steadily: 2.1GB→3.4GB→5.2GB→6.8GB→9.8GB over 8 minutes. Not a model-load spike — this is a memory leak in model rec-v3.2 cache eviction. Correct action: rollback to rec-v3.1 (prevents recurrence). Restart alone would leak again on next deploy.",
            },
        },
        # Variant 1: stream-processor OOM — rollback window config is the fix
        {
            "root_cause_service": "stream-processor",
            "root_cause_type":    "memory_leak",
            "correct_action":     "rollback",
            "alerts": [
                Alert("analytics-dashboard", "P1", "Data staleness alert",        "data_lag_seconds", 840.0, 60.0),
                Alert("alert-engine",        "P1", "Alert evaluation failures",    "error_rate",       0.44, 0.05),
                Alert("kafka-consumer",      "P2", "Consumer lag growing",         "consumer_lag",     48000.0, 1000.0),
            ],
            "logs": [
                # Root cause: only downstream effects visible
                LogEntry("T+00:01", "WARN",  "kafka-consumer",     "stream-processor not consuming — consumer lag growing"),
                LogEntry("T+00:02", "ERROR", "alert-engine",       "Cannot connect to stream-processor — metrics unavailable"),
                # Red herrings — config change 8 min ago
                LogEntry("T-08:00", "INFO",  "stream-processor",   "Tumbling window size updated: 5min → 10min"),
                LogEntry("T-20:00", "INFO",  "kafka-consumer",     "Topic rebalance: 3 partitions reassigned (routine)"),
                LogEntry("T-30:00", "INFO",  "alert-engine",       "Alert rule threshold updated: p99 500ms → 400ms"),
            ],
            "metrics": [
                ServiceMetric("stream-processor",     0.0,  0.0, 0.00,  0.0,   0.0),
                ServiceMetric("kafka-consumer",      22.0, 30.0, 0.08, 480.0,  140.0),
                ServiceMetric("alert-engine",        35.0, 42.0, 0.44, 2400.0, 80.0),
                ServiceMetric("analytics-dashboard", 28.0, 36.0, 0.22, 1800.0, 60.0),
            ],
            "dependency_graph": {
                "analytics-dashboard": ["stream-processor", "alert-engine"],
                "alert-engine":        ["stream-processor"],
                "kafka-consumer":      ["stream-processor"],
                "stream-processor":    [],
            },
            "hidden_diagnostic": {
                "service":    "stream-processor",
                "query_type": "metrics_history",
                "reveals":    "Window state grew continuously for 12 minutes after the 5min→10min window size change. Heap was not resized to match. Correct action: rollback the window config change to 5min (immediate fix). Restart alone will re-accumulate state once traffic resumes.",
            },
        },
        # Variant 2: log-aggregator OOM
        {
            "root_cause_service": "log-aggregator",
            "root_cause_type":    "memory_leak",
            "correct_action":     "restart_service",
            "alerts": [
                Alert("monitoring-dashboard", "P1", "Metrics missing for 12 services","missing_services", 12.0, 0.0),
                Alert("alerting-service",     "P1", "Alert pipeline failures",         "error_rate",       0.52, 0.05),
            ],
            "logs": [
                # Root cause: only downstream effects visible
                LogEntry("T+00:01", "ERROR", "alerting-service",   "Cannot reach log-aggregator — connection refused"),
                LogEntry("T+00:02", "ERROR", "monitoring-dashboard","Log stream interrupted — 12 services now dark"),
                # Red herrings — new log source and config change
                LogEntry("T-08:00", "INFO",  "log-aggregator",     "New log source onboarded: billing-service"),
                LogEntry("T-15:00", "INFO",  "alerting-service",   "Alert threshold updated: latency 500ms→300ms"),
                LogEntry("T-30:00", "INFO",  "monitoring-dashboard","Dashboard refresh rate increased: 30s→10s"),
            ],
            "metrics": [
                ServiceMetric("log-aggregator",        0.0,  0.0, 0.00,    0.0,   0.0),
                ServiceMetric("alerting-service",     38.0, 44.0, 0.52, 2200.0,  90.0),
                ServiceMetric("monitoring-dashboard", 25.0, 38.0, 0.30, 1400.0,  70.0),
            ],
            "dependency_graph": {
                "monitoring-dashboard": ["log-aggregator", "alerting-service"],
                "alerting-service":     ["log-aggregator"],
                "log-aggregator":       [],
            },
            "hidden_diagnostic": {
                "service":    "log-aggregator",
                "query_type": "metrics_history",
                "reveals":    "Buffer grew 400MB/minute after billing-service was added as a log source 8 minutes ago. Billing-service emits 40x more log volume than typical services. Buffer flush rate cannot keep up — heap consumed. Restart clears buffer; permanent fix is log sampling or buffer resize.",
            },
        },
    ]
    return base


# ─────────────────────────────────────────────────────────────────
# Shared diagnostic data for all services
# Every service queryable — no dead ends
# ─────────────────────────────────────────────────────────────────

def _build_generic_diagnostic(svc: str, is_healthy: bool, is_root_cause: bool) -> Dict:
    if is_root_cause:
        return {}  # filled by scenario-specific data
    if is_healthy:
        return {
            "metrics_history": f"{svc}: all metrics within normal bounds. No anomaly in past 30 minutes.",
            "recent_logs":     f"{svc}: no errors. Routine operational activity only.",
            "connections":     f"{svc}: connection counts normal. No timeouts.",
        }
    return {
        "metrics_history": f"{svc}: elevated metrics correlate with upstream failure. Not the root cause.",
        "recent_logs":     f"{svc}: errors are downstream effects — requests failing because dependency is down.",
        "connections":     f"{svc}: connection issues traceable to upstream service unavailability.",
    }


# ─────────────────────────────────────────────────────────────────
# Scenario registry
# ─────────────────────────────────────────────────────────────────

_FAMILIES = {
    "easy_memory_leak":         _easy_memory_leak_variants,
    "easy_cpu_spike":           _easy_cpu_spike_variants,
    "medium_db_contention":     _medium_db_contention_variants,
    "medium_network":           _medium_network_variants,
    "hard_bad_deploy":          _hard_bad_deploy_variants,
    "hard_cascade_oom":         _hard_cascade_oom_variants,
}

TASK_IDS    = list(_FAMILIES.keys())
EASY_TASKS  = [t for t in TASK_IDS if t.startswith("easy")]
MEDIUM_TASKS= [t for t in TASK_IDS if t.startswith("medium")]
HARD_TASKS  = [t for t in TASK_IDS if t.startswith("hard")]


def get_scenario(task_id: str, seed: int = 42) -> Dict[str, Any]:
    """
    Return a fully-built scenario dict for this task_id + seed.

    Variant selection: seed % 3 picks which rotation is active.
    Metric jitter:     seed-based ±5% noise on numeric values.
    Reproducible:      same task_id + seed = identical scenario every time.
    """
    if task_id not in _FAMILIES:
        raise ValueError(f"Unknown task '{task_id}'. Valid: {TASK_IDS}")

    # Derive difficulty and task_description from task_id
    if task_id.startswith("easy"):
        difficulty = "easy"
    elif task_id.startswith("medium"):
        difficulty = "medium"
    else:
        difficulty = "hard"

    _task_descriptions = {
        "easy_memory_leak":     "A memory alert fired on one service. Diagnose root cause and recommend remediation.",
        "easy_cpu_spike":       "CPU saturation alert. Users reporting slowness. Find the root cause.",
        "medium_db_contention": "Multiple services alerting simultaneously. Symptoms are downstream — trace to root cause.",
        "medium_network":       "Intermittent failures across services. Some requests succeed, some fail. Diagnose.",
        "hard_bad_deploy":      "Mass P1 incident. Multiple services degraded. Root cause is a quiet upstream service. Critical evidence requires investigation.",
        "hard_cascade_oom":     "P1 incident: loud downstream failures. Root cause is an upstream service with a memory issue. Precise failure mechanism requires investigation.",
    }

    rng      = random.Random(seed)
    variants = _FAMILIES[task_id]()
    variant  = variants[seed % len(variants)]
    scenario = deepcopy(variant)

    # Inject metadata
    scenario["difficulty"]        = difficulty
    scenario["task_description"]  = _task_descriptions.get(task_id, f"Diagnose the {task_id} incident.")

    # Jitter numeric metric values ±5%
    for m in scenario["metrics"]:
        m.cpu_percent    = round(min(99.9, m.cpu_percent    * rng.uniform(0.95, 1.05)), 1)
        m.memory_percent = round(min(99.9, m.memory_percent * rng.uniform(0.97, 1.03)), 1)
        m.error_rate     = round(min(1.00, m.error_rate     * rng.uniform(0.90, 1.10)), 3)
        m.latency_p99_ms = round(m.latency_p99_ms           * rng.uniform(0.93, 1.07), 1)

    # Jitter alert values ±5%
    for a in scenario["alerts"]:
        if a.value is not None:
            a.value = round(a.value * rng.uniform(0.95, 1.05), 2)

    # Build diagnostic_data — ensure every service in dependency_graph is queryable
    if "diagnostic_data" not in scenario:
        scenario["diagnostic_data"] = {}

    root_svc = scenario["root_cause_service"]
    for svc in scenario["dependency_graph"]:
        if svc not in scenario["diagnostic_data"]:
            is_root   = (svc == root_svc)
            alerting  = any(
                (a.service if hasattr(a, 'service') else a["service"]) == svc
                for a in scenario["alerts"]
            )
            scenario["diagnostic_data"][svc] = _build_generic_diagnostic(
                svc, is_healthy=not alerting, is_root_cause=is_root
            )

    # For hard tasks: inject the hidden evidence into diagnostic_data
    if "hidden_diagnostic" in scenario:
        hd = scenario["hidden_diagnostic"]
        if hd["service"] not in scenario["diagnostic_data"]:
            scenario["diagnostic_data"][hd["service"]] = {}
        scenario["diagnostic_data"][hd["service"]][hd["query_type"]] = hd["reveals"]

    return scenario
