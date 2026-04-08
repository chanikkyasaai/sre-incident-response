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
                Alert(service="payment-service", severity="P2", message="Memory usage critical", metric="memory_percent", value=93.8, threshold=85.0),
                Alert(service="api-gateway", severity="P3", message="Upstream error rate elevated", metric="error_rate", value=0.04, threshold=0.02),
            ],
            "logs": [
                LogEntry(timestamp="T+00:00", level="WARN", service="payment-service", message="JVM heap: 1.1GB / 2GB"),
                LogEntry(timestamp="T+01:00", level="WARN", service="payment-service", message="JVM heap: 1.5GB / 2GB — GC pressure increasing"),
                LogEntry(timestamp="T+02:00", level="WARN", service="payment-service", message="JVM heap: 1.9GB / 2GB — GC running continuously"),
                LogEntry(timestamp="T+03:00", level="ERROR", service="payment-service", message="OutOfMemoryError: Java heap space"),
                LogEntry(timestamp="T+03:01", level="ERROR", service="payment-service", message="GC overhead limit exceeded"),
                LogEntry(timestamp="T+03:05", level="WARN", service="api-gateway", message="503 upstream: payment-service not responding"),
                # Red herring: looks like auth is involved but it isn't
                LogEntry(timestamp="T+02:45", level="INFO", service="auth-service", message="Token cache eviction: 2400 entries pruned (scheduled)"),
            ],
            "metrics": [
                ServiceMetric(service="payment-service", cpu_percent=46.0, memory_percent=93.8, error_rate=0.14, latency_p99_ms=410.0, requests_per_second=135.0),
                ServiceMetric(service="api-gateway", cpu_percent=28.0, memory_percent=36.0, error_rate=0.04, latency_p99_ms=88.0, requests_per_second=480.0),
                ServiceMetric(service="auth-service", cpu_percent=20.0, memory_percent=34.0, error_rate=0.00, latency_p99_ms=38.0, requests_per_second=520.0),
                ServiceMetric(service="postgres-primary", cpu_percent=52.0, memory_percent=58.0, error_rate=0.00, latency_p99_ms=14.0, requests_per_second=175.0),
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
                Alert(service="auth-service", severity="P2", message="Memory usage critical", metric="memory_percent", value=91.4, threshold=85.0),
                Alert(service="api-gateway", severity="P3", message="Authentication latency high", metric="latency_p99_ms", value=1200.0, threshold=300.0),
            ],
            "logs": [
                LogEntry(timestamp="T+00:00", level="WARN", service="auth-service", message="Session store: 800MB / 1GB"),
                LogEntry(timestamp="T+01:30", level="WARN", service="auth-service", message="Session store: 1.1GB — not evicting expired sessions"),
                LogEntry(timestamp="T+02:30", level="ERROR", service="auth-service", message="OutOfMemoryError: unable to allocate session object"),
                LogEntry(timestamp="T+02:35", level="WARN", service="api-gateway", message="Auth timeout — requests queuing"),
                # Red herring: payment looks suspicious but is fine
                LogEntry(timestamp="T+02:00", level="INFO", service="payment-service", message="Stripe webhook received: payment_intent.succeeded"),
            ],
            "metrics": [
                ServiceMetric(service="auth-service", cpu_percent=55.0, memory_percent=91.4, error_rate=0.18, latency_p99_ms=1200.0, requests_per_second=200.0),
                ServiceMetric(service="api-gateway", cpu_percent=35.0, memory_percent=38.0, error_rate=0.06, latency_p99_ms=420.0, requests_per_second=490.0),
                ServiceMetric(service="payment-service", cpu_percent=22.0, memory_percent=40.0, error_rate=0.01, latency_p99_ms=85.0, requests_per_second=180.0),
                ServiceMetric(service="postgres-primary", cpu_percent=48.0, memory_percent=55.0, error_rate=0.00, latency_p99_ms=11.0, requests_per_second=170.0),
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
                Alert(service="worker-service", severity="P2", message="Memory usage critical", metric="memory_percent", value=95.1, threshold=85.0),
                Alert(service="job-queue", severity="P3", message="Job processing latency", metric="latency_p99_ms", value=8200.0, threshold=1000.0),
            ],
            "logs": [
                LogEntry(timestamp="T+00:00", level="WARN", service="worker-service", message="Heap: 3.1GB / 4GB — large object allocation"),
                LogEntry(timestamp="T+01:00", level="WARN", service="worker-service", message="Heap: 3.7GB / 4GB — GC unable to reclaim"),
                LogEntry(timestamp="T+02:00", level="ERROR", service="worker-service", message="OutOfMemoryError: GC overhead limit exceeded"),
                LogEntry(timestamp="T+02:05", level="WARN", service="job-queue", message="Worker not polling — jobs backing up"),
                # Red herring: redis looks related
                LogEntry(timestamp="T+01:30", level="INFO", service="redis-cache", message="Memory: 2.1GB / 8GB — within normal bounds"),
            ],
            "metrics": [
                ServiceMetric(service="worker-service", cpu_percent=44.0, memory_percent=95.1, error_rate=0.22, latency_p99_ms=8200.0, requests_per_second=60.0),
                ServiceMetric(service="job-queue", cpu_percent=18.0, memory_percent=32.0, error_rate=0.08, latency_p99_ms=4100.0, requests_per_second=140.0),
                ServiceMetric(service="redis-cache", cpu_percent=12.0, memory_percent=26.0, error_rate=0.00, latency_p99_ms=2.0, requests_per_second=900.0),
                ServiceMetric(service="postgres-primary", cpu_percent=50.0, memory_percent=55.0, error_rate=0.00, latency_p99_ms=10.0, requests_per_second=200.0),
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
                Alert(service="image-processor", severity="P2", message="CPU usage critical", metric="cpu_percent", value=97.0, threshold=80.0),
                Alert(service="upload-service", severity="P3", message="Processing latency elevated", metric="latency_p99_ms", value=4800.0, threshold=1000.0),
            ],
            "logs": [
                LogEntry(timestamp="T+00:00", level="INFO", service="image-processor", message="Batch triggered: compress 9200 images for CDN"),
                LogEntry(timestamp="T+00:10", level="WARN", service="image-processor", message="All 16 worker threads active"),
                LogEntry(timestamp="T+01:00", level="WARN", service="image-processor", message="Job queue depth: 3100 — falling behind"),
                LogEntry(timestamp="T+02:00", level="ERROR", service="upload-service", message="Timeout on image-processor after 30s"),
                # Red herring: postgres has elevated writes
                LogEntry(timestamp="T+01:30", level="INFO", service="postgres-primary", message="Checkpoint: 6200 buffers written (elevated — high write load)"),
            ],
            "metrics": [
                ServiceMetric(service="image-processor", cpu_percent=97.0, memory_percent=54.0, error_rate=0.06, latency_p99_ms=9100.0, requests_per_second=18.0),
                ServiceMetric(service="upload-service", cpu_percent=32.0, memory_percent=38.0, error_rate=0.09, latency_p99_ms=4800.0, requests_per_second=110.0),
                ServiceMetric(service="postgres-primary", cpu_percent=68.0, memory_percent=55.0, error_rate=0.00, latency_p99_ms=18.0, requests_per_second=220.0),
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
                Alert(service="report-generator", severity="P2", message="CPU usage critical", metric="cpu_percent", value=96.0, threshold=80.0),
                Alert(service="analytics-api", severity="P3", message="Report API latency elevated", metric="latency_p99_ms", value=6200.0, threshold=500.0),
            ],
            "logs": [
                LogEntry(timestamp="T+00:00", level="INFO", service="report-generator", message="Monthly report job started: 140 reports"),
                LogEntry(timestamp="T+00:30", level="WARN", service="report-generator", message="CPU-bound aggregation: 96% utilisation"),
                LogEntry(timestamp="T+01:30", level="WARN", service="report-generator", message="Report queue: 88 pending"),
                LogEntry(timestamp="T+02:30", level="ERROR", service="analytics-api", message="Timeout waiting for report-generator: 60s"),
                # Red herring: cache miss
                LogEntry(timestamp="T+01:00", level="INFO", service="redis-cache", message="Cache miss rate: 18% (elevated — cold start after deploy)"),
            ],
            "metrics": [
                ServiceMetric(service="report-generator", cpu_percent=96.0, memory_percent=60.0, error_rate=0.04, latency_p99_ms=6200.0, requests_per_second=12.0),
                ServiceMetric(service="analytics-api", cpu_percent=30.0, memory_percent=42.0, error_rate=0.07, latency_p99_ms=3100.0, requests_per_second=80.0),
                ServiceMetric(service="redis-cache", cpu_percent=15.0, memory_percent=30.0, error_rate=0.00, latency_p99_ms=3.0, requests_per_second=600.0),
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
                Alert(service="ml-training-job", severity="P2", message="CPU usage critical", metric="cpu_percent", value=98.0, threshold=80.0),
                Alert(service="model-api", severity="P3", message="Model serving latency high", metric="latency_p99_ms", value=3400.0, threshold=500.0),
            ],
            "logs": [
                LogEntry(timestamp="T+00:00", level="INFO", service="ml-training-job", message="Retraining started: 2.1M samples, 40 epochs"),
                LogEntry(timestamp="T+00:20", level="WARN", service="ml-training-job", message="CPU: 98% — all cores allocated to training"),
                LogEntry(timestamp="T+01:00", level="WARN", service="model-api", message="Inference requests queuing behind training job"),
                LogEntry(timestamp="T+02:00", level="ERROR", service="model-api", message="Request timeout: training job monopolising CPU"),
                # Red herring: disk IO
                LogEntry(timestamp="T+00:45", level="INFO", service="ml-training-job", message="Checkpoint saved: 1.8GB written to disk"),
            ],
            "metrics": [
                ServiceMetric(service="ml-training-job", cpu_percent=98.0, memory_percent=72.0, error_rate=0.00, latency_p99_ms=2100.0, requests_per_second=5.0),
                ServiceMetric(service="model-api", cpu_percent=55.0, memory_percent=48.0, error_rate=0.11, latency_p99_ms=3400.0, requests_per_second=90.0),
                ServiceMetric(service="postgres-primary", cpu_percent=40.0, memory_percent=50.0, error_rate=0.00, latency_p99_ms=12.0, requests_per_second=150.0),
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
                Alert(service="order-service", severity="P1", message="Error rate 21%", metric="error_rate", value=21, threshold=0.21),
                Alert(service="inventory-service", severity="P2", message="Latency p99 elevated", metric="latency_p99_ms", value=2800.0, threshold=500.0),
                Alert(service="postgres-primary", severity="P2", message="Lock wait queue length high", metric="lock_wait_count", value=48.0, threshold=5.0),
                Alert(service="redis-cache", severity="P3", message="Cache hit rate below target", metric="cache_hit_rate", value=0.88, threshold=0.92),
            ],
            "logs": [
                LogEntry(timestamp="T+00:01", level="ERROR", service="order-service", message="QueryTimeoutException after 30s — table: orders"),
                LogEntry(timestamp="T+00:02", level="ERROR", service="inventory-service", message="HikariPool: no connection available after 30s"),
                LogEntry(timestamp="T+00:04", level="WARN", service="postgres-primary", message="PID 14821 waiting on ShareLock — blocked by PID 14803"),
                LogEntry(timestamp="T+00:05", level="WARN", service="postgres-primary", message="autovacuum on public.orders: 1.2M dead tuples"),
                LogEntry(timestamp="T+00:06", level="WARN", service="postgres-primary", message="VACUUM holding ExclusiveLock — 44 queries waiting"),
                LogEntry(timestamp="T+00:08", level="INFO", service="order-service", message="Retry attempt 3/3 failed"),
                # Red herring: redis expiry happened before incident
                LogEntry(timestamp="T-05:00", level="INFO", service="redis-cache", message="Scheduled expiry: 11400 keys removed"),
            ],
            "metrics": [
                ServiceMetric(service="order-service", cpu_percent=56.0, memory_percent=60.0, error_rate=0.21, latency_p99_ms=3100.0, requests_per_second=190.0),
                ServiceMetric(service="inventory-service", cpu_percent=44.0, memory_percent=50.0, error_rate=0.11, latency_p99_ms=2800.0, requests_per_second=170.0),
                ServiceMetric(service="postgres-primary", cpu_percent=89.0, memory_percent=70.0, error_rate=0.00, latency_p99_ms=0.0, requests_per_second=0.0),
                ServiceMetric(service="redis-cache", cpu_percent=13.0, memory_percent=28.0, error_rate=0.00, latency_p99_ms=2.0, requests_per_second=820.0),
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
                Alert(service="notification-svc", severity="P1", message="Message delivery failures", metric="delivery_fail_rate", value=0.34, threshold=0.05),
                Alert(service="event-processor", severity="P2", message="Event processing latency", metric="latency_p99_ms", value=4200.0, threshold=500.0),
                Alert(service="message-broker", severity="P2", message="Connection pool exhausted", metric="active_connections", value=99.0, threshold=95.0),
            ],
            "logs": [
                LogEntry(timestamp="T+00:01", level="ERROR", service="notification-svc", message="AMQP connection refused: pool exhausted"),
                LogEntry(timestamp="T+00:02", level="ERROR", service="event-processor", message="Cannot acquire broker connection after 10s"),
                LogEntry(timestamp="T+00:04", level="WARN", service="message-broker", message="Connection pool: 99/100 — refusing new connections"),
                LogEntry(timestamp="T+00:05", level="WARN", service="message-broker", message="Long-running consumer detected: PID 8821 open 2h"),
                LogEntry(timestamp="T+00:07", level="INFO", service="notification-svc", message="Retry 2/3"),
                # Red herring: scheduled maintenance window started
                LogEntry(timestamp="T-10:00", level="INFO", service="monitoring", message="Maintenance window opened for network switch upgrade"),
            ],
            "metrics": [
                ServiceMetric(service="notification-svc", cpu_percent=38.0, memory_percent=44.0, error_rate=0.34, latency_p99_ms=3200.0, requests_per_second=160.0),
                ServiceMetric(service="event-processor", cpu_percent=42.0, memory_percent=48.0, error_rate=0.18, latency_p99_ms=4200.0, requests_per_second=140.0),
                ServiceMetric(service="message-broker", cpu_percent=71.0, memory_percent=65.0, error_rate=0.00, latency_p99_ms=0.0, requests_per_second=0.0),
                ServiceMetric(service="monitoring", cpu_percent=8.0, memory_percent=20.0, error_rate=0.00, latency_p99_ms=5.0, requests_per_second=10.0),
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
                Alert(service="session-service", severity="P1", message="Auth failure rate 28%", metric="error_rate", value=28, threshold=0.28),
                Alert(service="cart-service", severity="P2", message="Cart save failures", metric="error_rate", value=0.19, threshold=0.05),
                Alert(service="redis-cluster", severity="P2", message="Connection limit approaching", metric="active_connections", value=498.0, threshold=500.0),
            ],
            "logs": [
                LogEntry(timestamp="T+00:01", level="ERROR", service="session-service", message="Redis MAXCLIENTS reached — connection refused"),
                LogEntry(timestamp="T+00:02", level="ERROR", service="cart-service", message="Redis connection timeout after 5s"),
                LogEntry(timestamp="T+00:04", level="WARN", service="redis-cluster", message="Connected clients: 498/500"),
                LogEntry(timestamp="T+00:05", level="WARN", service="redis-cluster", message="Client with idle time >3600s detected: 80 connections"),
                LogEntry(timestamp="T+00:07", level="INFO", service="session-service", message="Retrying with new connection"),
                # Red herring: recent deploy with no issues
                LogEntry(timestamp="T-30:00", level="INFO", service="cart-service", message="Deploy v3.1.0 completed — all health checks passed"),
            ],
            "metrics": [
                ServiceMetric(service="session-service", cpu_percent=40.0, memory_percent=45.0, error_rate=0.28, latency_p99_ms=2400.0, requests_per_second=200.0),
                ServiceMetric(service="cart-service", cpu_percent=35.0, memory_percent=42.0, error_rate=0.19, latency_p99_ms=1800.0, requests_per_second=180.0),
                ServiceMetric(service="redis-cluster", cpu_percent=60.0, memory_percent=72.0, error_rate=0.00, latency_p99_ms=0.0, requests_per_second=0.0),
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
                Alert(service="checkout-service", severity="P1", message="Error rate 34%", metric="error_rate", value=34, threshold=0.34),
                Alert(service="payment-service", severity="P2", message="Connection refused errors", metric="conn_refused", value=120.0, threshold=0.0),
            ],
            "logs": [
                LogEntry(timestamp="T-00:10", level="INFO", service="service-mesh", message="Config reload triggered by operator"),
                LogEntry(timestamp="T+00:01", level="ERROR", service="checkout-service", message="ECONNREFUSED payment-service:8080"),
                LogEntry(timestamp="T+00:02", level="ERROR", service="checkout-service", message="ECONNREFUSED payment-service:8080"),
                LogEntry(timestamp="T+00:02", level="INFO", service="checkout-service", message="Request succeeded on retry — intermittent"),
                LogEntry(timestamp="T+00:04", level="WARN", service="service-mesh", message="iptables flush in progress — 28s window"),
                LogEntry(timestamp="T+00:32", level="INFO", service="service-mesh", message="Config reload complete — routing restored"),
                LogEntry(timestamp="T+00:33", level="INFO", service="checkout-service", message="Error rate returning to baseline"),
                # Red herring: unrelated DB activity
                LogEntry(timestamp="T-05:00", level="INFO", service="postgres-primary", message="Scheduled ANALYZE on public.orders complete"),
            ],
            "metrics": [
                ServiceMetric(service="checkout-service", cpu_percent=38.0, memory_percent=44.0, error_rate=0.34, latency_p99_ms=1200.0, requests_per_second=290.0),
                ServiceMetric(service="payment-service", cpu_percent=35.0, memory_percent=40.0, error_rate=0.18, latency_p99_ms=800.0, requests_per_second=240.0),
                ServiceMetric(service="service-mesh", cpu_percent=4.0, memory_percent=8.0, error_rate=0.00, latency_p99_ms=0.0, requests_per_second=0.0),
                ServiceMetric(service="postgres-primary", cpu_percent=28.0, memory_percent=52.0, error_rate=0.00, latency_p99_ms=14.0, requests_per_second=190.0),
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
                Alert(service="user-service", severity="P1", message="Error rate 29%", metric="error_rate", value=29, threshold=0.29),
                Alert(service="search-service", severity="P2", message="Connection timeout errors", metric="conn_refused", value=88.0, threshold=0.0),
            ],
            "logs": [
                LogEntry(timestamp="T-00:05", level="INFO", service="load-balancer", message="Health check config updated: interval 10s→60s"),
                LogEntry(timestamp="T+00:01", level="ERROR", service="user-service", message="Connection timeout to search-service:9200"),
                LogEntry(timestamp="T+00:02", level="INFO", service="user-service", message="Retry succeeded — intermittent failure"),
                LogEntry(timestamp="T+00:03", level="ERROR", service="search-service", message="Upstream timeout: load-balancer health check failing"),
                LogEntry(timestamp="T+00:05", level="WARN", service="load-balancer", message="Unhealthy backend removed — 1 of 3 nodes gone"),
                LogEntry(timestamp="T+00:06", level="WARN", service="load-balancer", message="Traffic concentration on 2 nodes — asymmetric load"),
                # Red herring: app deploy
                LogEntry(timestamp="T-45:00", level="INFO", service="user-service", message="Deploy v2.8.1 complete — canary 10% traffic"),
            ],
            "metrics": [
                ServiceMetric(service="user-service", cpu_percent=44.0, memory_percent=48.0, error_rate=0.29, latency_p99_ms=2200.0, requests_per_second=280.0),
                ServiceMetric(service="search-service", cpu_percent=78.0, memory_percent=55.0, error_rate=0.14, latency_p99_ms=1800.0, requests_per_second=240.0),
                ServiceMetric(service="load-balancer", cpu_percent=3.0, memory_percent=6.0, error_rate=0.00, latency_p99_ms=0.0, requests_per_second=0.0),
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
                Alert(service="recommendation-svc", severity="P1", message="External API failures 41%", metric="error_rate", value=41, threshold=0.41),
                Alert(service="product-service", severity="P2", message="Vendor API timeouts", metric="conn_refused", value=95.0, threshold=0.0),
            ],
            "logs": [
                LogEntry(timestamp="T-00:08", level="INFO", service="dns-resolver", message="Primary DNS config updated — new forwarders"),
                LogEntry(timestamp="T+00:01", level="ERROR", service="recommendation-svc", message="DNS resolution failed: vendor-api.external.com"),
                LogEntry(timestamp="T+00:02", level="INFO", service="recommendation-svc", message="Retry succeeded using cached record — intermittent"),
                LogEntry(timestamp="T+00:03", level="ERROR", service="product-service", message="getaddrinfo ENOTFOUND catalog.supplier.io"),
                LogEntry(timestamp="T+00:05", level="WARN", service="dns-resolver", message="Upstream forwarder 10.0.0.1: response timeout"),
                LogEntry(timestamp="T+00:07", level="WARN", service="dns-resolver", message="Falling back to secondary — higher latency"),
                # Red herring: certificate renewal
                LogEntry(timestamp="T-60:00", level="INFO", service="cert-manager", message="TLS certificate renewed: vendor-api.external.com"),
            ],
            "metrics": [
                ServiceMetric(service="recommendation-svc", cpu_percent=42.0, memory_percent=46.0, error_rate=0.41, latency_p99_ms=3400.0, requests_per_second=160.0),
                ServiceMetric(service="product-service", cpu_percent=38.0, memory_percent=44.0, error_rate=0.28, latency_p99_ms=2800.0, requests_per_second=200.0),
                ServiceMetric(service="dns-resolver", cpu_percent=2.0, memory_percent=4.0, error_rate=0.48, latency_p99_ms=820.0, requests_per_second=28.0),
                ServiceMetric(service="cert-manager", cpu_percent=1.0, memory_percent=3.0, error_rate=0.00, latency_p99_ms=0.0, requests_per_second=0.0),
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
                Alert(service="frontend", severity="P1", message="5xx error rate 47%", metric="error_rate", value=47, threshold=0.47),
                Alert(service="api-gateway", severity="P1", message="Request success rate 59%", metric="success_rate", value=59, threshold=0.59),
                Alert(service="auth-service", severity="P1", message="Token validation failures spiking", metric="validation_fails", value=880.0, threshold=10.0),
                Alert(service="payment-service", severity="P2", message="Elevated error rate", metric="error_rate", value=0.32, threshold=0.05),
            ],
            "logs": [
                # NO deploy log here — it's hidden in diagnostic_data
                LogEntry(timestamp="T+00:00", level="ERROR", service="auth-service", message="KeyError: config key not found — using fallback"),
                LogEntry(timestamp="T+00:00", level="WARN", service="auth-service", message="JWT secret fallback active — existing sessions invalidated"),
                LogEntry(timestamp="T+00:01", level="ERROR", service="payment-service", message="Configuration error: payment processor init failed"),
                LogEntry(timestamp="T+00:02", level="WARN", service="api-gateway", message="Auth token rejected — 401 returned to client"),
                LogEntry(timestamp="T+00:03", level="ERROR", service="frontend", message="401 flood — users being logged out"),
                # Red herrings — unlabeled, look plausible
                LogEntry(timestamp="T-02:00", level="INFO", service="postgres-replica", message="Replication lag 1.8s — 48 transactions behind"),
                LogEntry(timestamp="T-01:00", level="INFO", service="cdn", message="Edge cache invalidated for /api/v2/ — routine"),
                LogEntry(timestamp="T-00:30", level="WARN", service="redis-cache", message="Memory: 5.2GB / 8GB — approaching warning threshold"),
            ],
            "metrics": [
                ServiceMetric(service="config-service", cpu_percent=3.0, memory_percent=16.0, error_rate=0.00, latency_p99_ms=9.0, requests_per_second=6.0),   # Quiet. This is the answer.
                ServiceMetric(service="auth-service", cpu_percent=74.0, memory_percent=46.0, error_rate=0.88, latency_p99_ms=90.0, requests_per_second=860.0),
                ServiceMetric(service="frontend", cpu_percent=36.0, memory_percent=40.0, error_rate=0.47, latency_p99_ms=190.0, requests_per_second=1100.0),
                ServiceMetric(service="api-gateway", cpu_percent=60.0, memory_percent=55.0, error_rate=0.40, latency_p99_ms=220.0, requests_per_second=1050.0),
                ServiceMetric(service="payment-service", cpu_percent=42.0, memory_percent=50.0, error_rate=0.32, latency_p99_ms=410.0, requests_per_second=280.0),
                ServiceMetric(service="redis-cache", cpu_percent=22.0, memory_percent=65.0, error_rate=0.00, latency_p99_ms=3.0, requests_per_second=900.0),
                ServiceMetric(service="postgres-replica", cpu_percent=23.0, memory_percent=60.0, error_rate=0.00, latency_p99_ms=42.0, requests_per_second=380.0),
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
            "correct_action":     "rollback",
            "alerts": [
                Alert(service="order-service", severity="P1", message="Database connection failures 55%", metric="error_rate", value=55, threshold=0.55),
                Alert(service="api-gateway", severity="P1", message="Backend error rate elevated", metric="error_rate", value=0.38, threshold=0.01),
                Alert(service="auth-service", severity="P2", message="LDAP bind failures", metric="error_rate", value=0.24, threshold=0.05),
                Alert(service="analytics-svc", severity="P2", message="Data pipeline errors", metric="error_rate", value=0.19, threshold=0.05),
            ],
            "logs": [
                # NO secret rotation log — hidden in diagnostic_data
                LogEntry(timestamp="T+00:00", level="ERROR", service="order-service", message="PG auth failed: password authentication failed for user 'app'"),
                LogEntry(timestamp="T+00:01", level="ERROR", service="auth-service", message="LDAP bind error: invalid credentials"),
                LogEntry(timestamp="T+00:02", level="ERROR", service="analytics-svc", message="DB connection refused: authentication failure"),
                LogEntry(timestamp="T+00:03", level="WARN", service="api-gateway", message="Multiple backend services returning 500"),
                # Red herrings
                LogEntry(timestamp="T-05:00", level="INFO", service="order-service", message="Traffic spike: 2x normal order volume"),
                LogEntry(timestamp="T-03:00", level="INFO", service="postgres-primary", message="Checkpoint: pg_wal size 2.1GB"),
            ],
            "metrics": [
                ServiceMetric(service="secret-manager", cpu_percent=2.0, memory_percent=12.0, error_rate=0.00, latency_p99_ms=6.0, requests_per_second=4.0),  # Quiet. This is the answer.
                ServiceMetric(service="order-service", cpu_percent=58.0, memory_percent=55.0, error_rate=0.55, latency_p99_ms=2800.0, requests_per_second=170.0),
                ServiceMetric(service="auth-service", cpu_percent=45.0, memory_percent=48.0, error_rate=0.24, latency_p99_ms=1400.0, requests_per_second=200.0),
                ServiceMetric(service="analytics-svc", cpu_percent=38.0, memory_percent=44.0, error_rate=0.19, latency_p99_ms=1800.0, requests_per_second=140.0),
                ServiceMetric(service="api-gateway", cpu_percent=62.0, memory_percent=54.0, error_rate=0.38, latency_p99_ms=420.0, requests_per_second=900.0),
                ServiceMetric(service="postgres-primary", cpu_percent=55.0, memory_percent=60.0, error_rate=0.00, latency_p99_ms=22.0, requests_per_second=380.0),
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
                "reveals":    "Automated secret rotation at T-00:02: rotated DB passwords and LDAP bind credentials. New secrets propagated to vault but services not restarted — still using cached old credentials.",
            },
        },
        # Variant 2: feature-flag service bad rollout
        {
            "root_cause_service": "feature-flag-service",
            "root_cause_type":    "bad_deploy",
            "correct_action":     "rollback",
            "alerts": [
                Alert(service="checkout-service", severity="P1", message="Checkout failure rate 44%", metric="error_rate", value=44, threshold=0.44),
                Alert(service="pricing-service", severity="P1", message="Price calculation errors 38%", metric="error_rate", value=38, threshold=0.38),
                Alert(service="api-gateway", severity="P2", message="Elevated 500 error rate", metric="error_rate", value=500, threshold=0.28),
            ],
            "logs": [
                # NO flag rollout log — hidden in diagnostic_data
                LogEntry(timestamp="T+00:00", level="ERROR", service="checkout-service", message="FeatureFlagException: flag 'new_pricing_v2' returned null"),
                LogEntry(timestamp="T+00:01", level="ERROR", service="pricing-service", message="NullPointerException in pricing engine — flag value missing"),
                LogEntry(timestamp="T+00:02", level="WARN", service="api-gateway", message="Elevated 500s from checkout and pricing backends"),
                # Red herrings
                LogEntry(timestamp="T-10:00", level="INFO", service="checkout-service", message="A/B test 'checkout_flow_v3': 50% traffic split active"),
                LogEntry(timestamp="T-02:00", level="INFO", service="postgres-primary", message="Autovacuum: ANALYZE public.products complete"),
            ],
            "metrics": [
                ServiceMetric(service="feature-flag-service", cpu_percent=2.0, memory_percent=10.0, error_rate=0.00, latency_p99_ms=5.0, requests_per_second=3.0),  # Quiet. This is the answer.
                ServiceMetric(service="checkout-service", cpu_percent=50.0, memory_percent=52.0, error_rate=0.44, latency_p99_ms=2200.0, requests_per_second=220.0),
                ServiceMetric(service="pricing-service", cpu_percent=44.0, memory_percent=48.0, error_rate=0.38, latency_p99_ms=1800.0, requests_per_second=200.0),
                ServiceMetric(service="api-gateway", cpu_percent=58.0, memory_percent=52.0, error_rate=0.28, latency_p99_ms=350.0, requests_per_second=880.0),
                ServiceMetric(service="postgres-primary", cpu_percent=45.0, memory_percent=55.0, error_rate=0.00, latency_p99_ms=18.0, requests_per_second=340.0),
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
        # Variant 0: ml-inference OOM
        {
            "root_cause_service": "ml-inference-service",
            "root_cause_type":    "memory_leak",
            "correct_action":     "restart_service",
            "alerts": [
                Alert(service="checkout-service", severity="P1", message="Error rate 38%", metric="error_rate", value=38, threshold=0.38),
                Alert(service="recommendation-svc", severity="P1", message="Service unavailable", metric="availability", value=0.00, threshold=0.99),
                Alert(service="api-gateway", severity="P2", message="Latency p99 3400ms", metric="latency_p99_ms", value=3400, threshold=3400.0),
            ],
            "logs": [
                LogEntry(timestamp="T-08:00", level="INFO", service="ml-inference-service", message="Model rec-v3.2 loaded: 2.1GB"),
                LogEntry(timestamp="T-04:00", level="WARN", service="ml-inference-service", message="Model cache: 5.8GB — GC not reclaiming"),
                LogEntry(timestamp="T+00:00", level="ERROR", service="ml-inference-service", message="OutOfMemoryError: tensor allocation failed"),
                LogEntry(timestamp="T+00:01", level="ERROR", service="recommendation-svc", message="503 from ml-inference-service"),
                LogEntry(timestamp="T+00:02", level="ERROR", service="checkout-service", message="Degraded mode: recommendations unavailable"),
                # Red herrings — the model deploy looks causal but isn't the leak source
                LogEntry(timestamp="T-08:01", level="INFO", service="ml-inference-service", message="Previous model rec-v3.1 unloaded"),
                LogEntry(timestamp="T-15:00", level="INFO", service="api-gateway", message="Rate limit config updated: burst 500→1000 rps"),
            ],
            "metrics": [
                ServiceMetric(service="ml-inference-service", cpu_percent=0.0, memory_percent=0.0, error_rate=0.00, latency_p99_ms=0.0, requests_per_second=0.0),
                ServiceMetric(service="recommendation-svc", cpu_percent=0.0, memory_percent=0.0, error_rate=1.00, latency_p99_ms=0.0, requests_per_second=0.0),
                ServiceMetric(service="checkout-service", cpu_percent=48.0, memory_percent=50.0, error_rate=0.38, latency_p99_ms=1800.0, requests_per_second=310.0),
                ServiceMetric(service="api-gateway", cpu_percent=44.0, memory_percent=47.0, error_rate=0.12, latency_p99_ms=3400.0, requests_per_second=800.0),
                ServiceMetric(service="postgres-primary", cpu_percent=66.0, memory_percent=58.0, error_rate=0.00, latency_p99_ms=52.0, requests_per_second=400.0),
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
                "reveals":    "Memory grew steadily: 2.1GB→3.4GB→5.2GB→6.8GB→9.8GB over 8 minutes. Not a model-load spike — this is a leak. rec-v3.2 has a known cache eviction bug: LRU entries not being freed after inference. Restart clears cache; rollback to rec-v3.1 prevents recurrence.",
            },
        },
        # Variant 1: stream-processor OOM
        {
            "root_cause_service": "stream-processor",
            "root_cause_type":    "memory_leak",
            "correct_action":     "restart_service",
            "alerts": [
                Alert(service="analytics-dashboard", severity="P1", message="Data staleness alert", metric="data_lag_seconds", value=840.0, threshold=60.0),
                Alert(service="alert-engine", severity="P1", message="Alert evaluation failures", metric="error_rate", value=0.44, threshold=0.05),
                Alert(service="kafka-consumer", severity="P2", message="Consumer lag growing", metric="consumer_lag", value=48000.0, threshold=1000.0),
            ],
            "logs": [
                LogEntry(timestamp="T-12:00", level="INFO", service="stream-processor", message="Window state: 800MB — within normal range"),
                LogEntry(timestamp="T-06:00", level="WARN", service="stream-processor", message="Window state: 3.2GB — possible state accumulation"),
                LogEntry(timestamp="T+00:00", level="ERROR", service="stream-processor", message="OutOfMemoryError: window state too large — processor paused"),
                LogEntry(timestamp="T+00:01", level="WARN", service="kafka-consumer", message="stream-processor not consuming — lag growing"),
                LogEntry(timestamp="T+00:02", level="ERROR", service="alert-engine", message="Cannot fetch metrics: stream-processor offline"),
                # Red herrings
                LogEntry(timestamp="T-20:00", level="INFO", service="kafka-consumer", message="Topic rebalance: 3 partitions reassigned"),
                LogEntry(timestamp="T-08:00", level="INFO", service="stream-processor", message="Tumbling window size updated: 5min → 10min"),
            ],
            "metrics": [
                ServiceMetric(service="stream-processor", cpu_percent=0.0, memory_percent=0.0, error_rate=0.00, latency_p99_ms=0.0, requests_per_second=0.0),
                ServiceMetric(service="kafka-consumer", cpu_percent=22.0, memory_percent=30.0, error_rate=0.08, latency_p99_ms=480.0, requests_per_second=140.0),
                ServiceMetric(service="alert-engine", cpu_percent=35.0, memory_percent=42.0, error_rate=0.44, latency_p99_ms=2400.0, requests_per_second=80.0),
                ServiceMetric(service="analytics-dashboard", cpu_percent=28.0, memory_percent=36.0, error_rate=0.22, latency_p99_ms=1800.0, requests_per_second=60.0),
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
                "reveals":    "Window state grew continuously for 12 minutes. The window size change from 5min→10min doubled the state retention period but the heap was not resized. State accumulating at 200MB/minute. Restart clears in-memory state; permanent fix requires heap resize or window rollback.",
            },
        },
        # Variant 2: log-aggregator OOM
        {
            "root_cause_service": "log-aggregator",
            "root_cause_type":    "memory_leak",
            "correct_action":     "restart_service",
            "alerts": [
                Alert(service="monitoring-dashboard", severity="P1", message="Metrics missing for 12 services", metric="missing_services", value=12, threshold=12.0),
                Alert(service="alerting-service", severity="P1", message="Alert pipeline failures", metric="error_rate", value=0.52, threshold=0.05),
            ],
            "logs": [
                LogEntry(timestamp="T-10:00", level="INFO", service="log-aggregator", message="Ingest buffer: 1.2GB — normal"),
                LogEntry(timestamp="T-05:00", level="WARN", service="log-aggregator", message="Ingest buffer: 4.8GB — flushing slow"),
                LogEntry(timestamp="T+00:00", level="ERROR", service="log-aggregator", message="OutOfMemoryError: ingest buffer overflow — dropping logs"),
                LogEntry(timestamp="T+00:01", level="ERROR", service="alerting-service", message="Cannot read logs from aggregator — 503"),
                LogEntry(timestamp="T+00:02", level="ERROR", service="monitoring-dashboard", message="Log stream lost for 12 services"),
                # Red herrings
                LogEntry(timestamp="T-08:00", level="INFO", service="log-aggregator", message="New log source added: billing-service (high volume)"),
                LogEntry(timestamp="T-15:00", level="INFO", service="alerting-service", message="Alert rule updated: latency threshold 500ms→300ms"),
            ],
            "metrics": [
                ServiceMetric(service="log-aggregator", cpu_percent=0.0, memory_percent=0.0, error_rate=0.00, latency_p99_ms=0.0, requests_per_second=0.0),
                ServiceMetric(service="alerting-service", cpu_percent=38.0, memory_percent=44.0, error_rate=0.52, latency_p99_ms=2200.0, requests_per_second=90.0),
                ServiceMetric(service="monitoring-dashboard", cpu_percent=25.0, memory_percent=38.0, error_rate=0.30, latency_p99_ms=1400.0, requests_per_second=70.0),
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
