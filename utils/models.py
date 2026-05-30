"""
Shared data models used across all agents and tools.
Keeps the system typed and self-documenting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Enumerations ─────────────────────────────────────────────

class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class AnomalyType(str, Enum):
    CPU_SPIKE = "CPU_SPIKE"
    MEMORY_LEAK = "MEMORY_LEAK"
    MEMORY_SPIKE = "MEMORY_SPIKE"
    HIGH_ERROR_RATE = "HIGH_ERROR_RATE"
    HIGH_LATENCY = "HIGH_LATENCY"
    SERVICE_DOWN = "SERVICE_DOWN"
    DISK_PRESSURE = "DISK_PRESSURE"
    NETWORK_ANOMALY = "NETWORK_ANOMALY"
    LOG_ERROR_BURST = "LOG_ERROR_BURST"
    STATISTICAL_OUTLIER = "STATISTICAL_OUTLIER"


class RootCauseCategory(str, Enum):
    RESOURCE_EXHAUSTION = "RESOURCE_EXHAUSTION"
    APPLICATION_BUG = "APPLICATION_BUG"
    INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"
    DEPENDENCY_FAILURE = "DEPENDENCY_FAILURE"
    MISCONFIGURATION = "MISCONFIGURATION"
    TRAFFIC_SPIKE = "TRAFFIC_SPIKE"
    DATA_CORRUPTION = "DATA_CORRUPTION"
    UNKNOWN = "UNKNOWN"


# ── Metric / Log Models ───────────────────────────────────────

@dataclass
class MetricPoint:
    timestamp: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class LogEntry:
    timestamp: str
    level: str          # DEBUG | INFO | WARN | ERROR | FATAL
    message: str
    service: str
    host: str = ""
    trace_id: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedData:
    incident_id: str
    scenario: str
    start_time: str
    end_time: str
    services: List[str]
    metrics: Dict[str, List[MetricPoint]]   # metric_name → time-series
    logs: List[LogEntry]
    events: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Anomaly Models ────────────────────────────────────────────

@dataclass
class Anomaly:
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: Severity
    metric: str
    service: str
    detected_at: str
    value: float
    threshold: float
    description: str
    supporting_evidence: List[str] = field(default_factory=list)
    confidence: float = 1.0


# ── Root Cause Models ─────────────────────────────────────────

@dataclass
class CausalStep:
    step: int
    observation: str
    evidence: str
    confidence: float


@dataclass
class RootCause:
    category: RootCauseCategory
    title: str
    description: str
    confidence: float
    causal_chain: List[CausalStep]
    affected_services: List[str]
    supporting_anomalies: List[str]   # anomaly_ids


# ── Knowledge / Fix Models ────────────────────────────────────

@dataclass
class SimilarIncident:
    incident_id: str
    title: str
    similarity_score: float
    resolution: str
    tags: List[str]


@dataclass
class FixRecommendation:
    priority: int        # 1 = highest
    action: str
    rationale: str
    estimated_impact: str
    commands: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


# ── Final Pipeline Result ─────────────────────────────────────

@dataclass
class IncidentResult:
    incident_id: str
    scenario: str
    severity: Severity
    anomalies: List[Anomaly]
    root_cause: Optional[RootCause]
    similar_incidents: List[SimilarIncident]
    recommendations: List[FixRecommendation]
    report: str
    pipeline_metadata: Dict[str, Any] = field(default_factory=dict)
