"""
Incident Orchestrator — runs the full multi-agent pipeline in sequence.

Pipeline stages:
  1. DataIngestionAgent        → normalize raw data
  2. AnomalyDetectionAgent     → detect anomalies
  3. RootCauseAnalysisAgent    → causal reasoning
  4. KnowledgeRetrievalAgent   → similar incident search (MCP)
  5. FixRecommendationAgent    → remediation actions
  6. ReportGeneratorAgent      → structured incident report
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from agents import (
    AnomalyDetectionAgent,
    DataIngestionAgent,
    FixRecommendationAgent,
    KnowledgeRetrievalAgent,
    ReportGeneratorAgent,
    RootCauseAnalysisAgent,
)
from utils.logger import get_logger
from utils.models import Anomaly, FixRecommendation, RootCause, SimilarIncident


logger = get_logger(__name__)


class IncidentOrchestrator:
    """
    Orchestrates the full incident intelligence pipeline.
    Each agent receives the shared context dict and enriches it.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._agents = [
            DataIngestionAgent(config),
            AnomalyDetectionAgent(config),
            RootCauseAnalysisAgent(config),
            KnowledgeRetrievalAgent(config),
            FixRecommendationAgent(config),
            ReportGeneratorAgent(config),
        ]

    def run(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all pipeline stages sequentially.
        Returns a result dict containing:
            - incident_id
            - severity
            - anomalies
            - root_cause
            - similar_incidents
            - recommendations
            - report (markdown string)
            - pipeline_metadata
        """
        pipeline_start = time.monotonic()
        logger.info("=" * 60)
        logger.info("INCIDENT INTELLIGENCE PIPELINE — START")
        logger.info("=" * 60)

        context: Dict[str, Any] = {"raw_data": raw_data}

        for agent in self._agents:
            try:
                context = agent.run(context)
            except Exception as exc:
                logger.error(f"Agent [{agent.name}] raised: {exc}", exc_info=True)
                # Continue pipeline with degraded output
                context.setdefault("_errors", []).append({
                    "agent": agent.name,
                    "error": str(exc),
                })

        elapsed = time.monotonic() - pipeline_start
        logger.info(f"Pipeline completed in {elapsed*1000:.1f}ms")

        return self._build_result(context, elapsed)

    def _build_result(self, context: Dict[str, Any], elapsed_s: float) -> Dict[str, Any]:
        from utils.models import Severity
        nd = context.get("normalized_data")

        anomalies: List[Anomaly] = context.get("anomalies", [])
        rc: RootCause | None = context.get("root_cause")
        similar: List[SimilarIncident] = context.get("similar_incidents", [])
        recs: List[FixRecommendation] = context.get("recommendations", [])
        severity = context.get("severity", Severity.UNKNOWN)

        agent_timings = {}
        for key, val in context.get("_agent_meta", {}).items():
            agent_timings[key] = val

        return {
            "incident_id": context.get("incident_id", "UNKNOWN"),
            "scenario": nd.scenario if nd else "unknown",
            "severity": severity.value if hasattr(severity, "value") else str(severity),
            "anomaly_count": len(anomalies),
            "anomalies": [a.__dict__ for a in anomalies],
            "root_cause": rc.__dict__ if rc else None,
            "similar_incidents": [s.__dict__ for s in similar],
            "recommendations": [r.__dict__ for r in recs],
            "report": context.get("report", "_Report generation failed._"),
            "pipeline_metadata": {
                "total_elapsed_ms": round(elapsed_s * 1000, 2),
                "agent_count": len(self._agents),
                "errors": context.get("_errors", []),
            },
        }
