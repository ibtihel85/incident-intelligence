"""
Agent 1 — Data Ingestion Agent
Loads raw incident data, validates schema, normalises metrics and logs
into the canonical NormalizedData format used by all downstream agents.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from utils.models import LogEntry, MetricPoint, NormalizedData


class DataIngestionAgent(BaseAgent):
    name = "data_ingestion_agent"

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raw = context["raw_data"]
        self._logger.info(f"Ingesting raw data for scenario: {raw.get('scenario', 'unknown')}")

        incident_id = raw.get("incident_id") or f"INC-{uuid.uuid4().hex[:8].upper()}"

        metrics = self._normalise_metrics(raw.get("metrics", {}))
        logs = self._normalise_logs(raw.get("logs", []))
        events = raw.get("events", [])
        services = list(
            set(e.service for e in logs)
            | set(raw.get("services", []))
        )

        normalized = NormalizedData(
            incident_id=incident_id,
            scenario=raw.get("scenario", "unknown"),
            start_time=raw.get("start_time", datetime.now(timezone.utc).isoformat()),
            end_time=raw.get("end_time", datetime.now(timezone.utc).isoformat()),
            services=services,
            metrics=metrics,
            logs=logs,
            events=events,
            metadata=raw.get("metadata", {}),
        )

        self._logger.info(
            f"Ingested: {len(logs)} log entries, "
            f"{sum(len(v) for v in metrics.values())} metric points, "
            f"{len(services)} services"
        )

        context["normalized_data"] = normalized
        context["incident_id"] = incident_id
        return context

    def _normalise_metrics(self, raw_metrics: Dict[str, Any]) -> Dict[str, List[MetricPoint]]:
        normalised: Dict[str, List[MetricPoint]] = {}
        for metric_name, series in raw_metrics.items():
            points: List[MetricPoint] = []
            for pt in series:
                if isinstance(pt, dict):
                    points.append(
                        MetricPoint(
                            timestamp=str(pt.get("timestamp", "")),
                            value=float(pt.get("value", 0.0)),
                            labels=pt.get("labels", {}),
                        )
                    )
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    points.append(MetricPoint(timestamp=str(pt[0]), value=float(pt[1])))
            normalised[metric_name] = points
        return normalised

    def _normalise_logs(self, raw_logs: List[Any]) -> List[LogEntry]:
        normalised: List[LogEntry] = []
        for entry in raw_logs:
            if isinstance(entry, dict):
                level = entry.get("level", "INFO").upper()
                if level == "WARNING":
                    level = "WARN"
                normalised.append(
                    LogEntry(
                        timestamp=str(entry.get("timestamp", "")),
                        level=level,
                        message=entry.get("message", ""),
                        service=entry.get("service", "unknown"),
                        host=entry.get("host", ""),
                        trace_id=entry.get("trace_id", ""),
                        extra={k: v for k, v in entry.items()
                               if k not in {"timestamp", "level", "message", "service", "host", "trace_id"}},
                    )
                )
        return normalised
