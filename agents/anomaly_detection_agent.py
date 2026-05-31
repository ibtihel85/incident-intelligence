"""
Agent 2 — Anomaly Detection Agent
Applies statistical and rule-based detection across all metrics and logs.
Detects: spikes, sustained high values, monotonic growth (leaks), error bursts,
and general statistical outliers (Z-score / IQR).
"""

from __future__ import annotations

import math
import statistics
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from tools import MetricsRetrievalTool, LogQueryTool
from utils.models import Anomaly, AnomalyType, MetricPoint, NormalizedData, Severity


class AnomalyDetectionAgent(BaseAgent):
    name = "anomaly_detection_agent"

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        nd: NormalizedData = context["normalized_data"]
        cfg = self.config.get("anomaly_detection", {})

        anomalies: List[Anomaly] = []

        # MCP tool instances
        metrics_tool = MetricsRetrievalTool(self.config)
        log_tool = LogQueryTool(self.config)

        # ── Metric-based detection ────────────────────────────
        anomalies += self._detect_cpu(nd, cfg.get("cpu", {}), metrics_tool)
        anomalies += self._detect_memory(nd, cfg.get("memory", {}), metrics_tool)
        anomalies += self._detect_error_rate(nd, cfg.get("error_rate", {}), metrics_tool)
        anomalies += self._detect_latency(nd, cfg.get("latency", {}), metrics_tool)
        anomalies += self._detect_statistical_outliers(nd, cfg.get("statistical", {}))

        # ── Log-based detection ───────────────────────────────
        anomalies += self._detect_log_error_burst(nd, log_tool)

        self._logger.info(f"Detected {len(anomalies)} anomalies")
        context["anomalies"] = anomalies
        return context

    # ── CPU ───────────────────────────────────────────────────

    def _detect_cpu(
        self, nd: NormalizedData, cfg: Dict[str, Any], tool: MetricsRetrievalTool
    ) -> List[Anomaly]:
        anomalies = []
        for metric_name in ("cpu_usage", "cpu_percent", "cpu"):
            if metric_name not in nd.metrics:
                continue
            stats = tool.call(metrics=self._metrics_as_dicts(nd), metric_name=metric_name)
            if "error" in stats:
                continue

            warn_t = cfg.get("warning_threshold", 80.0)
            crit_t = cfg.get("critical_threshold", 95.0)
            peak = stats["stats"].get("max", 0)
            p95 = stats["stats"].get("p95", 0)

            if peak >= crit_t:
                anomalies.append(self._make_anomaly(
                    AnomalyType.CPU_SPIKE, Severity.CRITICAL,
                    metric_name, nd.services[0] if nd.services else "unknown",
                    nd.end_time, peak, crit_t,
                    f"CPU peaked at {peak:.1f}% (critical threshold: {crit_t}%)",
                    [f"p95={p95:.1f}%", f"trend={stats.get('trend')}"],
                    confidence=0.95,
                ))
            elif peak >= warn_t or p95 >= warn_t:
                anomalies.append(self._make_anomaly(
                    AnomalyType.CPU_SPIKE, Severity.HIGH,
                    metric_name, nd.services[0] if nd.services else "unknown",
                    nd.end_time, peak, warn_t,
                    f"CPU elevated at {peak:.1f}% (warning threshold: {warn_t}%)",
                    [f"p95={p95:.1f}%"],
                    confidence=0.80,
                ))
        return anomalies

    # ── Memory ────────────────────────────────────────────────

    def _detect_memory(
        self, nd: NormalizedData, cfg: Dict[str, Any], tool: MetricsRetrievalTool
    ) -> List[Anomaly]:
        anomalies = []
        for metric_name in ("memory_usage", "memory_percent", "memory"):
            if metric_name not in nd.metrics:
                continue
            stats = tool.call(metrics=self._metrics_as_dicts(nd), metric_name=metric_name)
            if "error" in stats:
                continue

            warn_t = cfg.get("warning_threshold", 75.0)
            crit_t = cfg.get("critical_threshold", 90.0)
            leak_slope = cfg.get("leak_slope_threshold", 0.5)
            peak = stats["stats"].get("max", 0)
            slope = stats.get("slope_per_sample", 0)

            # Monotonic growth → memory leak
            if slope >= leak_slope and stats["stats"].get("mean", 0) > 50:
                anomalies.append(self._make_anomaly(
                    AnomalyType.MEMORY_LEAK, Severity.HIGH,
                    metric_name, nd.services[0] if nd.services else "unknown",
                    nd.end_time, slope, leak_slope,
                    f"Memory growing monotonically at +{slope:.2f}% per sample — likely leak",
                    [f"mean={stats['stats'].get('mean'):.1f}%", f"max={peak:.1f}%"],
                    confidence=0.85,
                ))

            if peak >= crit_t:
                anomalies.append(self._make_anomaly(
                    AnomalyType.MEMORY_SPIKE, Severity.CRITICAL,
                    metric_name, nd.services[0] if nd.services else "unknown",
                    nd.end_time, peak, crit_t,
                    f"Memory at {peak:.1f}% — approaching OOM",
                    [f"slope={slope:.3f}/sample"],
                    confidence=0.90,
                ))
            elif peak >= warn_t:
                anomalies.append(self._make_anomaly(
                    AnomalyType.MEMORY_SPIKE, Severity.MEDIUM,
                    metric_name, nd.services[0] if nd.services else "unknown",
                    nd.end_time, peak, warn_t,
                    f"Memory elevated at {peak:.1f}%",
                    [],
                    confidence=0.70,
                ))
        return anomalies

    # ── Error rate ────────────────────────────────────────────

    def _detect_error_rate(
        self, nd: NormalizedData, cfg: Dict[str, Any], tool: MetricsRetrievalTool
    ) -> List[Anomaly]:
        anomalies = []
        for metric_name in ("error_rate", "errors_per_minute", "http_errors"):
            if metric_name not in nd.metrics:
                continue
            stats = tool.call(metrics=self._metrics_as_dicts(nd), metric_name=metric_name)
            if "error" in stats:
                continue

            warn_t = cfg.get("warning_threshold", 5.0)
            crit_t = cfg.get("critical_threshold", 20.0)
            peak = stats["stats"].get("max", 0)

            if peak >= crit_t:
                sev = Severity.CRITICAL
            elif peak >= warn_t:
                sev = Severity.HIGH
            else:
                continue

            anomalies.append(self._make_anomaly(
                AnomalyType.HIGH_ERROR_RATE, sev,
                metric_name, nd.services[0] if nd.services else "unknown",
                nd.end_time, peak, warn_t,
                f"Error rate spiked to {peak:.1f} errors/min",
                [f"mean={stats['stats'].get('mean'):.2f}"],
                confidence=0.90,
            ))
        return anomalies

    # ── Latency ───────────────────────────────────────────────

    def _detect_latency(
        self, nd: NormalizedData, cfg: Dict[str, Any], tool: MetricsRetrievalTool
    ) -> List[Anomaly]:
        anomalies = []
        for metric_name in ("latency_ms", "response_time_ms", "p99_latency"):
            if metric_name not in nd.metrics:
                continue
            stats = tool.call(metrics=self._metrics_as_dicts(nd), metric_name=metric_name)
            if "error" in stats:
                continue

            warn_t = cfg.get("warning_threshold_ms", 500)
            crit_t = cfg.get("critical_threshold_ms", 2000)
            p99 = stats["stats"].get("p99", 0)

            if p99 >= crit_t:
                sev = Severity.CRITICAL
            elif p99 >= warn_t:
                sev = Severity.HIGH
            else:
                continue

            anomalies.append(self._make_anomaly(
                AnomalyType.HIGH_LATENCY, sev,
                metric_name, nd.services[0] if nd.services else "unknown",
                nd.end_time, p99, warn_t,
                f"p99 latency at {p99:.0f}ms (threshold: {warn_t}ms)",
                [f"mean={stats['stats'].get('mean'):.0f}ms"],
                confidence=0.85,
            ))
        return anomalies

    # ── Statistical outliers (Z-score + IQR) ─────────────────

    def _detect_statistical_outliers(
        self, nd: NormalizedData, cfg: Dict[str, Any]
    ) -> List[Anomaly]:
        anomalies: List[Anomaly] = []
        zscore_t = cfg.get("zscore_threshold", 3.0)

        for metric_name, series in nd.metrics.items():
            # Skip metrics already handled above
            if any(kw in metric_name for kw in ("cpu", "memory", "error", "latency", "response")):
                continue
            values = [p.value for p in series]
            if len(values) < 6:
                continue
            mean = statistics.mean(values)
            stddev = statistics.stdev(values)
            if stddev == 0:
                continue
            peak = max(values, key=abs)
            zscore = abs((peak - mean) / stddev)
            if zscore >= zscore_t:
                anomalies.append(self._make_anomaly(
                    AnomalyType.STATISTICAL_OUTLIER, Severity.MEDIUM,
                    metric_name, nd.services[0] if nd.services else "unknown",
                    nd.end_time, peak, mean + zscore_t * stddev,
                    f"Statistical outlier in {metric_name}: z-score={zscore:.2f}",
                    [f"mean={mean:.2f}", f"stddev={stddev:.2f}"],
                    confidence=round(min(0.95, zscore / (zscore_t * 2)), 2),
                ))
        return anomalies

    # ── Log error burst ───────────────────────────────────────

    def _detect_log_error_burst(
        self, nd: NormalizedData, tool: LogQueryTool
    ) -> List[Anomaly]:
        anomalies: List[Anomaly] = []
        logs_as_dicts = [l.__dict__ for l in nd.logs]
        result = tool.call(logs=logs_as_dicts, level="ERROR")
        error_count = result.get("total_matched", 0)
        total = len(nd.logs)

        if total == 0:
            return []

        error_ratio = error_count / total
        if error_ratio >= 0.3:
            sev = Severity.CRITICAL
        elif error_ratio >= 0.1:
            sev = Severity.HIGH
        else:
            return []

        anomalies.append(self._make_anomaly(
            AnomalyType.LOG_ERROR_BURST, sev,
            "log_error_rate", nd.services[0] if nd.services else "unknown",
            nd.end_time, error_count, total * 0.1,
            f"Log error burst: {error_count}/{total} entries are ERROR level ({error_ratio*100:.0f}%)",
            [f"service_breakdown={result.get('service_distribution')}"],
            confidence=0.88,
        ))
        return anomalies

    # ── Helpers ───────────────────────────────────────────────

    def _make_anomaly(
        self,
        anomaly_type: AnomalyType,
        severity: Severity,
        metric: str,
        service: str,
        detected_at: str,
        value: float,
        threshold: float,
        description: str,
        evidence: List[str],
        confidence: float = 1.0,
    ) -> Anomaly:
        return Anomaly(
            anomaly_id=f"ANOM-{uuid.uuid4().hex[:6].upper()}",
            anomaly_type=anomaly_type,
            severity=severity,
            metric=metric,
            service=service,
            detected_at=detected_at,
            value=round(value, 3),
            threshold=round(threshold, 3),
            description=description,
            supporting_evidence=evidence,
            confidence=confidence,
        )

    def _metrics_as_dicts(self, nd: NormalizedData) -> Dict[str, List[Dict]]:
        return {
            name: [{"timestamp": p.timestamp, "value": p.value} for p in series]
            for name, series in nd.metrics.items()
        }
