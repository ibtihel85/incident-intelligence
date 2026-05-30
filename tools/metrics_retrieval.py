"""
MCP Tool: metrics_retrieval
Queries time-series metric data — returns statistics, trend info, and peaks.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from tools.base import MCPTool, register_tool


@register_tool
class MetricsRetrievalTool(MCPTool):
    name = "metrics_retrieval"
    description = (
        "Query time-series metrics. Returns descriptive statistics (min, max, "
        "mean, p95, p99), trend direction, and anomalous windows."
    )
    input_schema = {
        "metrics": "Dict[str, List[Dict]] — metric_name → [{timestamp, value}]",
        "metric_name": "str — which metric to query",
        "window_start": "Optional[str] — ISO timestamp",
        "window_end": "Optional[str] — ISO timestamp",
    }

    def execute(
        self,
        metrics: Dict[str, List[Dict[str, Any]]],
        metric_name: str,
        window_start: Optional[str] = None,
        window_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        series = metrics.get(metric_name)
        if series is None:
            return {"error": f"Metric '{metric_name}' not found", "available": list(metrics.keys())}

        points = [p if isinstance(p, dict) else {"timestamp": p.timestamp, "value": p.value} for p in series]

        if window_start:
            points = [p for p in points if p["timestamp"] >= window_start]
        if window_end:
            points = [p for p in points if p["timestamp"] <= window_end]

        if not points:
            return {"metric": metric_name, "count": 0, "stats": {}}

        values = [float(p["value"]) for p in points]
        sorted_vals = sorted(values)
        n = len(sorted_vals)

        def percentile(data: List[float], pct: float) -> float:
            idx = int(pct / 100 * (len(data) - 1))
            return data[idx]

        # Simple linear trend
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n)) or 1
        slope = numerator / denominator

        trend = "stable"
        if slope > 0.1:
            trend = "increasing"
        elif slope < -0.1:
            trend = "decreasing"

        # Peak windows (values above 90th percentile)
        p90 = percentile(sorted_vals, 90)
        peak_points = [p for p in points if float(p["value"]) >= p90]

        return {
            "metric": metric_name,
            "count": n,
            "stats": {
                "min": round(min(values), 3),
                "max": round(max(values), 3),
                "mean": round(statistics.mean(values), 3),
                "median": round(statistics.median(values), 3),
                "stddev": round(statistics.stdev(values) if n > 1 else 0.0, 3),
                "p95": round(percentile(sorted_vals, 95), 3),
                "p99": round(percentile(sorted_vals, 99), 3),
            },
            "trend": trend,
            "slope_per_sample": round(slope, 4),
            "peak_count": len(peak_points),
            "peak_timestamps": [p["timestamp"] for p in peak_points[:5]],
        }
