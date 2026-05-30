"""
MCP Tool: log_query
Filters and queries log entries by level, service, time window, and keyword.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from tools.base import MCPTool, register_tool
from utils.models import LogEntry


@register_tool
class LogQueryTool(MCPTool):
    name = "log_query"
    description = (
        "Query structured log entries. Filter by severity level, service name, "
        "time range, or keyword. Returns matching log lines with counts."
    )
    input_schema = {
        "logs": "List[LogEntry] — the log corpus to query",
        "level": "Optional[str] — minimum log level (DEBUG|INFO|WARN|ERROR|FATAL)",
        "service": "Optional[str] — filter to a specific service",
        "keyword": "Optional[str] — substring search in message",
        "limit": "int — maximum entries to return (default 100)",
    }

    _LEVEL_ORDER = {"DEBUG": 0, "INFO": 1, "WARN": 2, "WARNING": 2, "ERROR": 3, "FATAL": 4}

    def execute(
        self,
        logs: List[Dict[str, Any]],
        level: Optional[str] = None,
        service: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        min_level_rank = self._LEVEL_ORDER.get((level or "DEBUG").upper(), 0)

        for entry in logs:
            if isinstance(entry, LogEntry):
                entry = entry.__dict__

            entry_level = entry.get("level", "INFO").upper()
            entry_rank = self._LEVEL_ORDER.get(entry_level, 1)

            if entry_rank < min_level_rank:
                continue
            if service and entry.get("service", "") != service:
                continue
            if keyword and keyword.lower() not in entry.get("message", "").lower():
                continue

            results.append(entry)

        results = results[:limit]

        level_counts: Dict[str, int] = {}
        for e in results:
            lvl = e.get("level", "UNKNOWN").upper()
            level_counts[lvl] = level_counts.get(lvl, 0) + 1

        service_counts: Dict[str, int] = {}
        for e in results:
            svc = e.get("service", "unknown")
            service_counts[svc] = service_counts.get(svc, 0) + 1

        return {
            "total_matched": len(results),
            "level_distribution": level_counts,
            "service_distribution": service_counts,
            "entries": results,
        }
