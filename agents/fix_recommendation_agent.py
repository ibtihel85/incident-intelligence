"""
Agent 5 — Fix Recommendation Agent
Generates prioritised remediation actions based on root cause,
anomaly types, similar incident resolutions, and system state.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from utils.models import (
    Anomaly, AnomalyType, FixRecommendation, NormalizedData,
    RootCause, RootCauseCategory, SimilarIncident,
)


# ── Recommendation rule table ─────────────────────────────────
# Maps (RootCauseCategory, AnomalyType | None) → list of fix templates

_FIX_RULES: Dict[RootCauseCategory, List[Dict]] = {
    RootCauseCategory.APPLICATION_BUG: [
        {
            "priority": 1,
            "action": "Restart affected service to restore availability",
            "rationale": "Immediate mitigation to restore service while root cause is fixed.",
            "estimated_impact": "Restores service availability within 30–60 seconds",
            "commands": [
                "kubectl rollout restart deployment/<service-name>",
                "systemctl restart <service-name>",
            ],
        },
        {
            "priority": 2,
            "action": "Review recent deployments and consider rollback",
            "rationale": "New code is the most common cause of sudden error-rate spikes.",
            "estimated_impact": "Eliminates deployment-induced regressions",
            "commands": [
                "kubectl rollout history deployment/<service-name>",
                "kubectl rollout undo deployment/<service-name>",
                "git log --oneline -20",
            ],
        },
        {
            "priority": 3,
            "action": "Enable heap profiling to capture memory allocation patterns",
            "rationale": "Required to pinpoint exact allocation sites for leak fixes.",
            "estimated_impact": "Identifies memory leak source within one analysis cycle",
            "commands": [
                "py-spy record -o profile.svg --pid <pid>",
                "valgrind --leak-check=full ./<binary>",
            ],
            "references": ["https://py-spy.readthedocs.io"],
        },
        {
            "priority": 4,
            "action": "Add connection pool limits and implement connection context managers",
            "rationale": "Prevents resource exhaustion from unclosed connections.",
            "estimated_impact": "Prevents memory growth from connection accumulation",
            "commands": [],
        },
    ],
    RootCauseCategory.RESOURCE_EXHAUSTION: [
        {
            "priority": 1,
            "action": "Identify and terminate the top CPU-consuming process",
            "rationale": "Immediate relief from CPU saturation.",
            "estimated_impact": "CPU normalises within seconds of process termination",
            "commands": [
                "top -b -n1 | head -20",
                "ps aux --sort=-%cpu | head -10",
                "kill -9 <pid>   # only if safe to terminate",
            ],
        },
        {
            "priority": 2,
            "action": "Scale horizontally — add replicas to distribute load",
            "rationale": "Increases processing capacity without downtime.",
            "estimated_impact": "Reduces per-pod CPU by 1/N where N = new replica count",
            "commands": [
                "kubectl scale deployment/<service-name> --replicas=<N>",
                "aws autoscaling set-desired-capacity --auto-scaling-group-name <asg> --desired-capacity <N>",
            ],
        },
        {
            "priority": 3,
            "action": "Set CPU and memory resource limits on pods / containers",
            "rationale": "Prevents single service from monopolising node resources.",
            "estimated_impact": "Isolates resource usage, improves overall system stability",
            "commands": [
                "kubectl set resources deployment <name> --requests=cpu=200m,memory=256Mi --limits=cpu=1,memory=1Gi",
            ],
        },
        {
            "priority": 4,
            "action": "Schedule batch or background jobs during off-peak hours",
            "rationale": "Prevents batch workloads from competing with user-facing traffic.",
            "estimated_impact": "Eliminates CPU spikes during peak hours",
            "commands": [
                "crontab -e   # reschedule to 02:00 UTC",
            ],
        },
    ],
    RootCauseCategory.DEPENDENCY_FAILURE: [
        {
            "priority": 1,
            "action": "Check database health and connection pool status",
            "rationale": "Database failure is the most critical dependency to verify first.",
            "estimated_impact": "Identifies whether DB is reachable and accepting connections",
            "commands": [
                "psql -h <host> -U <user> -c 'SELECT 1'",
                "SELECT count(*), state FROM pg_stat_activity GROUP BY state;",
                "SHOW max_connections;",
            ],
        },
        {
            "priority": 2,
            "action": "Terminate idle/blocked database connections",
            "rationale": "Clears connection pool to restore new connection capacity.",
            "estimated_impact": "Immediately frees connections blocked by locks or long queries",
            "commands": [
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND query_start < now() - interval '5 minutes';",
            ],
        },
        {
            "priority": 3,
            "action": "Deploy connection pooler (PgBouncer) in front of database",
            "rationale": "Multiplexes many app connections over fewer DB connections.",
            "estimated_impact": "Supports 10–100x more application connections",
            "commands": [],
            "references": ["https://www.pgbouncer.org/config.html"],
        },
        {
            "priority": 4,
            "action": "Implement circuit-breaker pattern for downstream service calls",
            "rationale": "Prevents cascading failures when dependencies are degraded.",
            "estimated_impact": "Reduces blast radius of downstream outages",
            "commands": [],
        },
    ],
    RootCauseCategory.TRAFFIC_SPIKE: [
        {
            "priority": 1,
            "action": "Enable request rate limiting on API gateway",
            "rationale": "Immediately protects backend from being overwhelmed.",
            "estimated_impact": "Caps incoming traffic, allows existing requests to complete",
            "commands": [
                "# Nginx: limit_req_zone $binary_remote_addr zone=one:10m rate=10r/s;",
                "# AWS API Gateway: Enable usage plans with throttling",
            ],
        },
        {
            "priority": 2,
            "action": "Trigger auto-scaling group or HPA scale-up",
            "rationale": "Adds capacity to handle traffic surge.",
            "estimated_impact": "Scales service to match demand within minutes",
            "commands": [
                "kubectl autoscale deployment <name> --min=3 --max=20 --cpu-percent=70",
            ],
        },
    ],
    RootCauseCategory.UNKNOWN: [
        {
            "priority": 1,
            "action": "Collect full diagnostic snapshot for manual investigation",
            "rationale": "Preserves system state for post-incident analysis.",
            "estimated_impact": "Enables root cause identification by on-call engineer",
            "commands": [
                "kubectl describe nodes",
                "kubectl get events --sort-by='.metadata.creationTimestamp'",
                "dmesg | tail -50",
            ],
        },
    ],
}


class FixRecommendationAgent(BaseAgent):
    name = "fix_recommendation_agent"

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        root_cause: RootCause | None = context.get("root_cause")
        anomalies: List[Anomaly] = context.get("anomalies", [])
        similar: List[SimilarIncident] = context.get("similar_incidents", [])

        recs: List[FixRecommendation] = []

        # Primary recommendations from root cause category
        category = root_cause.category if root_cause else RootCauseCategory.UNKNOWN
        for fix in _FIX_RULES.get(category, _FIX_RULES[RootCauseCategory.UNKNOWN]):
            recs.append(FixRecommendation(**fix))

        # Supplement with knowledge-base resolutions
        for inc in similar[:2]:
            if inc.resolution and inc.resolution not in [r.action for r in recs]:
                recs.append(FixRecommendation(
                    priority=len(recs) + 1,
                    action=f"Apply resolution from similar incident {inc.incident_id}: {inc.title}",
                    rationale=f"Similar incident resolved with: {inc.resolution[:100]}",
                    estimated_impact="Proven resolution from historical incident",
                    commands=[],
                    references=[f"Internal incident: {inc.incident_id}"],
                ))

        self._logger.info(f"Generated {len(recs)} fix recommendations")
        context["recommendations"] = recs
        return context
