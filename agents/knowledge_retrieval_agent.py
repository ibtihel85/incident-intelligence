"""
Agent 4 — Knowledge Retrieval Agent (MCP-based)
Searches the knowledge base for similar historical incidents
to enrich the context with proven resolutions.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from tools import KnowledgeSearchTool
from utils.models import Anomaly, NormalizedData, RootCause, SimilarIncident


class KnowledgeRetrievalAgent(BaseAgent):
    name = "knowledge_retrieval_agent"

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        nd: NormalizedData = context["normalized_data"]
        root_cause: RootCause | None = context.get("root_cause")
        anomalies: List[Anomaly] = context.get("anomalies", [])

        tool = KnowledgeSearchTool(self.config)

        query = self._build_query(nd, root_cause, anomalies)
        tags = self._extract_tags(nd, root_cause, anomalies)

        self._logger.info(f"Knowledge search query: '{query}' | tags: {tags}")

        result = tool.call(query=query, tags=tags, top_k=5)

        similar: List[SimilarIncident] = []
        for inc in result.get("incidents", []):
            similar.append(SimilarIncident(
                incident_id=inc["incident_id"],
                title=inc["title"],
                similarity_score=inc["similarity_score"],
                resolution=inc["resolution"],
                tags=inc["tags"],
            ))

        self._logger.info(f"Found {len(similar)} similar historical incidents")
        context["similar_incidents"] = similar
        return context

    def _build_query(
        self,
        nd: NormalizedData,
        rc: RootCause | None,
        anomalies: List[Anomaly],
    ) -> str:
        parts = [nd.scenario.replace("_", " ")]
        if rc:
            parts.append(rc.title)
            parts.append(rc.category.value.replace("_", " "))
        for a in anomalies[:3]:
            parts.append(a.anomaly_type.value.replace("_", " ").lower())
        return " ".join(parts)

    def _extract_tags(
        self,
        nd: NormalizedData,
        rc: RootCause | None,
        anomalies: List[Anomaly],
    ) -> List[str]:
        tags: List[str] = []
        for metric in nd.metrics:
            for kw in ("cpu", "memory", "disk", "network", "database", "latency", "error"):
                if kw in metric:
                    tags.append(kw)
        for a in anomalies:
            tags += a.anomaly_type.value.lower().split("_")
        if rc:
            tags += rc.category.value.lower().split("_")
        return list(set(tags))
