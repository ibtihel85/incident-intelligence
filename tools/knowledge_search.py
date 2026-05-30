"""
MCP Tool: knowledge_search
TF-IDF based similarity search over historical incident knowledge base.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.base import MCPTool, register_tool
from utils.models import SimilarIncident


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _tfidf_score(query_tokens: List[str], doc_tokens: List[str], idf: Dict[str, float]) -> float:
    tf = Counter(doc_tokens)
    total = len(doc_tokens) or 1
    score = 0.0
    for t in query_tokens:
        score += (tf.get(t, 0) / total) * idf.get(t, 0)
    return score


@register_tool
class KnowledgeSearchTool(MCPTool):
    name = "knowledge_search"
    description = (
        "Search the historical incident knowledge base using TF-IDF similarity. "
        "Returns ranked similar incidents with resolutions and runbooks."
    )
    input_schema = {
        "query": "str — natural language description of the current incident",
        "tags": "Optional[List[str]] — filter by incident tags",
        "top_k": "int — number of results (default 5)",
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._kb: List[Dict[str, Any]] = []
        self._idf: Dict[str, float] = {}
        self._loaded = False

    def _load_kb(self) -> None:
        kb_path = Path(self.config.get("knowledge_base", {}).get("path", "data/knowledge_base.json"))
        if kb_path.exists():
            with open(kb_path) as f:
                self._kb = json.load(f)
        else:
            self._kb = _DEFAULT_KB
        self._build_idf()
        self._loaded = True
        self._logger.info(f"Knowledge base loaded: {len(self._kb)} incidents")

    def _build_idf(self) -> None:
        N = len(self._kb) or 1
        df: Counter = Counter()
        for doc in self._kb:
            text = f"{doc.get('title','')} {doc.get('description','')} {' '.join(doc.get('tags', []))}"
            for tok in set(_tokenize(text)):
                df[tok] += 1
        self._idf = {t: math.log(N / (cnt + 1)) + 1 for t, cnt in df.items()}

    def execute(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        if not self._loaded:
            self._load_kb()

        query_tokens = _tokenize(query)
        threshold = self.config.get("knowledge_base", {}).get("similarity_threshold", 0.0)

        candidates = self._kb
        if tags:
            tag_set = set(t.lower() for t in tags)
            candidates = [d for d in candidates if tag_set.intersection(t.lower() for t in d.get("tags", []))]

        scored = []
        for doc in candidates:
            text = f"{doc.get('title','')} {doc.get('description','')} {' '.join(doc.get('tags', []))}"
            doc_tokens = _tokenize(text)
            score = _tfidf_score(query_tokens, doc_tokens, self._idf)
            if score >= threshold:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        results = []
        for score, doc in top:
            results.append(
                SimilarIncident(
                    incident_id=doc.get("id", ""),
                    title=doc.get("title", ""),
                    similarity_score=round(score, 4),
                    resolution=doc.get("resolution", ""),
                    tags=doc.get("tags", []),
                ).__dict__
            )

        return {"query": query, "results_found": len(results), "incidents": results}


# ── Embedded fallback knowledge base ─────────────────────────
_DEFAULT_KB = [
    {
        "id": "INC-001",
        "title": "CPU spike due to runaway ML training job",
        "description": "A background ML training process consumed 100% CPU for 20 minutes causing service degradation",
        "tags": ["cpu", "spike", "ml", "training", "resource"],
        "resolution": "Kill the runaway process with `kill -9 <pid>`. Add CPU limits to training jobs via cgroups. Schedule training in off-peak windows.",
    },
    {
        "id": "INC-002",
        "title": "Memory leak in Python web service",
        "description": "Python Flask application accumulated memory over 6 hours due to unclosed database connections and cached objects never evicted",
        "tags": ["memory", "leak", "python", "flask", "database", "connection"],
        "resolution": "Restart the service immediately. Apply connection pool limits (max_overflow=10). Add LRU cache eviction. Deploy fix with context manager for DB connections.",
    },
    {
        "id": "INC-003",
        "title": "PostgreSQL connection pool exhaustion",
        "description": "All 100 database connection slots exhausted causing application errors and timeouts for all users",
        "tags": ["database", "postgres", "connection", "pool", "exhaustion", "timeout"],
        "resolution": "Run `SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE wait_event_type = 'Lock';`. Increase max_connections or add PgBouncer. Review long-running queries.",
    },
    {
        "id": "INC-004",
        "title": "Traffic spike causing API gateway overload",
        "description": "Sudden 10x traffic spike from marketing campaign overwhelmed API gateway with rate limiting disabled",
        "tags": ["traffic", "spike", "api", "gateway", "rate-limit", "scaling"],
        "resolution": "Enable rate limiting on API gateway. Trigger horizontal pod autoscaler. Add CDN caching for static responses. Coordinate with marketing for pre-scaling.",
    },
    {
        "id": "INC-005",
        "title": "Disk I/O saturation from log rotation failure",
        "description": "Log rotation cron job failed silently; logs filled disk causing I/O saturation and application write failures",
        "tags": ["disk", "io", "logs", "rotation", "cron", "storage"],
        "resolution": "Manually rotate logs: `logrotate -f /etc/logrotate.conf`. Clean up: `find /var/log -name '*.log' -mtime +7 -delete`. Fix cron job and add disk usage alerting.",
    },
    {
        "id": "INC-006",
        "title": "Redis cache stampede on cold start",
        "description": "After Redis restart all cache keys expired simultaneously, causing thundering herd on database",
        "tags": ["redis", "cache", "stampede", "database", "thundering-herd"],
        "resolution": "Implement cache warming script. Add jitter to TTL values. Use mutex locking pattern for cache-miss database calls. Pre-populate cache before routing traffic.",
    },
    {
        "id": "INC-007",
        "title": "Kubernetes OOMKilled pods due to misconfigured memory limits",
        "description": "Memory limits set too low for pods caused repeated OOMKill events under normal load",
        "tags": ["kubernetes", "oom", "memory", "limits", "pods", "k8s"],
        "resolution": "Increase memory limits: `kubectl set resources deployment <name> --limits=memory=2Gi`. Analyse with `kubectl top pods`. Set VPA for automatic right-sizing.",
    },
]
