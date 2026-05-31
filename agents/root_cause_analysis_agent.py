"""
Agent 3 — Root Cause Analysis Agent
Builds a causal reasoning chain from detected anomalies and log patterns.
Uses a rule-based expert system with weighted evidence scoring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent
from utils.models import (
    Anomaly, AnomalyType, CausalStep, NormalizedData,
    RootCause, RootCauseCategory, Severity,
)


# ── Causal rule definitions ───────────────────────────────────

class _Rule:
    def __init__(self, name, condition_fn, category, title, description, weight):
        self.name = name
        self.condition_fn = condition_fn
        self.category = category
        self.title = title
        self.description = description
        self.weight = weight


def _has(anomalies: List[Anomaly], anomaly_type: AnomalyType) -> Optional[Anomaly]:
    for a in anomalies:
        if a.anomaly_type == anomaly_type:
            return a
    return None


def _critical_count(anomalies: List[Anomaly]) -> int:
    return sum(1 for a in anomalies if a.severity == Severity.CRITICAL)


RULES: List[_Rule] = [
    # ── CPU rules (checked before memory so CPU spike dominates) ──
    _Rule(
        name="cpu_spike_with_errors",
        condition_fn=lambda a, nd: (
            _has(a, AnomalyType.CPU_SPIKE) is not None and _critical_count(a) >= 2
        ),
        category=RootCauseCategory.RESOURCE_EXHAUSTION,
        title="CPU Exhaustion Causing Cascading Failures",
        description=(
            "CPU spike reaching critical levels is causing request queue buildup, "
            "timeout errors, and downstream failures. The system is unable to process "
            "incoming requests at the current rate. Possible causes: infinite loop, "
            "N+1 query pattern, unoptimised algorithm, or runaway background job."
        ),
        weight=0.93,
    ),
    _Rule(
        name="cpu_spike_only",
        condition_fn=lambda a, nd: _has(a, AnomalyType.CPU_SPIKE) is not None,
        category=RootCauseCategory.RESOURCE_EXHAUSTION,
        title="CPU Spike — Resource Contention",
        description=(
            "Sustained CPU spike detected. Likely caused by a compute-intensive "
            "operation (batch job, ML inference, cryptographic work) or a sudden traffic surge "
            "that has overwhelmed available CPU capacity."
        ),
        weight=0.75,
    ),
    # ── Memory leak rules (only if no CPU spike) ─────────────────
    _Rule(
        name="memory_leak_with_errors",
        condition_fn=lambda a, nd: (
            _has(a, AnomalyType.MEMORY_LEAK) is not None
            and _has(a, AnomalyType.CPU_SPIKE) is None
            and (
                _has(a, AnomalyType.LOG_ERROR_BURST) is not None
                or _has(a, AnomalyType.HIGH_ERROR_RATE) is not None
            )
            # Defer to DB rule if strong DB signals dominate (DB rule weight 0.95 > 0.92)
            and sum(
                1 for e in nd.logs if e.level in ("ERROR", "FATAL")
                and any(kw in e.message.lower() for kw in (
                    "connection refused", "pool exhausted", "max_connections",
                    "connection timeout", "psycopg2", "sqlalchemy",
                    "jdbc", "postgres", "mysql", "database connection",
                ))
            ) < 2
        ),
        category=RootCauseCategory.APPLICATION_BUG,
        title="Memory Leak in Application Code",
        description=(
            "A memory leak was detected alongside elevated error rates. "
            "The application is accumulating objects in memory without releasing them, "
            "likely due to unclosed connections, unbounded caches, or event listener accumulation. "
            "As memory exhaustion approaches, the application begins failing requests."
        ),
        weight=0.92,
    ),
    _Rule(
        name="memory_leak_only",
        condition_fn=lambda a, nd: (
            _has(a, AnomalyType.MEMORY_LEAK) is not None
            and _has(a, AnomalyType.CPU_SPIKE) is None
        ),
        category=RootCauseCategory.APPLICATION_BUG,
        title="Gradual Memory Leak Detected",
        description=(
            "Memory usage is growing monotonically over time without corresponding load increases. "
            "Root cause is likely a programming error: objects are allocated but not garbage-collected "
            "due to lingering references, unclosed file handles, or accumulating in-process caches."
        ),
        weight=0.85,
    ),
    # ── Database / dependency ─────────────────────────────────────
    _Rule(
        name="db_failure_pattern",
        condition_fn=lambda a, nd: (
            _has(a, AnomalyType.HIGH_ERROR_RATE) is not None
            and _has(a, AnomalyType.CPU_SPIKE) is None
            and sum(
                1 for e in nd.logs if e.level in ("ERROR", "FATAL")
                and any(kw in e.message.lower() for kw in (
                    "connection refused", "pool exhausted", "max_connections",
                    "connection timeout", "psycopg2", "sqlalchemy", "jdbc",
                    "postgres", "mysql", "database connection",
                ))
            ) >= 2
        ),
        category=RootCauseCategory.DEPENDENCY_FAILURE,
        title="Database Layer Failure",
        description=(
            "Error logs contain database-related keywords (connection refused, timeout, "
            "pool exhausted). The application is unable to communicate with its database backend, "
            "causing widespread request failures. Root cause is likely connection pool exhaustion, "
            "database overload, or network partition between app and DB."
        ),
        weight=0.95,
    ),
    # ── Latency ───────────────────────────────────────────────────
    _Rule(
        name="latency_with_cpu",
        condition_fn=lambda a, nd: (
            _has(a, AnomalyType.HIGH_LATENCY) is not None
            and _has(a, AnomalyType.CPU_SPIKE) is not None
        ),
        category=RootCauseCategory.RESOURCE_EXHAUSTION,
        title="Latency Degradation Due to CPU Saturation",
        description=(
            "High latency is co-occurring with CPU saturation. "
            "Requests are queuing in the application thread pool "
            "because CPU is fully occupied, causing response time degradation."
        ),
        weight=0.80,
    ),
    # ── Generic error ─────────────────────────────────────────────
    _Rule(
        name="high_error_rate_only",
        condition_fn=lambda a, nd: _has(a, AnomalyType.HIGH_ERROR_RATE) is not None,
        category=RootCauseCategory.APPLICATION_BUG,
        title="Application Error Rate Spike",
        description=(
            "High error rate detected without clear resource exhaustion. "
            "Likely a recent deployment introduced a regression, "
            "or an upstream dependency has become unavailable."
        ),
        weight=0.70,
    ),
    # ── Fallback ──────────────────────────────────────────────────
    _Rule(
        name="statistical_outlier_only",
        condition_fn=lambda a, nd: (
            _has(a, AnomalyType.STATISTICAL_OUTLIER) is not None and len(a) == 1
        ),
        category=RootCauseCategory.UNKNOWN,
        title="Unexplained Statistical Anomaly",
        description=(
            "A statistical outlier was detected in system metrics, "
            "but no clear causal pattern has been identified. "
            "Manual investigation is recommended."
        ),
        weight=0.40,
    ),
]


class RootCauseAnalysisAgent(BaseAgent):
    name = "root_cause_analysis_agent"

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        anomalies: List[Anomaly] = context.get("anomalies", [])
        nd: NormalizedData = context["normalized_data"]
        cfg = self.config.get("root_cause", {})
        min_confidence = cfg.get("min_confidence", 0.4)

        if not anomalies:
            self._logger.warning("No anomalies to analyse — skipping RCA")
            context["root_cause"] = None
            return context

        # Score all matching rules
        matches: List[Tuple[float, _Rule]] = []
        for rule in RULES:
            try:
                if rule.condition_fn(anomalies, nd):
                    matches.append((rule.weight, rule))
            except Exception:
                pass

        if not matches:
            context["root_cause"] = self._unknown_root_cause(anomalies)
            return context

        # Pick highest-confidence rule
        matches.sort(key=lambda x: x[0], reverse=True)
        confidence, best_rule = matches[0]

        if confidence < min_confidence:
            context["root_cause"] = self._unknown_root_cause(anomalies)
            return context

        causal_chain = self._build_causal_chain(best_rule, anomalies, nd)
        affected = list(set(a.service for a in anomalies))

        root_cause = RootCause(
            category=best_rule.category,
            title=best_rule.title,
            description=best_rule.description,
            confidence=round(confidence, 3),
            causal_chain=causal_chain,
            affected_services=affected,
            supporting_anomalies=[a.anomaly_id for a in anomalies],
        )

        self._logger.info(
            f"Root cause identified: [{best_rule.category.value}] {best_rule.title} "
            f"(confidence={confidence:.2f})"
        )
        context["root_cause"] = root_cause
        return context

    def _build_causal_chain(
        self, rule: _Rule, anomalies: List[Anomaly], nd: NormalizedData
    ) -> List[CausalStep]:
        chain: List[CausalStep] = []
        step = 1

        if anomalies:
            first = min(anomalies, key=lambda a: a.detected_at)
            chain.append(CausalStep(
                step=step,
                observation=f"First anomaly detected: {first.anomaly_type.value}",
                evidence=first.description,
                confidence=first.confidence,
            ))
            step += 1

        critical_anomalies = [a for a in anomalies if a.severity == Severity.CRITICAL]
        if critical_anomalies:
            chain.append(CausalStep(
                step=step,
                observation=f"{len(critical_anomalies)} critical severity signal(s) detected",
                evidence="; ".join(a.anomaly_type.value for a in critical_anomalies),
                confidence=0.90,
            ))
            step += 1

        error_logs = [l for l in nd.logs if l.level in ("ERROR", "FATAL")]
        if error_logs:
            sample = error_logs[:3]
            chain.append(CausalStep(
                step=step,
                observation=f"{len(error_logs)} ERROR/FATAL log entries corroborate anomalies",
                evidence="; ".join(l.message[:80] for l in sample),
                confidence=0.85,
            ))
            step += 1

        chain.append(CausalStep(
            step=step,
            observation=f"Pattern matched rule: '{rule.name}'",
            evidence=f"Category={rule.category.value}, weight={rule.weight}",
            confidence=rule.weight,
        ))

        return chain

    def _unknown_root_cause(self, anomalies: List[Anomaly]) -> RootCause:
        return RootCause(
            category=RootCauseCategory.UNKNOWN,
            title="Root Cause Undetermined",
            description="No causal pattern matched the observed anomalies with sufficient confidence. Manual investigation required.",
            confidence=0.10,
            causal_chain=[],
            affected_services=list(set(a.service for a in anomalies)),
            supporting_anomalies=[a.anomaly_id for a in anomalies],
        )
