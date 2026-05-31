"""
Agent 6 — Report Generator Agent
Produces a structured, human-readable Markdown incident report
from all upstream agent outputs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from utils.models import (
    Anomaly, FixRecommendation, NormalizedData,
    RootCause, Severity, SimilarIncident,
)


_SEVERITY_EMOJI = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH: "🟠",
    Severity.MEDIUM: "🟡",
    Severity.LOW: "🟢",
    Severity.UNKNOWN: "⚪",
}


class ReportGeneratorAgent(BaseAgent):
    name = "report_generator_agent"

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        nd: NormalizedData = context["normalized_data"]
        anomalies: List[Anomaly] = context.get("anomalies", [])
        root_cause: RootCause | None = context.get("root_cause")
        similar: List[SimilarIncident] = context.get("similar_incidents", [])
        recs: List[FixRecommendation] = context.get("recommendations", [])

        overall_severity = self._overall_severity(anomalies)
        report = self._render(nd, anomalies, root_cause, similar, recs, overall_severity)

        context["report"] = report
        context["severity"] = overall_severity
        self._logger.info(f"Incident report generated ({len(report)} chars)")
        return context

    # ── Severity roll-up ──────────────────────────────────────

    def _overall_severity(self, anomalies: List[Anomaly]) -> Severity:
        if not anomalies:
            return Severity.UNKNOWN
        order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]
        for s in order:
            if any(a.severity == s for a in anomalies):
                return s
        return Severity.UNKNOWN

    # ── Markdown rendering ────────────────────────────────────

    def _render(
        self,
        nd: NormalizedData,
        anomalies: List[Anomaly],
        rc: RootCause | None,
        similar: List[SimilarIncident],
        recs: List[FixRecommendation],
        severity: Severity,
    ) -> str:
        sev_emoji = _SEVERITY_EMOJI.get(severity, "⚪")
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines: List[str] = []

        # ── Header ────────────────────────────────────────────
        lines += [
            f"# {sev_emoji} Incident Report — {nd.incident_id}",
            "",
            f"> **Generated:** {generated_at}  ",
            f"> **Scenario:** `{nd.scenario}`  ",
            f"> **Severity:** `{severity.value}`  ",
            f"> **Services:** {', '.join(f'`{s}`' for s in nd.services) or 'N/A'}  ",
            f"> **Time Window:** `{nd.start_time}` → `{nd.end_time}`",
            "",
            "---",
            "",
        ]

        # ── Executive Summary ─────────────────────────────────
        lines += ["## 📋 Executive Summary", ""]
        if rc:
            lines.append(
                f"**{rc.title}** was identified as the primary root cause "
                f"(confidence: `{rc.confidence*100:.0f}%`, category: `{rc.category.value}`). "
                f"{rc.description}"
            )
        else:
            lines.append(
                "No definitive root cause was identified. "
                "Manual investigation is required."
            )
        lines += ["", "---", ""]

        # ── Anomaly Summary ───────────────────────────────────
        lines += [
            "## 🔍 Detected Anomalies",
            "",
            f"**{len(anomalies)} anomal{'y' if len(anomalies) == 1 else 'ies'} detected**",
            "",
            "| ID | Type | Severity | Metric | Value | Threshold | Confidence |",
            "|---|---|---|---|---|---|---|",
        ]
        for a in sorted(anomalies, key=lambda x: x.severity.value):
            emoji = _SEVERITY_EMOJI.get(a.severity, "")
            lines.append(
                f"| `{a.anomaly_id}` | `{a.anomaly_type.value}` | {emoji} {a.severity.value} "
                f"| `{a.metric}` | `{a.value}` | `{a.threshold}` | `{a.confidence*100:.0f}%` |"
            )
        lines += [""]

        # Anomaly details
        for a in anomalies:
            lines += [
                f"### `{a.anomaly_id}` — {a.anomaly_type.value}",
                "",
                f"- **Description:** {a.description}",
                f"- **Detected at:** `{a.detected_at}`",
                f"- **Service:** `{a.service}`",
            ]
            if a.supporting_evidence:
                lines.append(f"- **Evidence:** {', '.join(str(e) for e in a.supporting_evidence)}")
            lines.append("")

        lines += ["---", ""]

        # ── Root Cause Analysis ───────────────────────────────
        lines += ["## 🧠 Root Cause Analysis", ""]
        if rc:
            lines += [
                f"**Category:** `{rc.category.value}`  ",
                f"**Confidence:** `{rc.confidence*100:.0f}%`  ",
                f"**Affected Services:** {', '.join(f'`{s}`' for s in rc.affected_services)}",
                "",
                f"> {rc.description}",
                "",
                "### Causal Chain",
                "",
            ]
            for step in rc.causal_chain:
                lines += [
                    f"**Step {step.step}** _(confidence: {step.confidence*100:.0f}%)_",
                    f"- 🔎 **Observation:** {step.observation}",
                    f"- 📎 **Evidence:** {step.evidence}",
                    "",
                ]
        else:
            lines += [
                "Root cause could not be determined automatically.",
                "",
                "**Recommended actions:**",
                "- Review recent deployments (git log, CI/CD history)",
                "- Inspect infrastructure changes (Terraform/Ansible diffs)",
                "- Check upstream dependency status pages",
                "",
            ]

        lines += ["---", ""]

        # ── Similar Incidents ─────────────────────────────────
        lines += ["## 📚 Similar Historical Incidents", ""]
        if similar:
            for inc in similar:
                score_pct = f"{inc.similarity_score * 100:.0f}%" if inc.similarity_score < 10 else f"{inc.similarity_score:.2f}"
                lines += [
                    f"### [{inc.incident_id}] {inc.title}",
                    f"**Similarity:** `{score_pct}` | **Tags:** {', '.join(f'`{t}`' for t in inc.tags)}",
                    "",
                    f"**Resolution:** {inc.resolution}",
                    "",
                ]
        else:
            lines += ["_No similar historical incidents found._", ""]

        lines += ["---", ""]

        # ── Recommendations ───────────────────────────────────
        lines += ["## 🛠️ Fix Recommendations", ""]
        if recs:
            for rec in sorted(recs, key=lambda r: r.priority):
                lines += [
                    f"### Priority {rec.priority} — {rec.action}",
                    "",
                    f"**Rationale:** {rec.rationale}  ",
                    f"**Estimated Impact:** {rec.estimated_impact}",
                    "",
                ]
                if rec.commands:
                    lines += ["```bash"]
                    lines += rec.commands
                    lines += ["```", ""]
                if rec.references:
                    lines.append(f"**References:** {', '.join(rec.references)}")
                    lines.append("")
        else:
            lines += ["_No automated recommendations available._", ""]

        lines += ["---", ""]

        # ── Metrics Snapshot ──────────────────────────────────
        if self.config.get("reporting", {}).get("include_raw_metrics", True):
            lines += ["## 📊 Metrics Snapshot", ""]
            lines += [
                "| Metric | Anomalous |",
                "|---|---|",
            ]
            anomalous_metrics = {a.metric for a in anomalies}
            for name in sorted(set(a.metric for a in anomalies)):
                flag = "✅ Yes" if name in anomalous_metrics else "—"
                lines.append(f"| `{name}` | {flag} |")
            lines += [""]

        # ── Footer ────────────────────────────────────────────
        lines += [
            "---",
            "",
            "_Report generated by **Autonomous Incident Intelligence System v1.0.0**_  ",
            f"_Incident ID: `{nd.incident_id}` | Pipeline completed at: {generated_at}_",
        ]

        return "\n".join(lines)
