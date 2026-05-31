"""
Evaluation Benchmark
Runs all synthetic scenarios through the pipeline and computes:
  - Anomaly detection accuracy
  - Root cause correctness
  - Report quality score
  - Overall pipeline score
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.orchestrator import IncidentOrchestrator
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Ground-truth definitions ──────────────────────────────────

@dataclass
class ScenarioGroundTruth:
    scenario_file: str
    expected_root_cause_category: str     # RootCauseCategory.value
    expected_severity: str                # Severity.value
    expected_anomaly_types: List[str]     # list of AnomalyType.value
    min_anomaly_count: int
    description: str


GROUND_TRUTH: List[ScenarioGroundTruth] = [
    ScenarioGroundTruth(
        scenario_file="data/synthetic/cpu_spike.json",
        expected_root_cause_category="RESOURCE_EXHAUSTION",
        expected_severity="CRITICAL",
        expected_anomaly_types=["CPU_SPIKE", "HIGH_ERROR_RATE", "HIGH_LATENCY"],
        min_anomaly_count=2,
        description="CPU spike from runaway ML training job",
    ),
    ScenarioGroundTruth(
        scenario_file="data/synthetic/memory_leak.json",
        expected_root_cause_category="APPLICATION_BUG",
        expected_severity="HIGH",
        expected_anomaly_types=["MEMORY_LEAK", "MEMORY_SPIKE", "LOG_ERROR_BURST"],
        min_anomaly_count=2,
        description="Memory leak via unclosed DB connections",
    ),
    ScenarioGroundTruth(
        scenario_file="data/synthetic/database_failure.json",
        expected_root_cause_category="DEPENDENCY_FAILURE",
        expected_severity="CRITICAL",
        expected_anomaly_types=["HIGH_ERROR_RATE", "HIGH_LATENCY"],
        min_anomaly_count=2,
        description="PostgreSQL connection pool exhaustion",
    ),
    ScenarioGroundTruth(
        scenario_file="data/synthetic/normal_behavior.json",
        expected_root_cause_category="UNKNOWN",
        expected_severity="LOW",
        expected_anomaly_types=[],
        min_anomaly_count=0,
        description="Normal system operation — no incidents",
    ),
]


@dataclass
class ScenarioResult:
    scenario: str
    description: str
    # Anomaly
    detected_anomaly_count: int
    expected_anomaly_types: List[str]
    detected_anomaly_types: List[str]
    anomaly_type_recall: float
    anomaly_count_ok: bool
    # Root cause
    expected_rc: str
    detected_rc: str
    rc_correct: bool
    rc_confidence: float
    # Severity
    expected_severity: str
    detected_severity: str
    severity_correct: bool
    # Report quality
    report_length: int
    report_has_executive_summary: bool
    report_has_recommendations: bool
    report_has_causal_chain: bool
    report_quality_score: float
    # Overall
    overall_score: float
    passed: bool


@dataclass
class BenchmarkSummary:
    total_scenarios: int
    passed: int
    anomaly_detection_accuracy: float
    root_cause_accuracy: float
    severity_accuracy: float
    avg_report_quality: float
    overall_score: float
    scenario_results: List[ScenarioResult] = field(default_factory=list)


class IncidentBenchmark:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.orchestrator = IncidentOrchestrator(config)

    def run(self) -> BenchmarkSummary:
        print("\n" + "=" * 70)
        print("  AUTONOMOUS INCIDENT INTELLIGENCE SYSTEM — BENCHMARK")
        print("=" * 70)

        results: List[ScenarioResult] = []
        for gt in GROUND_TRUTH:
            result = self._evaluate_scenario(gt)
            results.append(result)
            self._print_scenario_result(result)

        summary = self._compute_summary(results)
        self._print_summary(summary)
        return summary

    def _evaluate_scenario(self, gt: ScenarioGroundTruth) -> ScenarioResult:
        path = Path(gt.scenario_file)
        if not path.exists():
            logger.error(f"Scenario file not found: {path}")
            raise FileNotFoundError(path)

        with open(path) as f:
            raw_data = json.load(f)

        pipeline_result = self.orchestrator.run(raw_data)

        # ── Anomaly evaluation ────────────────────────────────
        detected_types = list(set(
            a["anomaly_type"] for a in pipeline_result.get("anomalies", [])
        ))
        detected_count = pipeline_result.get("anomaly_count", 0)

        if gt.expected_anomaly_types:
            hits = sum(1 for t in gt.expected_anomaly_types if t in detected_types)
            anomaly_recall = hits / len(gt.expected_anomaly_types)
        else:
            # Normal scenario: good score if no anomalies detected
            anomaly_recall = 1.0 if detected_count == 0 else max(0.0, 1.0 - detected_count * 0.2)

        anomaly_count_ok = detected_count >= gt.min_anomaly_count

        # ── Root cause evaluation ─────────────────────────────
        rc = pipeline_result.get("root_cause") or {}
        detected_rc = rc.get("category", "UNKNOWN") if rc else "UNKNOWN"
        rc_correct = detected_rc == gt.expected_root_cause_category
        rc_confidence = rc.get("confidence", 0.0) if rc else 0.0

        # ── Severity evaluation ───────────────────────────────
        detected_severity = pipeline_result.get("severity", "UNKNOWN")
        severity_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3, "UNKNOWN": -1}
        expected_rank = severity_order.get(gt.expected_severity, -1)
        detected_rank = severity_order.get(detected_severity, -1)
        severity_correct = abs(expected_rank - detected_rank) <= 1  # within 1 level

        # ── Report quality evaluation ─────────────────────────
        report = pipeline_result.get("report", "")
        has_summary = "Executive Summary" in report
        has_recs = "Fix Recommendations" in report
        has_chain = "Causal Chain" in report or "Root Cause" in report
        report_quality = (
            (0.3 if has_summary else 0.0) +
            (0.3 if has_recs else 0.0) +
            (0.2 if has_chain else 0.0) +
            (0.1 if len(report) > 500 else 0.0) +
            (0.1 if "---" in report else 0.0)
        )

        # ── Overall score ─────────────────────────────────────
        overall = (
            anomaly_recall * 0.30 +
            (1.0 if rc_correct else 0.0) * 0.35 +
            (1.0 if severity_correct else 0.0) * 0.20 +
            report_quality * 0.15
        )

        return ScenarioResult(
            scenario=gt.scenario_file,
            description=gt.description,
            detected_anomaly_count=detected_count,
            expected_anomaly_types=gt.expected_anomaly_types,
            detected_anomaly_types=detected_types,
            anomaly_type_recall=round(anomaly_recall, 3),
            anomaly_count_ok=anomaly_count_ok,
            expected_rc=gt.expected_root_cause_category,
            detected_rc=detected_rc,
            rc_correct=rc_correct,
            rc_confidence=round(rc_confidence, 3),
            expected_severity=gt.expected_severity,
            detected_severity=detected_severity,
            severity_correct=severity_correct,
            report_length=len(report),
            report_has_executive_summary=has_summary,
            report_has_recommendations=has_recs,
            report_has_causal_chain=has_chain,
            report_quality_score=round(report_quality, 3),
            overall_score=round(overall, 3),
            passed=overall >= 0.65,
        )

    def _compute_summary(self, results: List[ScenarioResult]) -> BenchmarkSummary:
        n = len(results)
        return BenchmarkSummary(
            total_scenarios=n,
            passed=sum(1 for r in results if r.passed),
            anomaly_detection_accuracy=round(sum(r.anomaly_type_recall for r in results) / n, 3),
            root_cause_accuracy=round(sum(1 for r in results if r.rc_correct) / n, 3),
            severity_accuracy=round(sum(1 for r in results if r.severity_correct) / n, 3),
            avg_report_quality=round(sum(r.report_quality_score for r in results) / n, 3),
            overall_score=round(sum(r.overall_score for r in results) / n, 3),
            scenario_results=results,
        )

    def _print_scenario_result(self, r: ScenarioResult) -> None:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        print(f"\n{'─'*60}")
        print(f"  {status}  {r.description}")
        print(f"{'─'*60}")
        print(f"  Anomaly recall:    {r.anomaly_type_recall*100:>5.1f}%  "
              f"(detected {r.detected_anomaly_count} anomalies)")
        print(f"  Root cause:        {'✓' if r.rc_correct else '✗'}  "
              f"expected={r.expected_rc:<25} got={r.detected_rc} "
              f"(conf={r.rc_confidence*100:.0f}%)")
        print(f"  Severity:          {'✓' if r.severity_correct else '✗'}  "
              f"expected={r.expected_severity:<10} got={r.detected_severity}")
        print(f"  Report quality:    {r.report_quality_score*100:>5.1f}%  "
              f"({r.report_length} chars)")
        print(f"  Overall score:     {r.overall_score*100:>5.1f}%")

    def _print_summary(self, s: BenchmarkSummary) -> None:
        bar_len = 30
        def bar(val: float) -> str:
            filled = int(val * bar_len)
            return "█" * filled + "░" * (bar_len - filled)

        print("\n" + "=" * 70)
        print("  BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"  Scenarios:              {s.passed}/{s.total_scenarios} passed")
        print(f"  Anomaly Detection:      {bar(s.anomaly_detection_accuracy)}  {s.anomaly_detection_accuracy*100:.1f}%")
        print(f"  Root Cause Accuracy:    {bar(s.root_cause_accuracy)}  {s.root_cause_accuracy*100:.1f}%")
        print(f"  Severity Accuracy:      {bar(s.severity_accuracy)}  {s.severity_accuracy*100:.1f}%")
        print(f"  Report Quality:         {bar(s.avg_report_quality)}  {s.avg_report_quality*100:.1f}%")
        print(f"  ─────────────────────────────────────────────────────────")
        print(f"  OVERALL SCORE:          {bar(s.overall_score)}  {s.overall_score*100:.1f}%")
        print("=" * 70 + "\n")
