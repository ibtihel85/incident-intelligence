"""
Unit tests for all agents.
Run with: pytest tests/ -v
"""

import json
from pathlib import Path

import pytest

from utils.config import load_config
from utils.models import (
    AnomalyType, NormalizedData, RootCauseCategory, Severity,
)
from agents.data_ingestion_agent import DataIngestionAgent
from agents.anomaly_detection_agent import AnomalyDetectionAgent
from agents.root_cause_analysis_agent import RootCauseAnalysisAgent
from agents.knowledge_retrieval_agent import KnowledgeRetrievalAgent
from agents.fix_recommendation_agent import FixRecommendationAgent
from agents.report_generator_agent import ReportGeneratorAgent
from pipeline.orchestrator import IncidentOrchestrator


@pytest.fixture
def config():
    return load_config("config/settings.yaml")


@pytest.fixture
def cpu_spike_data():
    with open("data/synthetic/cpu_spike.json") as f:
        return json.load(f)


@pytest.fixture
def memory_leak_data():
    with open("data/synthetic/memory_leak.json") as f:
        return json.load(f)


@pytest.fixture
def db_failure_data():
    with open("data/synthetic/database_failure.json") as f:
        return json.load(f)


@pytest.fixture
def normal_data():
    with open("data/synthetic/normal_behavior.json") as f:
        return json.load(f)


# ── DataIngestionAgent ────────────────────────────────────────

class TestDataIngestionAgent:
    def test_ingests_cpu_spike(self, config, cpu_spike_data):
        agent = DataIngestionAgent(config)
        ctx = agent.run({"raw_data": cpu_spike_data})
        nd: NormalizedData = ctx["normalized_data"]
        assert nd.incident_id == "INC-CPU-2024-001"
        assert nd.scenario == "cpu_spike"
        assert "cpu_usage" in nd.metrics
        assert len(nd.logs) > 0
        assert len(nd.metrics["cpu_usage"]) == 12

    def test_generates_incident_id_if_missing(self, config):
        agent = DataIngestionAgent(config)
        raw = {"scenario": "test", "metrics": {}, "logs": []}
        ctx = agent.run({"raw_data": raw})
        assert ctx["incident_id"].startswith("INC-")

    def test_normalises_log_levels(self, config):
        agent = DataIngestionAgent(config)
        raw = {
            "scenario": "test",
            "metrics": {},
            "logs": [{"timestamp": "2024-01-01T00:00:00Z", "level": "WARNING",
                       "message": "test", "service": "svc"}],
        }
        ctx = agent.run({"raw_data": raw})
        nd: NormalizedData = ctx["normalized_data"]
        assert nd.logs[0].level == "WARN"

    def test_extracts_services_from_logs(self, config, memory_leak_data):
        agent = DataIngestionAgent(config)
        ctx = agent.run({"raw_data": memory_leak_data})
        nd: NormalizedData = ctx["normalized_data"]
        assert "recommendation-service" in nd.services or "postgres" in nd.services


# ── AnomalyDetectionAgent ─────────────────────────────────────

class TestAnomalyDetectionAgent:
    def _ingest(self, config, raw):
        ctx = DataIngestionAgent(config).run({"raw_data": raw})
        return AnomalyDetectionAgent(config).run(ctx)

    def test_detects_cpu_spike(self, config, cpu_spike_data):
        ctx = self._ingest(config, cpu_spike_data)
        types = [a.anomaly_type for a in ctx["anomalies"]]
        assert AnomalyType.CPU_SPIKE in types

    def test_cpu_spike_is_critical(self, config, cpu_spike_data):
        ctx = self._ingest(config, cpu_spike_data)
        cpu_anomalies = [a for a in ctx["anomalies"] if a.anomaly_type == AnomalyType.CPU_SPIKE]
        assert any(a.severity == Severity.CRITICAL for a in cpu_anomalies)

    def test_detects_memory_leak(self, config, memory_leak_data):
        ctx = self._ingest(config, memory_leak_data)
        types = [a.anomaly_type for a in ctx["anomalies"]]
        assert AnomalyType.MEMORY_LEAK in types or AnomalyType.MEMORY_SPIKE in types

    def test_detects_high_error_rate(self, config, db_failure_data):
        ctx = self._ingest(config, db_failure_data)
        types = [a.anomaly_type for a in ctx["anomalies"]]
        assert AnomalyType.HIGH_ERROR_RATE in types

    def test_normal_baseline_has_few_anomalies(self, config, normal_data):
        ctx = self._ingest(config, normal_data)
        anomalies = ctx["anomalies"]
        critical = [a for a in anomalies if a.severity == Severity.CRITICAL]
        assert len(critical) == 0

    def test_anomaly_ids_are_unique(self, config, cpu_spike_data):
        ctx = self._ingest(config, cpu_spike_data)
        ids = [a.anomaly_id for a in ctx["anomalies"]]
        assert len(ids) == len(set(ids))


# ── RootCauseAnalysisAgent ────────────────────────────────────

class TestRootCauseAnalysisAgent:
    def _run_pipeline(self, config, raw):
        ctx = DataIngestionAgent(config).run({"raw_data": raw})
        ctx = AnomalyDetectionAgent(config).run(ctx)
        return RootCauseAnalysisAgent(config).run(ctx)

    def test_cpu_spike_root_cause(self, config, cpu_spike_data):
        ctx = self._run_pipeline(config, cpu_spike_data)
        rc = ctx["root_cause"]
        assert rc is not None
        assert rc.category == RootCauseCategory.RESOURCE_EXHAUSTION

    def test_memory_leak_root_cause(self, config, memory_leak_data):
        ctx = self._run_pipeline(config, memory_leak_data)
        rc = ctx["root_cause"]
        assert rc is not None
        assert rc.category in (RootCauseCategory.APPLICATION_BUG, RootCauseCategory.DEPENDENCY_FAILURE)

    def test_db_failure_root_cause(self, config, db_failure_data):
        ctx = self._run_pipeline(config, db_failure_data)
        rc = ctx["root_cause"]
        assert rc is not None
        assert rc.category == RootCauseCategory.DEPENDENCY_FAILURE

    def test_confidence_is_valid(self, config, cpu_spike_data):
        ctx = self._run_pipeline(config, cpu_spike_data)
        rc = ctx["root_cause"]
        assert rc is not None
        assert 0.0 <= rc.confidence <= 1.0

    def test_causal_chain_is_non_empty(self, config, cpu_spike_data):
        ctx = self._run_pipeline(config, cpu_spike_data)
        rc = ctx["root_cause"]
        assert rc is not None and len(rc.causal_chain) >= 1

    def test_no_anomalies_returns_none(self, config):
        raw = {"scenario": "test", "metrics": {}, "logs": []}
        ctx = DataIngestionAgent(config).run({"raw_data": raw})
        ctx["anomalies"] = []
        ctx = RootCauseAnalysisAgent(config).run(ctx)
        assert ctx["root_cause"] is None


# ── KnowledgeRetrievalAgent ───────────────────────────────────

class TestKnowledgeRetrievalAgent:
    def _run(self, config, raw):
        ctx = DataIngestionAgent(config).run({"raw_data": raw})
        ctx = AnomalyDetectionAgent(config).run(ctx)
        ctx = RootCauseAnalysisAgent(config).run(ctx)
        return KnowledgeRetrievalAgent(config).run(ctx)

    def test_returns_similar_incidents(self, config, cpu_spike_data):
        ctx = self._run(config, cpu_spike_data)
        assert "similar_incidents" in ctx
        assert isinstance(ctx["similar_incidents"], list)

    def test_similar_incidents_have_resolution(self, config, memory_leak_data):
        ctx = self._run(config, memory_leak_data)
        for inc in ctx["similar_incidents"]:
            assert inc.resolution


# ── FixRecommendationAgent ────────────────────────────────────

class TestFixRecommendationAgent:
    def _run(self, config, raw):
        ctx = DataIngestionAgent(config).run({"raw_data": raw})
        ctx = AnomalyDetectionAgent(config).run(ctx)
        ctx = RootCauseAnalysisAgent(config).run(ctx)
        ctx = KnowledgeRetrievalAgent(config).run(ctx)
        return FixRecommendationAgent(config).run(ctx)

    def test_generates_recommendations(self, config, cpu_spike_data):
        ctx = self._run(config, cpu_spike_data)
        recs = ctx["recommendations"]
        assert len(recs) >= 2

    def test_recommendations_are_prioritised(self, config, cpu_spike_data):
        ctx = self._run(config, cpu_spike_data)
        priorities = [r.priority for r in ctx["recommendations"]]
        assert 1 in priorities

    def test_db_failure_recommends_db_fix(self, config, db_failure_data):
        ctx = self._run(config, db_failure_data)
        actions = " ".join(r.action.lower() for r in ctx["recommendations"])
        assert "database" in actions or "connection" in actions or "postgres" in actions.lower()


# ── ReportGeneratorAgent ──────────────────────────────────────

class TestReportGeneratorAgent:
    def _run_full(self, config, raw):
        ctx = DataIngestionAgent(config).run({"raw_data": raw})
        ctx = AnomalyDetectionAgent(config).run(ctx)
        ctx = RootCauseAnalysisAgent(config).run(ctx)
        ctx = KnowledgeRetrievalAgent(config).run(ctx)
        ctx = FixRecommendationAgent(config).run(ctx)
        return ReportGeneratorAgent(config).run(ctx)

    def test_report_is_generated(self, config, cpu_spike_data):
        ctx = self._run_full(config, cpu_spike_data)
        assert "report" in ctx
        assert len(ctx["report"]) > 200

    def test_report_has_required_sections(self, config, cpu_spike_data):
        ctx = self._run_full(config, cpu_spike_data)
        report = ctx["report"]
        for section in ["Executive Summary", "Detected Anomalies", "Root Cause", "Fix Recommendations"]:
            assert section in report, f"Missing section: {section}"

    def test_report_contains_incident_id(self, config, cpu_spike_data):
        ctx = self._run_full(config, cpu_spike_data)
        assert "INC-CPU-2024-001" in ctx["report"]


# ── Full pipeline integration ─────────────────────────────────

class TestFullPipeline:
    def test_cpu_spike_end_to_end(self, config, cpu_spike_data):
        orchestrator = IncidentOrchestrator(config)
        result = orchestrator.run(cpu_spike_data)
        assert result["severity"] == "CRITICAL"
        assert result["anomaly_count"] >= 2
        assert result["root_cause"]["category"] == "RESOURCE_EXHAUSTION"

    def test_memory_leak_end_to_end(self, config, memory_leak_data):
        orchestrator = IncidentOrchestrator(config)
        result = orchestrator.run(memory_leak_data)
        assert result["anomaly_count"] >= 1
        assert result["root_cause"]["category"] in ("APPLICATION_BUG", "DEPENDENCY_FAILURE")

    def test_db_failure_end_to_end(self, config, db_failure_data):
        orchestrator = IncidentOrchestrator(config)
        result = orchestrator.run(db_failure_data)
        assert result["severity"] in ("CRITICAL", "HIGH")
        assert result["root_cause"]["category"] == "DEPENDENCY_FAILURE"

    def test_pipeline_result_schema(self, config, cpu_spike_data):
        orchestrator = IncidentOrchestrator(config)
        result = orchestrator.run(cpu_spike_data)
        required_keys = ["incident_id", "severity", "anomaly_count", "anomalies",
                         "root_cause", "similar_incidents", "recommendations",
                         "report", "pipeline_metadata"]
        for key in required_keys:
            assert key in result, f"Missing key in result: {key}"
