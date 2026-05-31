"""Unit tests for all MCP tools."""

import pytest
from tools.log_query import LogQueryTool
from tools.metrics_retrieval import MetricsRetrievalTool
from tools.knowledge_search import KnowledgeSearchTool
from tools.code_executor import CodeExecutorTool
from tools.base import list_tools


@pytest.fixture
def config():
    return {}


SAMPLE_LOGS = [
    {"timestamp": "2024-01-01T00:00:00Z", "level": "INFO",  "service": "api", "message": "Request received"},
    {"timestamp": "2024-01-01T00:01:00Z", "level": "WARN",  "service": "api", "message": "High latency detected"},
    {"timestamp": "2024-01-01T00:02:00Z", "level": "ERROR", "service": "db",  "message": "Connection failed"},
    {"timestamp": "2024-01-01T00:03:00Z", "level": "ERROR", "service": "api", "message": "Database timeout error"},
    {"timestamp": "2024-01-01T00:04:00Z", "level": "FATAL", "service": "db",  "message": "Database crashed"},
]

SAMPLE_METRICS = {
    "cpu_usage": [
        {"timestamp": f"2024-01-01T00:0{i}:00Z", "value": v}
        for i, v in enumerate([10, 15, 80, 95, 98, 97])
    ]
}


class TestLogQueryTool:
    def test_filter_by_level(self, config):
        tool = LogQueryTool(config)
        result = tool.call(logs=SAMPLE_LOGS, level="ERROR")
        assert result["total_matched"] == 3   # ERROR + FATAL

    def test_filter_by_service(self, config):
        tool = LogQueryTool(config)
        result = tool.call(logs=SAMPLE_LOGS, service="db")
        assert result["total_matched"] == 2

    def test_filter_by_keyword(self, config):
        tool = LogQueryTool(config)
        result = tool.call(logs=SAMPLE_LOGS, keyword="database")
        assert result["total_matched"] == 2

    def test_limit_respected(self, config):
        tool = LogQueryTool(config)
        result = tool.call(logs=SAMPLE_LOGS, limit=2)
        assert len(result["entries"]) <= 2

    def test_level_distribution(self, config):
        tool = LogQueryTool(config)
        result = tool.call(logs=SAMPLE_LOGS, level="INFO")
        assert "INFO" in result["level_distribution"]

    def test_empty_logs(self, config):
        tool = LogQueryTool(config)
        result = tool.call(logs=[], level="ERROR")
        assert result["total_matched"] == 0


class TestMetricsRetrievalTool:
    def test_basic_stats(self, config):
        tool = MetricsRetrievalTool(config)
        result = tool.call(metrics=SAMPLE_METRICS, metric_name="cpu_usage")
        stats = result["stats"]
        assert stats["min"] < stats["max"]
        assert stats["mean"] > 0
        assert "p95" in stats

    def test_trend_increasing(self, config):
        tool = MetricsRetrievalTool(config)
        result = tool.call(metrics=SAMPLE_METRICS, metric_name="cpu_usage")
        assert result["trend"] == "increasing"

    def test_unknown_metric(self, config):
        tool = MetricsRetrievalTool(config)
        result = tool.call(metrics=SAMPLE_METRICS, metric_name="nonexistent")
        assert "error" in result

    def test_peak_detection(self, config):
        tool = MetricsRetrievalTool(config)
        result = tool.call(metrics=SAMPLE_METRICS, metric_name="cpu_usage")
        assert result["peak_count"] >= 1


class TestKnowledgeSearchTool:
    def test_returns_results(self, config):
        tool = KnowledgeSearchTool(config)
        result = tool.call(query="memory leak python database connection")
        assert result["results_found"] >= 1

    def test_cpu_query(self, config):
        tool = KnowledgeSearchTool(config)
        result = tool.call(query="cpu spike runaway process training job")
        incidents = result.get("incidents", [])
        assert len(incidents) >= 1

    def test_results_have_resolution(self, config):
        tool = KnowledgeSearchTool(config)
        result = tool.call(query="database connection pool exhausted timeout")
        for inc in result["incidents"]:
            assert inc["resolution"]

    def test_top_k_respected(self, config):
        tool = KnowledgeSearchTool(config)
        result = tool.call(query="cpu memory error", top_k=2)
        assert len(result["incidents"]) <= 2


class TestCodeExecutorTool:
    def test_safe_command_simulated(self, config):
        tool = CodeExecutorTool(config)
        result = tool.call(command="uptime", simulate=True)
        assert result["returncode"] == 0
        assert "load average" in result["stdout"]

    def test_unsafe_command_blocked(self, config):
        tool = CodeExecutorTool(config)
        result = tool.call(command="rm -rf /")
        assert "error" in result

    def test_empty_command_error(self, config):
        tool = CodeExecutorTool(config)
        result = tool.call(command="")
        assert "error" in result


class TestToolRegistry:
    def test_all_tools_registered(self):
        tools = list_tools()
        assert "log_query" in tools
        assert "metrics_retrieval" in tools
        assert "knowledge_search" in tools
        assert "code_executor" in tools
