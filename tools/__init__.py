"""
Tools package — importing this registers all MCP tools automatically.
"""

from tools.base import MCPTool, get_tool, list_tools, register_tool  # noqa: F401
from tools.log_query import LogQueryTool  # noqa: F401
from tools.metrics_retrieval import MetricsRetrievalTool  # noqa: F401
from tools.knowledge_search import KnowledgeSearchTool  # noqa: F401
from tools.code_executor import CodeExecutorTool  # noqa: F401

__all__ = [
    "MCPTool",
    "register_tool",
    "get_tool",
    "list_tools",
    "LogQueryTool",
    "MetricsRetrievalTool",
    "KnowledgeSearchTool",
    "CodeExecutorTool",
]
