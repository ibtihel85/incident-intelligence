"""
MCP Tool Base — all tools inherit from MCPTool and register themselves.
Follows the Model Context Protocol pattern: each tool has a schema,
can be called with structured inputs, and returns structured outputs.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from utils.logger import get_logger

logger = get_logger(__name__)

# Global tool registry
_TOOL_REGISTRY: Dict[str, Type["MCPTool"]] = {}


def register_tool(cls: Type["MCPTool"]) -> Type["MCPTool"]:
    """Class decorator — register a tool in the global MCP registry."""
    _TOOL_REGISTRY[cls.name] = cls
    logger.debug(f"Registered MCP tool: {cls.name}")
    return cls


def get_tool(name: str) -> Optional[Type["MCPTool"]]:
    return _TOOL_REGISTRY.get(name)


def list_tools() -> List[str]:
    return list(_TOOL_REGISTRY.keys())


class MCPTool(ABC):
    """Abstract base class for all MCP tools."""

    name: str = "base_tool"
    description: str = ""
    input_schema: Dict[str, Any] = {}

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._logger = get_logger(f"tools.{self.name}")

    def call(self, **kwargs: Any) -> Dict[str, Any]:
        """Invoke the tool — wraps execute() with timing and error handling."""
        start = time.monotonic()
        self._logger.info(f"Tool invoked: {self.name} | inputs={list(kwargs.keys())}")
        try:
            result = self.execute(**kwargs)
            elapsed = time.monotonic() - start
            result["_tool_meta"] = {
                "tool": self.name,
                "elapsed_ms": round(elapsed * 1000, 2),
                "status": "ok",
            }
            return result
        except Exception as exc:
            elapsed = time.monotonic() - start
            self._logger.error(f"Tool error [{self.name}]: {exc}")
            return {
                "error": str(exc),
                "_tool_meta": {
                    "tool": self.name,
                    "elapsed_ms": round(elapsed * 1000, 2),
                    "status": "error",
                },
            }

    @abstractmethod
    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Tool-specific implementation."""
        ...
