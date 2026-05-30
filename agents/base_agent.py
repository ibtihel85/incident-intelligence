"""Base agent class — all agents inherit from this."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict

from utils.logger import get_logger


class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent pipeline.
    Each agent receives context, performs its task, and returns an enriched result.
    """

    name: str = "base_agent"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._logger = get_logger(f"agents.{self.name}")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent — wraps process() with timing and logging."""
        self._logger.info(f"[{self.name}] Starting")
        start = time.monotonic()
        try:
            result = self.process(context)
            elapsed = time.monotonic() - start
            self._logger.info(f"[{self.name}] Completed in {elapsed*1000:.1f}ms")
            result.setdefault("_agent_meta", {})[self.name] = {
                "elapsed_ms": round(elapsed * 1000, 2),
                "status": "ok",
            }
            return result
        except Exception as exc:
            elapsed = time.monotonic() - start
            self._logger.error(f"[{self.name}] Failed after {elapsed*1000:.1f}ms: {exc}")
            raise

    @abstractmethod
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Agent-specific implementation. Must return updated context dict."""
        ...
