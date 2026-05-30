"""Configuration loader — reads YAML settings with env-var override support."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Load YAML config and apply environment variable overrides."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    _apply_env_overrides(config)
    return config


def _apply_env_overrides(config: Dict[str, Any], prefix: str = "AIIS") -> None:
    """
    Override config values from environment variables.
    E.g. AIIS_ANOMALY_DETECTION__CPU__WARNING_THRESHOLD=85
    """
    for key, value in os.environ.items():
        if not key.startswith(f"{prefix}_"):
            continue
        parts = key[len(prefix) + 1:].lower().split("__")
        node = config
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        leaf = parts[-1]
        if leaf in node:
            try:
                node[leaf] = type(node[leaf])(value)
            except (ValueError, TypeError):
                node[leaf] = value
        else:
            node[leaf] = value
