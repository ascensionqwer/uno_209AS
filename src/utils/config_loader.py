"""Configuration loader for policy generators."""

import json
from pathlib import Path
from typing import Dict, Any

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.json"
_config_cache: Dict[str, Any] = None


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json."""
    global _config_cache
    if _config_cache is None:
        if not _CONFIG_PATH.exists():
            raise FileNotFoundError(f"Config file not found: {_CONFIG_PATH}")
        with open(_CONFIG_PATH, "r") as f:
            _config_cache = json.load(f)
    return _config_cache


def get_policy1_config() -> Dict[str, Any]:
    """Get configuration for policy generator 1."""
    config = load_config()
    return config.get("policy1", {"gamma": 0.95})


def get_policy2_config() -> Dict[str, Any]:
    """Get configuration for policy generator 2."""
    config = load_config()
    return config.get(
        "policy2",
        {
            "num_particles": 1000,
            "mcts_iterations": 1000,
            "planning_horizon": 5,
            "gamma": 0.95,
        },
    )
