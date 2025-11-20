"""Configuration loader for policy generators."""

import json
import re
from pathlib import Path
from typing import Dict, Any

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.jsonc"
_config_cache: Dict[str, Any] = None


def strip_jsonc_comments(text: str) -> str:
    """
    Strip comments from JSONC (JSON with Comments) format.
    Removes:
    - Single-line comments (// ...)
    - Multi-line comments (/* ... */)
    """
    # Remove single-line comments
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    
    # Remove multi-line comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    
    return text


def load_config() -> Dict[str, Any]:
    """Load configuration from config.jsonc (JSON with Comments)."""
    global _config_cache
    if _config_cache is None:
        if not _CONFIG_PATH.exists():
            raise FileNotFoundError(f"Config file not found: {_CONFIG_PATH}")
        with open(_CONFIG_PATH, "r") as f:
            content = f.read()
        # Strip comments before parsing
        json_content = strip_jsonc_comments(content)
        _config_cache = json.loads(json_content)
    return _config_cache


def get_policy1_config() -> Dict[str, Any]:
    """Get configuration for policy generator 1."""
    config = load_config()
    return config.get("policy1", {
        "gamma": 0.95,
        "num_belief_samples": 50,
        "max_depth": 3
    })


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
            "num_observations": 1000,
        },
    )
