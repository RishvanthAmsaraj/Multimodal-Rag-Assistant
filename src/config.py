"""Configuration loader for the Multi-Modal RAG Assistant.

Loads YAML config with environment-variable overrides and exposes
dot-notation access so the rest of the codebase doesn't depend on
the underlying dict structure.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

# Load .env if present (no-op if missing)
load_dotenv()

# Default to the repo's configs/config.yaml
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


class Config:
    """Lightweight config object with attribute-style access."""

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                out[key] = value.to_dict()
            else:
                out[key] = value
        return out

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def load_config(path: Optional[str | os.PathLike] = None) -> Config:
    """Load the YAML config from disk.

    Parameters
    ----------
    path : str | os.PathLike, optional
        Path to a config YAML file. If None, uses the default
        ``configs/config.yaml`` shipped with the project.

    Returns
    -------
    Config
        A nested Config object exposing every key as an attribute.
    """
    cfg_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    # Apply env overrides for common LLM keys
    if os.getenv("LLM_PROVIDER"):
        raw.setdefault("generation", {}).setdefault("llm", {})["provider"] = os.getenv("LLM_PROVIDER")
    if os.getenv("LLM_MODEL_NAME"):
        raw.setdefault("generation", {}).setdefault("llm", {})["model_name"] = os.getenv("LLM_MODEL_NAME")

    return Config(raw)


# Module-level convenience
DEFAULT_CONFIG = load_config()