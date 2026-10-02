"""Settings, read from the environment once at start-up and then frozen.

Reading config in one place means a missing key fails immediately with a clear
message, rather than surfacing as a confusing 500 from inside a request.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4o"
# The grounding check is a smaller job than answering: read one answer against
# the excerpts it cited and say which claims they do not support. A cheaper
# model is the point, since this doubles the calls per answered question.
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_JUDGE_MODE = "shadow"
JUDGE_MODES = ("off", "shadow", "enforce")

# Both picked with `evaluate.py`. top_k=4 takes recall from 95% to 100% on the
# eval set. min_score is the highest gate that blocks no covered question.
DEFAULT_TOP_K = 4
DEFAULT_MIN_SCORE = 0.25


class ConfigurationError(RuntimeError):
    """Raised when the environment is missing or holds an unusable value."""


@dataclass(frozen=True)
class Settings:
    """Validated configuration. Frozen so it cannot drift after start-up."""

    api_base: str
    api_key: str
    embed_model: str
    chat_model: str
    judge_model: str
    judge_mode: str
    top_k: int
    min_score: float


def load_settings(environment: Mapping[str, str] | None = None) -> Settings:
    """Read settings from the environment, failing loudly on bad input.

    Passing `environment` explicitly keeps this testable without touching the
    real process environment.
    """
    if environment is None:
        # Picks up .env outside Docker. Real environment variables win, so
        # compose's env_file still wins, and a missing .env is a no-op.
        load_dotenv()
        environment = os.environ

    api_key = environment.get("LLM_API_KEY", "").strip()
    if not api_key or api_key == "your-own-key-here":
        raise ConfigurationError(
            "LLM_API_KEY is not set. Copy .env.example to .env and put your key in it."
        )

    min_score = _read_float(environment, "RAG_MIN_SCORE", DEFAULT_MIN_SCORE)
    if not 0.0 <= min_score <= 1.0:
        raise ConfigurationError(
            f"RAG_MIN_SCORE must be between 0.0 and 1.0, got {min_score}."
        )

    top_k = _read_int(environment, "RAG_TOP_K", DEFAULT_TOP_K)
    if top_k < 1:
        raise ConfigurationError(f"RAG_TOP_K must be at least 1, got {top_k}.")

    judge_mode = environment.get("JUDGE_MODE", DEFAULT_JUDGE_MODE).strip().lower()
    if judge_mode not in JUDGE_MODES:
        raise ConfigurationError(
            f"JUDGE_MODE must be one of {', '.join(JUDGE_MODES)}, got {judge_mode!r}."
        )

    return Settings(
        api_base=environment.get("LLM_API_BASE", DEFAULT_API_BASE).rstrip("/"),
        api_key=api_key,
        embed_model=environment.get("LLM_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        chat_model=environment.get("LLM_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        judge_model=environment.get("LLM_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        judge_mode=judge_mode,
        top_k=top_k,
        min_score=min_score,
    )


def _read_int(environment: Mapping[str, str], name: str, fallback: int) -> int:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return fallback
    try:
        return int(raw)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}.") from error


def _read_float(environment: Mapping[str, str], name: str, fallback: float) -> float:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return fallback
    try:
        return float(raw)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}.") from error
