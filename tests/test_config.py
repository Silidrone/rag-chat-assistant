"""Config is validated at start-up so bad input fails loudly, not mid-request."""

from __future__ import annotations

import pytest

from rag.config import ConfigurationError, load_settings

MINIMAL = {"LLM_API_KEY": "sk-test"}


class TestApiKey:
    def test_missing_key_is_rejected(self):
        with pytest.raises(ConfigurationError, match="LLM_API_KEY is not set"):
            load_settings({})

    def test_placeholder_key_is_rejected(self):
        """The value shipped in .env.example must not silently 401 later."""
        with pytest.raises(ConfigurationError, match="LLM_API_KEY is not set"):
            load_settings({"LLM_API_KEY": "your-own-key-here"})


class TestDefaults:
    def test_only_the_key_is_required(self):
        settings = load_settings(MINIMAL)

        assert settings.api_key == "sk-test"
        assert settings.chat_model == "gpt-4o"
        assert settings.embed_model == "text-embedding-3-small"
        assert settings.top_k == 4

    def test_trailing_slash_is_stripped_from_the_base_url(self):
        """Otherwise every request URL ends up with a double slash."""
        settings = load_settings({**MINIMAL, "LLM_API_BASE": "https://x.test/v1/"})

        assert settings.api_base == "https://x.test/v1"

    def test_blank_values_fall_back_to_defaults(self):
        """docker compose passes through empty vars as empty strings, not absent."""
        settings = load_settings({**MINIMAL, "RAG_TOP_K": "", "RAG_MIN_SCORE": ""})

        assert settings.top_k == 4
        assert settings.min_score == 0.25


class TestValidation:
    def test_non_numeric_threshold_is_rejected(self):
        with pytest.raises(ConfigurationError, match="must be a number"):
            load_settings({**MINIMAL, "RAG_MIN_SCORE": "high"})

    def test_out_of_range_threshold_is_rejected(self):
        with pytest.raises(ConfigurationError, match="between 0.0 and 1.0"):
            load_settings({**MINIMAL, "RAG_MIN_SCORE": "2.5"})

    def test_non_integer_top_k_is_rejected(self):
        with pytest.raises(ConfigurationError, match="must be an integer"):
            load_settings({**MINIMAL, "RAG_TOP_K": "3.5"})

    def test_zero_top_k_is_rejected(self):
        with pytest.raises(ConfigurationError, match="at least 1"):
            load_settings({**MINIMAL, "RAG_TOP_K": "0"})

    def test_settings_are_frozen(self):
        settings = load_settings(MINIMAL)

        with pytest.raises(Exception):
            settings.api_key = "changed"  # type: ignore[misc]
