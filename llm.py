"""Thin wrapper over an OpenAI-compatible API.

Config comes from a validated `Settings` built once at start-up.
`complete_structured` sends a strict JSON schema and returns a validated
Pydantic model. 429, 5xx and transport errors get three attempts with backoff.

Point LLM_API_BASE at anything OpenAI-compatible. Swapping provider means
reimplementing these two methods and nothing else.
"""

from __future__ import annotations

import json
import time
from typing import Any, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from rag.config import Settings

ModelT = TypeVar("ModelT", bound=BaseModel)

# Embeddings are sent in batches so a large knowledge base does not become one
# enormous request. The six current articles fit in a single batch.
EMBED_BATCH_SIZE = 100

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 1.0
RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

# 429 means either a rate limit, which clears on its own, or an exhausted
# quota, which does not. The error type in the body tells them apart.
NON_RETRYABLE_ERROR_TYPES = frozenset({"insufficient_quota", "billing_not_active"})

EMBED_TIMEOUT = 30
CHAT_TIMEOUT = 60


class LlmError(RuntimeError):
    """Raised when the provider cannot be reached or returns something unusable."""


class OpenAIClient:
    """Embeddings and structured chat completion against an OpenAI-compatible API."""

    def __init__(self, settings: Settings, session: requests.Session | None = None) -> None:
        self._settings = settings
        self._session = session or requests.Session()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input string, in the same order."""
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            payload = {"model": self._settings.embed_model, "input": batch}
            body = self._post("/embeddings", payload, timeout=EMBED_TIMEOUT)
            try:
                # Sort by index rather than trusting response order.
                items = sorted(body["data"], key=lambda item: item["index"])
                vectors.extend(item["embedding"] for item in items)
            except (KeyError, TypeError) as error:
                raise LlmError(f"unexpected embeddings response: {body}") from error

        if len(vectors) != len(texts):
            raise LlmError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
        return vectors

    def complete_structured(
        self,
        system: str,
        user: str,
        *,
        schema: dict[str, Any],
        output_model: type[ModelT],
        model: str | None = None,
    ) -> ModelT:
        """Return a validated `output_model`, forced by a strict JSON schema.

        No repair loop: under strict mode a violation is a provider bug and
        should be loud.
        """
        payload = {
            "model": model or self._settings.chat_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Deterministic: the same question against the same excerpts should
            # give the same answer, which is what makes the eval set meaningful.
            "temperature": 0,
            "response_format": {"type": "json_schema", "json_schema": schema},
        }
        body = self._post("/chat/completions", payload, timeout=CHAT_TIMEOUT)

        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise LlmError(f"unexpected chat response: {body}") from error

        # A safety refusal comes back in its own field, with content set to null.
        if message.get("refusal"):
            raise LlmError(f"model refused to respond: {message['refusal']}")

        content = message.get("content")
        if not content:
            raise LlmError("model returned empty content")

        try:
            return output_model.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as error:
            raise LlmError(f"model output did not match the schema: {error}") from error

    def _post(self, path: str, payload: dict[str, Any], *, timeout: int) -> dict[str, Any]:
        """POST to the API, retrying only the failures that retrying can fix.

        Rate limits, 5xx and transport errors get another attempt. A 4xx does
        not, since the request will stay wrong.
        """
        url = f"{self._settings.api_base}{path}"
        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._session.post(
                    url, headers=headers, json=payload, timeout=timeout
                )
            except requests.RequestException as error:
                last_error = error  # connection reset, timeout, DNS: retry
            else:
                if not _is_retryable(response):
                    if response.status_code >= 400:
                        raise LlmError(
                            f"{path} returned {response.status_code}: "
                            f"{response.text[:200]}"
                        )
                    try:
                        return response.json()
                    except ValueError as error:
                        raise LlmError(f"{path} returned non-JSON body") from error
                last_error = LlmError(
                    f"{path} returned {response.status_code}: {response.text[:200]}"
                )

            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_SECONDS * attempt)

        raise LlmError(f"{path} failed after {MAX_ATTEMPTS} attempts: {last_error}")


def _is_retryable(response: Any) -> bool:
    """Is another attempt worth making, given the status and the error body?"""
    if response.status_code not in RETRY_STATUS:
        return False
    return _error_type(response) not in NON_RETRYABLE_ERROR_TYPES


def _error_type(response: Any) -> str:
    """Pull `error.type` out of the body, tolerating any body at all."""
    try:
        body = response.json()
    except ValueError:
        return ""
    if not isinstance(body, dict):
        return ""
    error = body.get("error")
    return error.get("type", "") if isinstance(error, dict) else ""
