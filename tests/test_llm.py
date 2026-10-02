"""The provider wrapper, driven through a fake requests session.

`OpenAIClient` takes its session by injection, which is what lets these tests
cover retries, batching and malformed responses without a network.
"""

from __future__ import annotations

import json

import pytest
import requests

import llm
from llm import LlmError, OpenAIClient
from rag.models import GROUNDED_ANSWER_SCHEMA, GroundedAnswer


class FakeResponse:
    def __init__(self, status_code: int, body: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("no JSON object could be decoded")
        return self._body


class FakeSession:
    """Returns queued responses in order and records every request.

    A queued item that is an exception is raised instead of returned, which is
    how transport failures (timeouts, resets) are simulated.
    """

    def __init__(self, *responses) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        if not self._responses:
            raise AssertionError("post called more times than expected")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def embedding_response(vectors: list[list[float]]) -> FakeResponse:
    return FakeResponse(
        200,
        {"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]},
    )


def chat_response(payload: dict) -> FakeResponse:
    return FakeResponse(
        200, {"choices": [{"message": {"content": json.dumps(payload)}}]}
    )


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Retries are tested for behaviour, not for wall-clock patience."""
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)


class TestEmbed:
    def test_returns_vectors_in_order(self, settings):
        session = FakeSession(embedding_response([[1.0, 0.0], [0.0, 1.0]]))
        client = OpenAIClient(settings, session)

        assert client.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]

    def test_reorders_by_index_rather_than_trusting_the_response(self, settings):
        """The API documents an index field; relying on list order is luck."""
        session = FakeSession(
            FakeResponse(
                200,
                {
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]},
                    ]
                },
            )
        )
        client = OpenAIClient(settings, session)

        assert client.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]

    def test_empty_input_makes_no_request(self, settings):
        session = FakeSession()

        assert OpenAIClient(settings, session).embed([]) == []
        assert session.calls == []

    def test_large_input_is_batched(self, settings, monkeypatch):
        monkeypatch.setattr(llm, "EMBED_BATCH_SIZE", 2)
        session = FakeSession(
            embedding_response([[1.0], [2.0]]),
            embedding_response([[3.0]]),
        )
        client = OpenAIClient(settings, session)

        assert client.embed(["a", "b", "c"]) == [[1.0], [2.0], [3.0]]
        assert len(session.calls) == 2

    def test_short_count_is_an_error(self, settings):
        session = FakeSession(embedding_response([[1.0]]))
        client = OpenAIClient(settings, session)

        with pytest.raises(LlmError, match="asked for 2 embeddings, got 1"):
            client.embed(["a", "b"])

    def test_malformed_body_is_an_error(self, settings):
        session = FakeSession(FakeResponse(200, {"unexpected": True}))

        with pytest.raises(LlmError, match="unexpected embeddings response"):
            OpenAIClient(settings, session).embed(["a"])


class TestCompleteStructured:
    def call(self, client):
        return client.complete_structured(
            "system",
            "user",
            schema=GROUNDED_ANSWER_SCHEMA,
            output_model=GroundedAnswer,
        )

    def test_parses_and_validates_into_the_model(self, settings):
        session = FakeSession(
            chat_response({"answered": True, "answer": "30 days.", "chunk_ids": ["c0"]})
        )

        result = self.call(OpenAIClient(settings, session))

        assert result == GroundedAnswer(answered=True, answer="30 days.", chunk_ids=["c0"])

    def test_sends_the_strict_schema_and_zero_temperature(self, settings):
        session = FakeSession(
            chat_response({"answered": False, "answer": "No.", "chunk_ids": []})
        )

        self.call(OpenAIClient(settings, session))
        sent = session.calls[0]["json"]

        assert sent["temperature"] == 0
        assert sent["response_format"]["json_schema"]["strict"] is True
        assert sent["model"] == settings.chat_model

    def test_safety_refusal_is_surfaced(self, settings):
        session = FakeSession(
            FakeResponse(200, {"choices": [{"message": {"refusal": "no", "content": None}}]})
        )

        with pytest.raises(LlmError, match="model refused"):
            self.call(OpenAIClient(settings, session))

    def test_invalid_json_is_an_error(self, settings):
        session = FakeSession(
            FakeResponse(200, {"choices": [{"message": {"content": "{not json"}}]})
        )

        with pytest.raises(LlmError, match="did not match the schema"):
            self.call(OpenAIClient(settings, session))

    def test_schema_violation_is_an_error(self, settings):
        session = FakeSession(chat_response({"answered": "maybe"}))

        with pytest.raises(LlmError, match="did not match the schema"):
            self.call(OpenAIClient(settings, session))


class TestRetries:
    def test_transient_status_is_retried_then_succeeds(self, settings):
        session = FakeSession(
            FakeResponse(429, text="rate limited"),
            FakeResponse(503, text="unavailable"),
            embedding_response([[1.0]]),
        )

        assert OpenAIClient(settings, session).embed(["a"]) == [[1.0]]
        assert len(session.calls) == 3

    def test_gives_up_after_the_attempt_limit(self, settings):
        session = FakeSession(*[FakeResponse(500, text="boom") for _ in range(3)])

        with pytest.raises(LlmError, match="failed after 3 attempts"):
            OpenAIClient(settings, session).embed(["a"])

    def test_client_error_is_not_retried(self, settings):
        """A 401 will not fix itself, so burning two more calls is waste."""
        session = FakeSession(FakeResponse(401, text="invalid api key"))

        with pytest.raises(LlmError, match="returned 401"):
            OpenAIClient(settings, session).embed(["a"])
        assert len(session.calls) == 1

    def test_transport_failure_is_retried(self, settings):
        session = FakeSession(
            requests.ConnectionError("connection reset"),
            embedding_response([[1.0]]),
        )

        assert OpenAIClient(settings, session).embed(["a"]) == [[1.0]]
        assert len(session.calls) == 2

    def test_non_json_body_is_an_error(self, settings):
        session = FakeSession(FakeResponse(200))

        with pytest.raises(LlmError, match="non-JSON body"):
            OpenAIClient(settings, session).embed(["a"])

    def test_exhausted_quota_is_not_retried(self, settings):
        """429 means two things. An empty account will not refill in a second."""
        session = FakeSession(
            FakeResponse(
                429,
                {"error": {"type": "insufficient_quota", "message": "no credits"}},
                text="no credits remaining",
            )
        )

        with pytest.raises(LlmError, match="returned 429"):
            OpenAIClient(settings, session).embed(["a"])
        assert len(session.calls) == 1

    def test_rate_limit_is_still_retried(self, settings):
        """The other kind of 429 does clear on its own."""
        session = FakeSession(
            FakeResponse(429, {"error": {"type": "rate_limit_exceeded"}}, text="slow down"),
            embedding_response([[1.0]]),
        )

        assert OpenAIClient(settings, session).embed(["a"]) == [[1.0]]
        assert len(session.calls) == 2
