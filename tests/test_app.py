"""The HTTP contract, driven through the real pipeline with fake clients."""

from __future__ import annotations

import pytest

from app import create_app
from llm import LlmError
from rag.answer import REFUSAL_MESSAGE

from .fakes import ScriptedGenerator, answered, refused


@pytest.fixture
def client_for(index, embedder, settings):
    def build(generator):
        app = create_app(
            settings=settings, index=index, embedder=embedder, generator=generator
        )
        app.config.update(TESTING=True)
        return app.test_client()

    return build


class TestHealth:
    def test_reports_what_was_loaded(self, client_for):
        response = client_for(ScriptedGenerator()).get("/health")

        assert response.status_code == 200
        assert response.get_json() == {
            "ok": True,
            "documents_loaded": 3,
            "chunks_indexed": 3,
            "top_k": 3,
            "min_score": 0.25,
            "judge_mode": "off",
        }


class TestArticles:
    def test_serves_the_indexed_knowledge_base(self, client_for):
        response = client_for(ScriptedGenerator()).get("/articles")
        body = response.get_json()

        assert response.status_code == 200
        assert [a["source"] for a in body] == [
            "cancellation.md",
            "billing.md",
            "seats_and_roles.md",
        ]
        assert body[0]["id"] == "c0"
        assert "30 days" in body[0]["text"]
        assert body[0]["heading"] == "Cancellation and Refunds"

    def test_what_is_served_is_what_is_retrievable(self, client_for, index):
        """A reader checking an answer must see the text that was embedded."""
        body = client_for(ScriptedGenerator()).get("/articles").get_json()

        assert len(body) == len(index.chunks)
        assert {a["id"] for a in body} == {c.id for c in index.chunks}


class TestAskContract:
    def test_grounded_question_returns_answer_and_source(self, client_for):
        generator = ScriptedGenerator(answered("A full refund within 30 days.", ["c0"]))

        response = client_for(generator).post(
            "/ask", json={"question": "refunds cancelled within 30 days"}
        )
        body = response.get_json()

        assert response.status_code == 200
        assert body["answer"] == "A full refund within 30 days."
        assert body["sources"] == ["cancellation.md"]

    def test_out_of_kb_question_declines_with_empty_sources(self, client_for):
        response = client_for(ScriptedGenerator()).post(
            "/ask", json={"question": "who won the world cup in 1998"}
        )
        body = response.get_json()

        assert response.status_code == 200
        assert body["sources"] == []
        assert body["answer"] == REFUSAL_MESSAGE

    def test_model_refusal_also_declines(self, client_for):
        generator = ScriptedGenerator(refused("I don't have that information."))

        response = client_for(generator).post(
            "/ask", json={"question": "do you price match a refund"}
        )

        assert response.get_json()["sources"] == []

    @pytest.mark.parametrize(
        "payload", [{}, {"question": ""}, {"question": "   "}, {"q": "wrong key"}]
    )
    def test_missing_question_is_a_400(self, client_for, payload):
        response = client_for(ScriptedGenerator()).post("/ask", json=payload)

        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_non_json_body_is_a_400_not_a_crash(self, client_for):
        response = client_for(ScriptedGenerator()).post(
            "/ask", data="not json", content_type="text/plain"
        )

        assert response.status_code == 400


class TestUpstreamFailure:
    def test_provider_failure_is_a_503(self, client_for):
        def broken(system, user):
            raise LlmError("connection reset")

        response = client_for(broken).post(
            "/ask", json={"question": "refunds cancelled within 30 days"}
        )

        assert response.status_code == 503
        assert "language model unavailable" in response.get_json()["error"]
