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
        body = response.get_json()

        assert response.status_code == 200
        # cached_vectors is process-wide and depends on what ran before, so it
        # is asserted as present rather than pinned to a number.
        assert isinstance(body.pop("cached_vectors"), int)
        assert body == {
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
        assert "30 days" in body[0]["text"]
        assert body[0]["heading"] == "Cancellation and Refunds"

    def test_served_markdown_round_trips_through_the_editor(self, client_for, index):
        """What the editor loads must re-chunk to the same thing it replaced.

        Serving the bare chunk body would strip the heading on the first save,
        and the heading is embedded with the body on purpose.
        """
        from rag.index import chunk_document

        body = client_for(ScriptedGenerator()).get("/articles").get_json()

        assert len(body) == len({c.source for c in index.chunks})
        for article in body:
            assert article["text"].startswith(f"# {article['heading']}")
            rechunked = chunk_document(article["source"], article["text"])
            original = [c for c in index.chunks if c.source == article["source"]]
            assert [c.heading for c in rechunked] == [c.heading for c in original]
            assert [c.text for c in rechunked] == [c.text for c in original]


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


class TestCallerSuppliedArticles:
    """A caller can bring its own knowledge base, so the demo can be edited
    without the service holding state one visitor can change for another."""

    def test_answers_against_the_supplied_articles(self, client_for):
        generator = ScriptedGenerator(answered("Badgers are nocturnal.", ["c0"]))

        response = client_for(generator).post(
            "/ask",
            json={
                "question": "badgers nocturnal habits",
                "articles": [
                    {"source": "badgers.md", "text": "# Badgers\n\nBadgers are nocturnal."}
                ],
            },
        )
        body = response.get_json()

        assert response.status_code == 200
        assert body["sources"] == ["badgers.md"]
        # The shipped fixture articles are not in play at all.
        assert all(h["source"] == "badgers.md" for h in body["debug"]["retrieved"])

    def test_omitting_articles_uses_the_shipped_set(self, client_for):
        generator = ScriptedGenerator(answered("30 days.", ["c0"]))

        body = client_for(generator).post(
            "/ask", json={"question": "refunds cancelled within 30 days"}
        ).get_json()

        assert body["sources"] == ["cancellation.md"]

    @pytest.mark.parametrize(
        "articles,expected",
        [
            ([], "non-empty list"),
            ("nope", "non-empty list"),
            ([{"source": "a.md", "text": ""}], "is empty"),
            ([{"source": "", "text": "x"}], "no source name"),
            ([{"source": "a/b.md", "text": "x"}], "path separator"),
            ([{"source": "a.md", "text": "x"}, {"source": "A.MD", "text": "y"}], "duplicate"),
        ],
    )
    def test_a_bad_knowledge_base_is_refused_not_repaired(
        self, client_for, articles, expected
    ):
        response = client_for(ScriptedGenerator()).post(
            "/ask", json={"question": "anything", "articles": articles}
        )

        assert response.status_code == 400
        assert expected in response.get_json()["error"]

    def test_oversized_knowledge_base_is_refused(self, client_for):
        """Every distinct chunk is a paid embedding call on a public endpoint."""
        response = client_for(ScriptedGenerator()).post(
            "/ask",
            json={
                "question": "anything",
                "articles": [{"source": "big.md", "text": "x" * 9_000}],
            },
        )

        assert response.status_code == 400
        assert "limit is" in response.get_json()["error"]
