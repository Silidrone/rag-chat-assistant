"""The guardrails: no answer without coverage, no sources on a refusal, and no
citation to a document that was not retrieved."""

from __future__ import annotations

import pytest

from rag.answer import REFUSAL_MESSAGE, answer_question
from rag.config import Settings
from rag.models import GroundedAnswer

from .fakes import ScriptedGenerator, answered, refused


def ask(question, *, index, embedder, generator, settings):
    return answer_question(
        question,
        index=index,
        embedder=embedder,
        generator=generator,
        settings=settings,
    )


class TestRetrievalGate:
    """Guardrail 1: refuse cheaply, before spending a generation."""

    def test_unrelated_question_is_refused_without_calling_the_model(
        self, index, embedder, settings
    ):
        generator = ScriptedGenerator()  # empty: calling it at all is the failure

        result = ask(
            "who won the world cup in 1998",
            index=index,
            embedder=embedder,
            generator=generator,
            settings=settings,
        )

        assert result.sources == []
        assert result.refused_by == "retrieval_gate"
        assert result.answer == REFUSAL_MESSAGE
        assert generator.calls == []

    def test_covered_question_passes_the_gate(self, index, embedder, settings):
        generator = ScriptedGenerator(answered("30 days.", ["c0"]))

        result = ask(
            "refunds cancelled within 30 days",
            index=index,
            embedder=embedder,
            generator=generator,
            settings=settings,
        )

        assert result.refused_by is None
        assert len(generator.calls) == 1

    def test_threshold_is_configurable(self, index, embedder, settings):
        """A gate at 1.0 rejects everything, which proves the setting is live."""
        strict = Settings(**{**settings.__dict__, "min_score": 1.0})
        generator = ScriptedGenerator()

        result = ask(
            "refunds cancelled within 30 days",
            index=index,
            embedder=embedder,
            generator=generator,
            settings=strict,
        )

        assert result.refused_by == "retrieval_gate"
        assert generator.calls == []


class TestModelRefusal:
    """Guardrail 2: the model declares coverage in a field, not in prose."""

    def test_refusal_returns_empty_sources(self, index, embedder, settings):
        generator = ScriptedGenerator(refused("I don't have information on that."))

        result = ask(
            "refunds cancelled within 30 days",
            index=index,
            embedder=embedder,
            generator=generator,
            settings=settings,
        )

        assert result.sources == []
        assert result.refused_by == "model"
        # One outcome, one phrasing: the model's own wording is kept for
        # diagnosis but never reaches the customer.
        assert result.answer == REFUSAL_MESSAGE
        assert result.original_answer == "I don't have information on that."

    def test_cited_ids_are_ignored_when_the_model_refuses(self, index, embedder, settings):
        """A refusal that still cites something must not leak a source."""
        contradictory = ScriptedGenerator(
            GroundedAnswer(answered=False, answer="Not covered.", chunk_ids=["c0", "c1"])
        )

        result = ask(
            "refunds cancelled within 30 days",
            index=index,
            embedder=embedder,
            generator=contradictory,
            settings=settings,
        )

        assert result.sources == []

    def test_blank_model_wording_leaves_no_note(self, index, embedder, settings):
        """Nothing worth diagnosing, so the note stays empty rather than blank."""
        generator = ScriptedGenerator(refused("   "))

        result = ask(
            "refunds cancelled within 30 days",
            index=index,
            embedder=embedder,
            generator=generator,
            settings=settings,
        )

        assert result.answer == REFUSAL_MESSAGE
        assert result.original_answer is None


class TestCitationMapping:
    """Guardrail 3: a citation is only honoured if we retrieved that chunk."""

    def test_valid_id_maps_to_its_filename(self, index, embedder, settings):
        generator = ScriptedGenerator(answered("30 days from delivery.", ["c0"]))

        result = ask(
            "refunds cancelled within 30 days",
            index=index,
            embedder=embedder,
            generator=generator,
            settings=settings,
        )

        assert result.sources == ["cancellation.md"]
        assert result.answer == "30 days from delivery."

    def test_invented_id_is_dropped(self, index, embedder, settings):
        """The model cites one real chunk and one it made up."""
        generator = ScriptedGenerator(answered("30 days.", ["c0", "policies/secret.md"]))

        result = ask(
            "refunds cancelled within 30 days",
            index=index,
            embedder=embedder,
            generator=generator,
            settings=settings,
        )

        assert result.sources == ["cancellation.md"]

    def test_answer_with_no_resolvable_citation_fails_closed(
        self, index, embedder, settings
    ):
        """Claiming an answer while citing nothing real is treated as a refusal.

        We cannot show the customer where the answer came from, so we do not
        make the claim at all.
        """
        generator = ScriptedGenerator(answered("Definitely 90 days.", ["c99"]))

        result = ask(
            "refunds cancelled within 30 days",
            index=index,
            embedder=embedder,
            generator=generator,
            settings=settings,
        )

        assert result.sources == []
        assert result.refused_by == "unciteable"
        assert result.answer == REFUSAL_MESSAGE
        assert "90 days" not in result.answer

    def test_duplicate_sources_are_collapsed(self, index, embedder, settings):
        """Two chunks from one file cite once."""
        from rag.index import Chunk, VectorIndex
        import numpy as np

        chunks = [
            Chunk(id="c0", source="cancellation.md", heading="Cancellation", text="30-day refund window."),
            Chunk(id="c1", source="cancellation.md", heading="Cancellation", text="Access runs to period end."),
        ]
        pair_index = VectorIndex(chunks, np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))
        generator = ScriptedGenerator(answered("30 days, access runs to period end.", ["c0", "c1"]))

        result = answer_question(
            "returns",
            index=pair_index,
            embedder=lambda texts: [[1.0, 0.0] for _ in texts],
            generator=generator,
            settings=settings,
        )

        assert result.sources == ["cancellation.md"]

    def test_citation_order_follows_the_model(self, index, embedder, settings):
        generator = ScriptedGenerator(answered("Both apply.", ["c2", "c0"]))

        result = ask(
            "refunds and invite people",
            index=index,
            embedder=embedder,
            generator=generator,
            settings=settings,
        )

        assert result.sources == ["seats_and_roles.md", "cancellation.md"]


class TestTrace:
    """The debug trace is what makes a bad answer diagnosable after the fact."""

    def test_retrieved_chunks_are_reported(self, index, embedder, settings):
        generator = ScriptedGenerator(answered("30 days.", ["c0"]))

        result = ask(
            "refunds cancelled within 30 days",
            index=index,
            embedder=embedder,
            generator=generator,
            settings=settings,
        )

        assert len(result.retrieved) == settings.top_k
        assert result.retrieved[0]["source"] == "cancellation.md"
        assert result.top_score == pytest.approx(result.retrieved[0]["score"], abs=1e-4)
