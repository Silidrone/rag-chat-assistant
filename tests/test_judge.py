"""Guardrail 4: is the answer text true to the article it cited?

The checker is a self-report, so a clean verdict proves less than a dirty one.
Most of what is below therefore mutates a correct answer and asserts the check
notices, rather than asserting that correct answers pass.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rag.answer import answer_question
from rag.index import Chunk, VectorIndex
from rag.judge import cited_text, figures_not_in_source, run_check
from rag.models import Claim, SupportVerdict

from .fakes import BrokenJudge, ScriptedGenerator, ScriptedJudge, answered

SOURCE = "Annual plans cancelled within 30 days are refunded in full."


def one_chunk_index() -> VectorIndex:
    chunk = Chunk(id="c0", source="cancellation.md", heading="Cancellation", text=SOURCE)
    return VectorIndex([chunk], np.array([[1.0, 0.0]], dtype=np.float32))


def ask(question, *, index, settings, generator, judge=None):
    return answer_question(
        question,
        index=index,
        embedder=lambda texts: [[1.0, 0.0] for _ in texts],
        generator=generator,
        settings=settings,
        judge=judge,
    )


class TestFigureCheck:
    """Tier 1, pure code. No model, so it holds whatever wrote the answer."""

    def test_matching_figure_passes(self):
        assert figures_not_in_source("Refunded within 30 days.", SOURCE) == []

    @pytest.mark.parametrize("wrong", ["45", "29", "300", "3"])
    def test_mutated_figure_is_caught(self, wrong):
        answer = f"Refunded within {wrong} days."

        assert figures_not_in_source(answer, SOURCE) == [wrong]

    def test_formatting_is_not_a_false_alarm(self):
        """05, 5 and 5.0 are the same figure."""
        assert figures_not_in_source("Within 030 days.", SOURCE) == []

    def test_decimals_stay_whole(self):
        """99.9 is one figure, not a 99 and a 9."""
        assert figures_not_in_source("Uptime is 99.9%.", "targets 99.9% uptime") == []
        assert figures_not_in_source("Uptime is 99.99%.", "targets 99.9% uptime") == ["99.99"]


class TestCitedText:
    def test_only_cited_articles_are_shown_to_the_checker(self, index, embedder):
        """A claim must not be marked supported by an article nobody cited."""
        hits = index.search(embedder(["billing"])[0], top_k=3)

        text = cited_text(["billing.md"], hits)

        assert "Billing" in text
        assert "Cancellation" not in text


class TestModes:
    def test_off_runs_nothing(self, settings):
        judge = ScriptedJudge(("anything", False))

        result = ask(
            "q",
            index=one_chunk_index(),
            settings=replace(settings, judge_mode="off"),
            generator=ScriptedGenerator(answered("Refunded within 30 days.", ["c0"])),
            judge=judge,
        )

        assert result.grounding is None
        assert judge.calls == []

    def test_shadow_records_the_finding_but_keeps_the_answer(self, settings):
        """The point of shadow: measure the check without it touching anyone."""
        result = ask(
            "q",
            index=one_chunk_index(),
            settings=replace(settings, judge_mode="shadow"),
            generator=ScriptedGenerator(answered("Refunded within 45 days.", ["c0"])),
            judge=ScriptedJudge(("refund window is 45 days", False)),
        )

        assert result.answer == "Refunded within 45 days."
        assert result.sources == ["cancellation.md"]
        assert result.refused_by is None
        assert result.grounding.supported is False
        assert result.grounding.unsupported_claims == ["refund window is 45 days"]
        assert result.grounding.figures_not_in_source == ["45"]

    def test_enforce_withholds_an_unsupported_answer(self, settings):
        result = ask(
            "q",
            index=one_chunk_index(),
            settings=replace(settings, judge_mode="enforce"),
            generator=ScriptedGenerator(answered("Refunded within 45 days.", ["c0"])),
            judge=ScriptedJudge(("refund window is 45 days", False)),
        )

        assert result.refused_by == "ungrounded"
        assert result.sources == []
        assert "45 days" not in result.answer
        # Kept for diagnosis, never shown to the customer.
        assert result.original_answer == "Refunded within 45 days."

    def test_enforce_passes_a_supported_answer_through(self, settings):
        result = ask(
            "q",
            index=one_chunk_index(),
            settings=replace(settings, judge_mode="enforce"),
            generator=ScriptedGenerator(answered("Refunded within 30 days.", ["c0"])),
            judge=ScriptedJudge(("refund window is 30 days", True)),
        )

        assert result.refused_by is None
        assert result.answer == "Refunded within 30 days."
        assert result.grounding.supported is True
        assert result.grounding.claims_checked == 1


class TestTiersAreIndependent:
    def test_code_catches_a_figure_the_model_waved_through(self, settings):
        """Tier 1 does not defer to tier 2. A judge that misses still fails."""
        result = ask(
            "q",
            index=one_chunk_index(),
            settings=replace(settings, judge_mode="shadow"),
            generator=ScriptedGenerator(answered("Refunded within 45 days.", ["c0"])),
            judge=ScriptedJudge(("refund window", True)),  # judge says fine
        )

        assert result.grounding.supported is False
        assert result.grounding.figures_not_in_source == ["45"]
        assert result.grounding.unsupported_claims == []


class TestCheckerFailure:
    def test_a_broken_checker_is_not_a_pass(self, settings):
        """Fail closed. An unusable check must not read as a clean one."""
        result = ask(
            "q",
            index=one_chunk_index(),
            settings=replace(settings, judge_mode="shadow"),
            generator=ScriptedGenerator(answered("Refunded within 30 days.", ["c0"])),
            judge=BrokenJudge(),
        )

        assert result.grounding.supported is False
        assert "judge unavailable" in result.grounding.error

    def test_enforce_withholds_when_the_checker_breaks(self, settings):
        result = ask(
            "q",
            index=one_chunk_index(),
            settings=replace(settings, judge_mode="enforce"),
            generator=ScriptedGenerator(answered("Refunded within 30 days.", ["c0"])),
            judge=BrokenJudge(),
        )

        assert result.refused_by == "ungrounded"

    def test_no_checker_configured_is_reported_not_hidden(self, settings):
        """Tier 1 alone is a partial check and says so, even when it passes."""
        index = one_chunk_index()
        hits = index.search([1.0, 0.0], top_k=1)

        grounding = run_check(
            "Refunded within 30 days.",
            ["cancellation.md"],
            hits,
            mode="shadow",
            judge=None,
        )

        assert grounding.figures_not_in_source == []
        assert grounding.supported is True
        assert grounding.claims_checked == 0
        assert grounding.error == "no claim checker configured, figures only"


class TestVerdictIsComputedNotAsked:
    def test_summary_comes_from_the_claim_list(self):
        """No overall boolean is requested, so two fields cannot disagree."""
        assert "supported" not in SupportVerdict.model_fields
        assert set(SupportVerdict.model_fields) == {"claims"}

    def test_claim_text_is_generated_before_its_verdict(self):
        """Field order is load-bearing: state the claim, then judge it."""
        assert list(Claim.model_fields) == ["claim", "supported"]
