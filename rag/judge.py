"""Guardrail 4: does the answer text say what its source says?

Guardrail 3 proves `sources` names an article the model was actually shown. It
does not prove the answer is true to that article. A reply can cite
`cancellation.md` correctly and still say 45 days where the article says 30,
and that reply is more dangerous than a refusal because it looks sourced.

Two tiers, cheapest first:

1. `figures_not_in_source` is pure code. Every number in the answer has to
   appear somewhere in the cited excerpts. No model call, nothing billed, and
   it covers the error class that costs the most, since a number is the part a
   customer acts on.

2. `judge_support` asks a model to break the answer into claims and mark each
   one. The summary verdict is then computed here, in code, from that list.
   The model is never asked for an overall boolean, because two independently
   generated fields can disagree and the boolean is the one that gets read.

The checker is a self-report, so a clean result is weaker evidence than a dirty
one. `tests/test_judge.py` pins that down by mutating correct answers and
asserting the check catches them.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from .index import Hit
from .models import Grounding, SupportVerdict

Judge = Callable[[str, str], SupportVerdict]

# Digit runs, with decimals kept together so "99.9" is one figure rather than
# two. Words like "thirty" are out of scope: the model writes figures as digits
# because the articles do.
_FIGURE = re.compile(r"\d+(?:\.\d+)?")


def _normalise(figure: str) -> str:
    """Make 05, 5 and 5.0 the same figure, so formatting is not a false alarm."""
    try:
        value = float(figure)
    except ValueError:
        return figure
    return str(int(value)) if value.is_integer() else str(value)


def figures_not_in_source(answer: str, source_text: str) -> list[str]:
    """Figures the answer states that appear nowhere in the cited excerpts.

    Deliberately strict about presence and silent about meaning. It cannot tell
    that "within 30 days" became "after 30 days", only that 30 came from
    somewhere real. Cheap, deterministic, and worth running before any model.
    """
    in_source = {_normalise(f) for f in _FIGURE.findall(source_text)}
    missing: list[str] = []
    for raw in _FIGURE.findall(answer):
        if _normalise(raw) not in in_source and raw not in missing:
            missing.append(raw)
    return missing


def cited_text(sources: Sequence[str], hits: Sequence[Hit]) -> str:
    """The text of just the chunks behind the cited files.

    The checker sees the cited excerpts and nothing else. Handing it the whole
    knowledge base would let it mark a claim supported by some article the
    answer never cited, which is the failure it exists to catch.
    """
    wanted = set(sources)
    return "\n\n".join(
        f"[{hit.chunk.source}] {hit.chunk.heading}\n{hit.chunk.text}"
        for hit in hits
        if hit.chunk.source in wanted
    )


def run_check(
    answer: str,
    sources: Sequence[str],
    hits: Sequence[Hit],
    *,
    mode: str,
    judge: Judge | None,
) -> Grounding | None:
    """Run both tiers and summarise. Returns None when the check is switched off.

    The summary boolean is computed here rather than asked for, and a checker
    that errors reports `supported=False` with the reason. Treating a failed
    check as a pass is how a guardrail quietly stops being one.
    """
    if mode == "off":
        return None

    excerpts = cited_text(sources, hits)
    figures = figures_not_in_source(answer, excerpts)

    if judge is None:
        # Tier 1 only. `supported` reflects what was actually checked, and the
        # note is always set: "figures looked fine" is not "the answer is
        # grounded", and a partial check must not read as a full one.
        return Grounding(
            mode=mode,
            supported=not figures,
            figures_not_in_source=figures,
            claims_checked=0,
            error="no claim checker configured, figures only",
        )

    try:
        verdict = judge(answer, excerpts)
    except Exception as error:  # noqa: BLE001 - any checker failure is the same outcome
        return Grounding(
            mode=mode,
            supported=False,
            figures_not_in_source=figures,
            error=f"grounding check failed: {error}",
        )

    unsupported = [claim.claim for claim in verdict.claims if not claim.supported]
    return Grounding(
        mode=mode,
        supported=not unsupported and not figures,
        unsupported_claims=unsupported,
        figures_not_in_source=figures,
        claims_checked=len(verdict.claims),
    )
