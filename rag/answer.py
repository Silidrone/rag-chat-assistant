"""The /ask pipeline: retrieve, gate, generate, verify.

    embed question
      -> search the index
      -> [gate]     refuse cheaply if nothing is close enough        (code)
      -> generate   answer + answered flag + cited ids, schema-bound  (model)
      -> [verify]   map cited ids to filenames, drop anything unknown (code)

Only the middle step is the model's. The two guardrails around it are code, so
the endpoint cannot cite a document it did not retrieve and cannot return
sources alongside a refusal.

`embedder` and `generator` are injected, so tests drive this with no network.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .config import Settings
from .index import Hit, VectorIndex
from .judge import Judge, run_check
from .models import AskResult, GroundedAnswer
from .prompts import COMPANY_NAME, SYSTEM_PROMPT, build_user_prompt

Embedder = Callable[[list[str]], list[list[float]]]
Generator = Callable[[str, str], GroundedAnswer]

REFUSAL_MESSAGE = (
    f"I don't have information on that in {COMPANY_NAME}'s help centre. "
    f"Please contact {COMPANY_NAME} support and they'll be able to help."
)


def answer_question(
    question: str,
    *,
    index: VectorIndex,
    embedder: Embedder,
    generator: Generator,
    settings: Settings,
    judge: Judge | None = None,
) -> AskResult:
    """Answer a question from the index, or refuse when it is not covered."""
    query_vector = embedder([question])[0]
    hits = index.search(query_vector, settings.top_k)
    top_score = hits[0].score if hits else 0.0
    trace = _trace(hits)

    def refuse(reason: str, *, original: str | None = None) -> AskResult:
        """Every refusal, whatever caused it, says the same thing."""
        return AskResult(
            answer=REFUSAL_MESSAGE,
            sources=[],
            refused_by=reason,
            top_score=top_score,
            retrieved=trace,
            original_answer=original,
        )

    # Guardrail 1: nothing retrieved is close enough to be worth reading.
    # Tuned for recall, so the model makes the fine-grained call below.
    if not hits or top_score < settings.min_score:
        return refuse("retrieval_gate")

    result = generator(SYSTEM_PROMPT, build_user_prompt(question, hits))

    # Guardrail 2: the model says the excerpts do not answer the question. Its
    # own wording is kept for diagnosis and never shown to the customer.
    if not result.answered:
        return refuse("model", original=result.answer.strip() or None)

    # Guardrail 3: an id the model invented, or one belonging to a chunk it was
    # never shown, is dropped here rather than reaching the caller.
    sources = _resolve_sources(result.chunk_ids, hits)

    # Fail closed: an answer with no resolvable citation is discarded.
    if not sources:
        return refuse("unciteable")

    answer = result.answer.strip()

    # Guardrail 4: the citation is real, but is the text true to it? Runs after
    # the answer exists, so it can only ever withhold one, never write one.
    grounding = run_check(
        answer, sources, hits, mode=settings.judge_mode, judge=judge
    )

    # In shadow the finding is recorded and the customer still gets the answer,
    # which is what makes it safe to leave on while the check earns trust.
    if grounding and not grounding.supported and settings.judge_mode == "enforce":
        return AskResult(
            answer=REFUSAL_MESSAGE,
            sources=[],
            refused_by="ungrounded",
            top_score=top_score,
            retrieved=trace,
            original_answer=answer,
            grounding=grounding,
        )

    return AskResult(
        answer=answer,
        sources=sources,
        refused_by=None,
        top_score=top_score,
        retrieved=trace,
        grounding=grounding,
    )


def _resolve_sources(chunk_ids: Sequence[str], hits: Sequence[Hit]) -> list[str]:
    """Map cited chunk ids to filenames, keeping first-cited order.

    Ids are looked up in the retrieved set, not the whole index, so a citation
    is only honoured if that chunk was genuinely in front of the model.
    """
    retrieved = {hit.chunk.id: hit.chunk.source for hit in hits}
    sources: list[str] = []
    for chunk_id in chunk_ids:
        source = retrieved.get(chunk_id.strip())
        if source is not None and source not in sources:
            sources.append(source)
    return sources


def _trace(hits: Sequence[Hit]) -> list[dict[str, object]]:
    """What was retrieved and how well it scored, for debugging and evals."""
    return [
        {"id": hit.chunk.id, "source": hit.chunk.source, "score": round(hit.score, 4)}
        for hit in hits
    ]
