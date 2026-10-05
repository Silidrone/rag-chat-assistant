"""Grounded Q&A over a support knowledge base.

    POST /ask  {"question": "..."}  ->  {"answer": "...", "sources": ["billing.md"]}

The endpoint validates the request, calls `rag.answer.answer_question`, and
serialises the result. Everything else lives in rag/, which knows nothing about
HTTP.

Built as a factory so tests can supply a fake index, embedder and generator.
"""

from __future__ import annotations

import glob
import os

from flask import Flask, jsonify, request

from llm import LlmError, OpenAIClient
from rag.answer import Embedder, Generator, answer_question
from rag.judge import Judge
from rag.config import ConfigurationError, Settings, load_settings
from rag.index import VectorIndex, build_index
from rag.prompts import JUDGE_SYSTEM_PROMPT, build_judge_prompt
from rag.models import (
    GROUNDED_ANSWER_SCHEMA,
    SUPPORT_VERDICT_SCHEMA,
    GroundedAnswer,
    SupportVerdict,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_documents(data_dir: str = DATA_DIR) -> list[tuple[str, str]]:
    """Load every markdown article in data/ as (source_name, text)."""
    docs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read().strip()))
    return docs


def create_app(
    *,
    settings: Settings | None = None,
    index: VectorIndex | None = None,
    embedder: Embedder | None = None,
    generator: Generator | None = None,
    judge: Judge | None = None,
) -> Flask:
    """Build the app, wiring in real clients unless fakes are supplied."""
    if settings is None:
        settings = load_settings()

    wants_judge = settings.judge_mode != "off"
    if embedder is None or generator is None or (wants_judge and judge is None):
        client = OpenAIClient(settings)
        embedder = embedder or client.embed
        generator = generator or _make_generator(client)
        if wants_judge and judge is None:
            judge = _make_judge(client, settings)

    # Indexed once at start-up: the knowledge base is static, so paying the
    # embedding cost per request would buy nothing.
    if index is None:
        index = build_index(load_documents(), embedder)

    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(
            {
                "ok": True,
                "documents_loaded": len({chunk.source for chunk in index.chunks}),
                "chunks_indexed": len(index),
                # The effective retrieval config. A caller cannot read a score
                # in /ask without knowing the gate it was measured against.
                "top_k": settings.top_k,
                "min_score": settings.min_score,
                "judge_mode": settings.judge_mode,
            }
        )

    @app.get("/articles")
    def articles():
        """The knowledge base, exactly as the retriever holds it.

        Served from the index rather than re-read from disk, so what a reader
        checks an answer against is the same text that was embedded and
        retrieved, chunk boundaries and all.
        """
        return jsonify(
            [
                {
                    "id": chunk.id,
                    "source": chunk.source,
                    "heading": chunk.heading,
                    "text": chunk.text,
                }
                for chunk in index.chunks
            ]
        )

    @app.post("/ask")
    def ask():
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()
        if not question:
            return jsonify({"error": 'send JSON like {"question": "..."}'}), 400

        try:
            result = answer_question(
                question,
                index=index,
                embedder=embedder,
                generator=generator,
                settings=settings,
                judge=judge,
            )
        except LlmError as error:
            # Upstream is unreachable or misbehaving, which is not the
            # caller's fault. 503 says retry; 500 would say the bug is here.
            app.logger.exception("llm call failed")
            return jsonify({"error": f"language model unavailable: {error}"}), 503

        return jsonify(result.to_response())

    return app


def _make_judge(client: OpenAIClient, settings: Settings) -> Judge:
    """Adapt the LLM client to the (answer, excerpts) -> SupportVerdict seam.

    Runs on `judge_model`, which defaults to a cheaper model than the one that
    writes answers. Reading one answer against the excerpts it cited is a
    smaller job than producing it.
    """

    def check(answer: str, excerpts: str) -> SupportVerdict:
        return client.complete_structured(
            JUDGE_SYSTEM_PROMPT,
            build_judge_prompt(answer, excerpts),
            schema=SUPPORT_VERDICT_SCHEMA,
            output_model=SupportVerdict,
            model=settings.judge_model,
        )

    return check


def _make_generator(client: OpenAIClient) -> Generator:
    """Adapt the LLM client to the (system, user) -> GroundedAnswer seam."""

    def generate(system: str, user: str) -> GroundedAnswer:
        return client.complete_structured(
            system,
            user,
            schema=GROUNDED_ANSWER_SCHEMA,
            output_model=GroundedAnswer,
        )

    return generate


if __name__ == "__main__":
    try:
        create_app().run(host="0.0.0.0", port=5000)
    except ConfigurationError as error:
        raise SystemExit(f"Configuration error: {error}")
