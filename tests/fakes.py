"""Test doubles for the embedder, generator and grounding checker."""

from __future__ import annotations

import hashlib
import re

from rag.models import Claim, GroundedAnswer, SupportVerdict

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """A deterministic bag-of-words embedder.

    Not semantic, but it has real cosine behaviour: shared vocabulary scores
    high, disjoint vocabulary near zero. That is what the gate tests need.
    """

    def __init__(self, dims: int = 256) -> None:
        self.dims = dims
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.md5(token.encode()).digest()
            # Four bytes, not one: a single byte caps the bucket space at 256
            # whatever `dims` says, and unrelated vocabulary then collides hard
            # enough to clear the retrieval gate, which is the property these
            # tests exist to check.
            vector[int.from_bytes(digest[:4], "big") % self.dims] += 1.0
        return vector


class ScriptedGenerator:
    """Returns queued answers in order and records the prompts it was given."""

    def __init__(self, *answers: GroundedAnswer) -> None:
        self._answers = list(answers)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> GroundedAnswer:
        self.calls.append((system, user))
        if not self._answers:
            raise AssertionError("generator called more times than expected")
        return self._answers.pop(0)


def answered(answer: str, chunk_ids: list[str]) -> GroundedAnswer:
    return GroundedAnswer(answered=True, answer=answer, chunk_ids=chunk_ids)


def refused(answer: str = "I don't have that information.") -> GroundedAnswer:
    return GroundedAnswer(answered=False, answer=answer, chunk_ids=[])


class ScriptedJudge:
    """A grounding checker whose verdict is decided by the test.

    `marks` maps a substring to whether a claim containing it is supported, so
    a test says what the checker found without hand-building claim objects.
    """

    def __init__(self, *claims: tuple[str, bool]) -> None:
        self._claims = list(claims)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, answer: str, excerpts: str) -> SupportVerdict:
        self.calls.append((answer, excerpts))
        return SupportVerdict(
            claims=[Claim(claim=text, supported=ok) for text, ok in self._claims]
        )


class BrokenJudge:
    """A checker that cannot run, to pin down what happens when it fails."""

    def __call__(self, answer: str, excerpts: str) -> SupportVerdict:
        raise RuntimeError("judge unavailable")
