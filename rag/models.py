"""The contract the model must fill in, and the shape returned to the caller.

`answered` is a boolean, so the guardrail reads a flag instead of matching text
in the prose. `chunk_ids` holds opaque ids, not filenames, so the model picks
from a closed set and the id-to-filename mapping happens in code it cannot reach.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class GroundedAnswer(BaseModel):
    """What the chat model is forced to return."""

    answered: bool = Field(
        description=(
            "true only if the excerpts fully answer the question; "
            "false if they do not cover it"
        )
    )
    answer: str = Field(
        description=(
            "the answer drawn only from the excerpts, or a short polite refusal "
            "when answered is false"
        )
    )
    chunk_ids: list[str] = Field(
        description=(
            "ids of the excerpts the answer came from, e.g. ['c0']; "
            "empty when answered is false"
        )
    )


# Hand-written because strict mode needs every property in `required` and
# additionalProperties false, which the Pydantic generator does not emit.
GROUNDED_ANSWER_SCHEMA = {
    "name": "grounded_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answered": {
                "type": "boolean",
                "description": "true only if the excerpts fully answer the question",
            },
            "answer": {
                "type": "string",
                "description": "the answer, drawn only from the excerpts",
            },
            "chunk_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "ids of the excerpts used, empty when answered is false",
            },
        },
        "required": ["answered", "answer", "chunk_ids"],
        "additionalProperties": False,
    },
}


class Claim(BaseModel):
    """One factual assertion lifted out of the answer, and whether it holds.

    `claim` is generated before `supported`, deliberately. Asking for a verdict
    first makes the model commit to it before it has written down what it is
    judging, and it then rationalises whatever it picked.
    """

    claim: str = Field(description="a single factual assertion made by the answer")
    supported: bool = Field(
        description="true only if the excerpts state this, not merely relate to it"
    )


class SupportVerdict(BaseModel):
    """Every claim in the answer, each judged against the cited excerpts.

    There is no overall boolean here on purpose. Two independently generated
    fields can contradict each other, so the summary is computed in code from
    this list instead of being asked for.
    """

    claims: list[Claim] = Field(description="every claim the answer makes, in order")


SUPPORT_VERDICT_SCHEMA = {
    "name": "support_verdict",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "description": "every factual claim the answer makes, in order",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": "a single factual assertion from the answer",
                        },
                        "supported": {
                            "type": "boolean",
                            "description": (
                                "true only if the excerpts state this, "
                                "not merely relate to it"
                            ),
                        },
                    },
                    "required": ["claim", "supported"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["claims"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class Grounding:
    """What the grounding check found, if it ran."""

    mode: str
    supported: bool
    unsupported_claims: list[str] = field(default_factory=list)
    # Numbers in the answer that appear in no cited excerpt. Found in code, so
    # this holds whatever model wrote the answer.
    figures_not_in_source: list[str] = field(default_factory=list)
    claims_checked: int = 0
    # Set when the checker itself could not run. A broken checker must not be
    # read as a clean bill of health.
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "supported": self.supported,
            "unsupported_claims": self.unsupported_claims,
            "figures_not_in_source": self.figures_not_in_source,
            "claims_checked": self.claims_checked,
            "error": self.error,
        }


@dataclass(frozen=True)
class AskResult:
    """The endpoint's response, plus the trace we keep for debugging and evals."""

    answer: str
    sources: list[str]
    # Everything below is diagnostic. It is returned under "debug" so the
    # documented {answer, sources} contract stays exactly as specified.
    refused_by: str | None = None
    top_score: float = 0.0
    retrieved: list[dict[str, object]] = field(default_factory=list)
    # What the model wrote when it declined. Kept for diagnosis, never shown to
    # the customer: one outcome gets one phrasing. See rag/answer.py.
    original_answer: str | None = None
    grounding: Grounding | None = None

    def to_response(self) -> dict[str, object]:
        """Serialise to the response contract in the README."""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "debug": {
                "refused_by": self.refused_by,
                "top_score": round(self.top_score, 4),
                "retrieved": self.retrieved,
                "original_answer": self.original_answer,
                "grounding": self.grounding.to_dict() if self.grounding else None,
            },
        }
