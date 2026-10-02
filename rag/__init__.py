"""Retrieval-augmented answering over a support knowledge base.

Kept free of any HTTP or provider imports so the pipeline can be tested, and
reused, without standing up a web server or holding an API key.
"""

from .answer import REFUSAL_MESSAGE, answer_question
from .config import ConfigurationError, Settings, load_settings
from .index import Chunk, Hit, VectorIndex, build_index, chunk_document
from .judge import figures_not_in_source, run_check
from .models import (
    GROUNDED_ANSWER_SCHEMA,
    SUPPORT_VERDICT_SCHEMA,
    AskResult,
    Claim,
    Grounding,
    GroundedAnswer,
    SupportVerdict,
)

__all__ = [
    "GROUNDED_ANSWER_SCHEMA",
    "SUPPORT_VERDICT_SCHEMA",
    "REFUSAL_MESSAGE",
    "AskResult",
    "Chunk",
    "Claim",
    "ConfigurationError",
    "Grounding",
    "GroundedAnswer",
    "Hit",
    "Settings",
    "SupportVerdict",
    "VectorIndex",
    "answer_question",
    "build_index",
    "chunk_document",
    "figures_not_in_source",
    "load_settings",
    "run_check",
]
