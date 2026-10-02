"""Shared fixtures. Everything here is offline."""

from __future__ import annotations

import pytest

from rag.config import Settings
from rag.index import build_index

from .fakes import HashingEmbedder

# A miniature knowledge base: one topic per file, same shape as data/ but
# deliberately separate from it, so editing the shipped articles cannot quietly
# change what the tests assert.
DOCS = [
    ("cancellation.md", "# Cancellation and Refunds\n\nAnnual plans cancelled within 30 days are refunded in full."),
    ("billing.md", "# Billing and Plans\n\nAnnual plans are discounted 20% against monthly billing."),
    ("seats_and_roles.md", "# Seats and Roles\n\nOwners and admins can invite people; members cannot."),
]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        api_base="https://example.invalid/v1",
        api_key="test-key",
        embed_model="test-embed",
        chat_model="test-chat",
        judge_model="test-judge",
        # Off by default so the pipeline tests exercise guardrails 1 to 3 on
        # their own. tests/test_judge.py turns it on explicitly.
        judge_mode="off",
        top_k=3,
        min_score=0.25,
    )


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def index(embedder: HashingEmbedder):
    return build_index(DOCS, embedder)
