"""Chunking and retrieval mechanics."""

from __future__ import annotations

import numpy as np
import pytest

from rag.index import Chunk, VectorIndex, build_index, chunk_document

from .conftest import DOCS


class TestChunking:
    def test_short_article_stays_whole(self):
        chunks = chunk_document("cancellation.md", DOCS[0][1])

        assert len(chunks) == 1
        assert chunks[0].heading == "Cancellation and Refunds"
        assert "within 30 days" in chunks[0].text

    def test_heading_is_kept_for_embedding(self):
        """The heading carries topic words the body leaves implicit."""
        chunk = chunk_document("cancellation.md", DOCS[0][1])[0]

        assert chunk.embedding_text.startswith("Cancellation and Refunds")
        assert "30 days" in chunk.embedding_text

    def test_each_heading_becomes_its_own_chunk(self):
        text = "# One\n\nFirst body.\n\n# Two\n\nSecond body."

        chunks = chunk_document("multi.md", text)

        assert [c.heading for c in chunks] == ["One", "Two"]
        assert [c.text for c in chunks] == ["First body.", "Second body."]

    def test_long_section_splits_on_paragraph_boundaries(self):
        """Never mid-sentence: a half sentence retrieves badly and reads worse."""
        paragraphs = [f"Paragraph {i} " + "word " * 20 for i in range(6)]
        text = "# Long\n\n" + "\n\n".join(paragraphs)

        chunks = chunk_document("long.md", text, max_chars=300)

        assert len(chunks) > 1
        assert all(c.heading == "Long" for c in chunks)
        # Every paragraph survives intact in exactly one chunk.
        rejoined = "\n\n".join(c.text for c in chunks)
        for paragraph in paragraphs:
            assert paragraph.strip() in rejoined

    def test_oversized_single_paragraph_is_not_split(self):
        """One paragraph over the cap is kept whole rather than cut mid-sentence."""
        text = "# Big\n\n" + "word " * 500

        chunks = chunk_document("big.md", text, max_chars=100)

        assert len(chunks) == 1

    def test_body_without_a_heading_is_kept(self):
        chunks = chunk_document("plain.md", "Just a bare sentence.")

        assert len(chunks) == 1
        assert chunks[0].heading == ""


class TestIndex:
    def test_ids_are_unique_and_positional(self, index):
        ids = [chunk.id for chunk in index.chunks]

        assert ids == ["c0", "c1", "c2"]
        assert index.get("c0").source == "cancellation.md"
        assert index.get("nope") is None

    def test_search_ranks_the_matching_document_first(self, index, embedder):
        query = embedder(["owners and admins can invite people"])[0]

        hits = index.search(query, top_k=3)

        assert hits[0].chunk.source == "seats_and_roles.md"
        assert hits[0].score > hits[-1].score

    def test_search_respects_top_k(self, index, embedder):
        query = embedder(["billing"])[0]

        assert len(index.search(query, top_k=1)) == 1
        assert len(index.search(query, top_k=99)) == 3

    def test_unrelated_query_scores_near_zero(self, index, embedder):
        """The property the retrieval gate depends on."""
        query = embedder(["photosynthesis chlorophyll stomata"])[0]

        assert index.search(query, top_k=1)[0].score < 0.25

    def test_mismatched_vector_count_is_rejected(self):
        chunks = [Chunk(id="c0", source="a.md", heading="A", text="a")]

        with pytest.raises(ValueError, match="1 chunks but 2 vectors"):
            VectorIndex(chunks, np.zeros((2, 4), dtype=np.float32))

    def test_zero_vector_does_not_blow_up(self):
        """An all-zero embedding must not divide by zero during normalisation."""
        chunks = [Chunk(id="c0", source="a.md", heading="A", text="a")]
        index = VectorIndex(chunks, np.zeros((1, 4), dtype=np.float32))

        assert index.search([0.0, 0.0, 0.0, 0.0], top_k=1)[0].score == 0.0

    def test_empty_knowledge_base_is_an_error(self, embedder):
        with pytest.raises(ValueError, match="no chunks to index"):
            build_index([], embedder)

    def test_documents_are_embedded_in_one_call(self, embedder):
        build_index(DOCS, embedder)

        assert len(embedder.calls) == 1
        assert len(embedder.calls[0]) == 3
