"""Tests for KnowledgeService."""

import pytest
from pathlib import Path

# Import using relative path for package context
try:
    from ..service import KnowledgeService, get_knowledge_service, search_knowledge
except ImportError:
    from service import KnowledgeService, get_knowledge_service, search_knowledge


class TestKnowledgeService:
    """Test KnowledgeService initialization and basic operations."""

    def test_init(self):
        """Test KnowledgeService initializes correctly."""
        kb = KnowledgeService(auto_load=True)
        assert kb is not None
        assert kb.retriever is not None

    def test_len(self):
        """Test KnowledgeService returns document count."""
        kb = KnowledgeService(auto_load=True)
        doc_count = len(kb)
        assert doc_count > 0
        assert doc_count >= 18000  # We expect at least 18K documents

    def test_stats(self):
        """Test KnowledgeService stats method."""
        kb = KnowledgeService(auto_load=True)
        stats = kb.stats()
        assert stats["status"] == "loaded"
        assert "total_documents" in stats
        assert "categories" in stats
        assert stats["total_documents"] > 0

    def test_search_returns_results(self):
        """Test search returns non-empty results."""
        kb = KnowledgeService(auto_load=True)
        results = kb.search("value investing", k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_search_result_structure(self):
        """Test search results have expected structure."""
        kb = KnowledgeService(auto_load=True)
        results = kb.search("margin of safety", k=1)
        result = results[0]
        assert "content" in result
        assert "metadata" in result
        assert "similarity" in result
        assert "id" in result

    def test_search_with_category_filter(self):
        """Test search with category filter."""
        kb = KnowledgeService(auto_load=True)

        # Test Investing_Books category
        investing_results = kb.search("value investing", k=5, category="Investing_Books")
        for result in investing_results:
            assert result["metadata"]["category"] == "Investing_Books"

        # Test Psychological_Books category
        psych_results = kb.search("loss aversion", k=5, category="Psychological_Books")
        for result in psych_results:
            assert result["metadata"]["category"] == "Psychological_Books"

    def test_get_context(self):
        """Test get_context returns formatted string."""
        kb = KnowledgeService(auto_load=True)
        context = kb.get_context("risk management", k=2)
        assert isinstance(context, str)
        assert len(context) > 0
        assert "[" in context  # Has book title markers

    def test_get_context_with_max_tokens(self):
        """Test get_context respects max_tokens limit."""
        kb = KnowledgeService(auto_load=True)
        context_short = kb.get_context("investing", k=5, max_tokens=500)
        context_long = kb.get_context("investing", k=5, max_tokens=2000)
        assert len(context_short) <= len(context_long)

    def test_inject_knowledge(self):
        """Test inject_knowledge enhances prompt."""
        kb = KnowledgeService(auto_load=True)
        original_prompt = "Analyze this stock."
        enhanced = kb.inject_knowledge(original_prompt, "value investing", k=2)
        assert original_prompt in enhanced
        assert "== Relevant Knowledge ==" in enhanced or "Use the following knowledge" in enhanced

    def test_search_min_similarity(self):
        """Test search with minimum similarity filter."""
        kb = KnowledgeService(auto_load=True)
        results_no_filter = kb.search("investing", k=10)
        results_filtered = kb.search("investing", k=10, min_similarity=0.5)
        assert len(results_filtered) <= len(results_no_filter)


class TestKnowledgeServiceErrors:
    """Test KnowledgeService error handling."""

    def test_invalid_index_path(self):
        """Test error on invalid index path."""
        with pytest.raises(FileNotFoundError):
            KnowledgeService(index_path="/nonexistent/path", auto_load=True)

    def test_search_before_load(self):
        """Test search fails if index not loaded."""
        kb = KnowledgeService(auto_load=False)
        with pytest.raises(RuntimeError):
            kb.search("test")

    def test_add_pdfs_nonexistent_file(self):
        """Test add_pdfs handles missing files gracefully."""
        kb = KnowledgeService(auto_load=True)
        # Should not raise, just print warning
        added = kb.add_pdfs(["/nonexistent/file.pdf"])
        assert added == 0


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_get_knowledge_service(self):
        """Test get_knowledge_service returns singleton."""
        kb1 = get_knowledge_service()
        kb2 = get_knowledge_service()
        assert kb1 is kb2  # Same instance

    def test_search_knowledge(self):
        """Test search_knowledge convenience function."""
        results = search_knowledge("diversification", k=2)
        assert len(results) > 0
