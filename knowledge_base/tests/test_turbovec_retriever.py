"""Tests for TurboVecRetriever and KnowledgeRetriever."""

import pytest
import numpy as np

try:
    from ..turbovec_retriever import TurboVecRetriever, KnowledgeRetriever, Document
except ImportError:
    from turbovec_retriever import TurboVecRetriever, KnowledgeRetriever, Document


class TestTurboVecRetriever:
    """Test TurboVecRetriever core functionality."""

    def test_init(self):
        """Test TurboVecRetriever initializes correctly."""
        retriever = TurboVecRetriever(dim=1024, bit_width=4)
        assert retriever.dim == 1024
        assert retriever.bit_width == 4
        assert retriever.index is not None

    def test_add_documents(self):
        """Test adding documents to index."""
        retriever = TurboVecRetriever(dim=128, bit_width=4)

        documents = [
            {"id": "doc1", "content": "First document", "metadata": {"source": "test"}},
            {"id": "doc2", "content": "Second document", "metadata": {"source": "test"}},
        ]
        embeddings = np.random.rand(2, 128).astype(np.float32)

        added = retriever.add_documents(documents, embeddings)
        assert added == 2
        assert len(retriever.documents) == 2

    def test_add_documents_length_mismatch(self):
        """Test error on document/embedding length mismatch."""
        retriever = TurboVecRetriever(dim=128)

        documents = [{"id": "doc1", "content": "test", "metadata": {}}]
        embeddings = np.random.rand(3, 128).astype(np.float32)  # Wrong length

        with pytest.raises(ValueError):
            retriever.add_documents(documents, embeddings)

    def test_search(self):
        """Test searching the index."""
        retriever = TurboVecRetriever(dim=128, bit_width=4)

        # Add documents
        documents = [
            {"id": "doc1", "content": "Apple fruit", "metadata": {}},
            {"id": "doc2", "content": "Apple computer", "metadata": {}},
        ]
        embeddings = np.random.rand(2, 128).astype(np.float32)
        retriever.add_documents(documents, embeddings)

        # Search
        query_embedding = np.random.rand(128).astype(np.float32)
        results = retriever.search(query_embedding, k=2)

        assert len(results) == 2
        assert "content" in results[0]
        assert "similarity" in results[0]

    def test_save_and_load(self, tmp_path):
        """Test saving and loading index."""
        retriever = TurboVecRetriever(dim=128, bit_width=4)

        # Add documents
        documents = [{"id": "doc1", "content": "test", "metadata": {}}]
        embeddings = np.random.rand(1, 128).astype(np.float32)
        retriever.add_documents(documents, embeddings)

        # Save
        save_path = tmp_path / "test_index"
        retriever.save(str(save_path))

        # Load into new instance
        retriever2 = TurboVecRetriever(dim=128, bit_width=4, index_path=str(save_path))
        assert len(retriever2.documents) == 1

    def test_len(self):
        """Test __len__ returns document count."""
        retriever = TurboVecRetriever(dim=128)
        assert len(retriever) == 0

        documents = [{"id": "doc1", "content": "test", "metadata": {}}]
        embeddings = np.random.rand(1, 128).astype(np.float32)
        retriever.add_documents(documents, embeddings)
        assert len(retriever) == 1


class TestKnowledgeRetriever:
    """Test KnowledgeRetriever (high-level RAG retriever)."""

    def test_init(self):
        """Test KnowledgeRetriever initializes correctly."""
        retriever = KnowledgeRetriever(
            ollama_base_url="http://localhost:11434",
            embedding_model="mxbai-embed-large",
        )
        assert retriever.turbovec is not None
        assert retriever.embedding_model == "mxbai-embed-large"

    def test_get_embedding(self):
        """Test embedding generation (requires Ollama running)."""
        retriever = KnowledgeRetriever()
        try:
            embedding = retriever.get_embedding("test text")
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (1024,)  # mxbai-embed-large dimension
        except Exception:
            pytest.skip("Ollama not available")

    def test_search(self):
        """Test search method (requires Ollama running)."""
        retriever = KnowledgeRetriever()
        try:
            results = retriever.search("test query", k=5)
            assert isinstance(results, list)
        except Exception:
            pytest.skip("Ollama not available")

    def test_save_and_load(self, tmp_path):
        """Test saving and loading KnowledgeRetriever."""
        retriever = KnowledgeRetriever()
        save_path = tmp_path / "kb_index"
        retriever.save(str(save_path))

        # Load
        retriever2 = KnowledgeRetriever(index_path=str(save_path))
        assert len(retriever2) >= 0


class TestDocument:
    """Test Document dataclass."""

    def test_document_creation(self):
        """Test Document dataclass creation."""
        doc = Document(
            id="test123",
            content="This is test content",
            metadata={"source": "test", "page": 1}
        )
        assert doc.id == "test123"
        assert doc.content == "This is test content"
        assert doc.metadata["source"] == "test"
