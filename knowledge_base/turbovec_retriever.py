#!/usr/bin/env python3
"""
TurboVec Retriever for RAG-based Knowledge Search

Uses TurboVec's quantized vector index for fast semantic search.

Features:
- 4-bit quantization for 4x memory reduction vs float32
- Fast approximate nearest neighbor search
- Simple add/search API

Usage:
    from turbovec_retriever import TurboVecRetriever

    retriever = TurboVecRetriever(dim=1024)
    retriever.add_documents(documents, embeddings)
    results = retriever.search(query_embedding, k=10)
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from turbovec import TurboQuantIndex
except ImportError:
    os.system("pip install turbovec")
    from turbovec import TurboQuantIndex

import numpy as np


@dataclass
class Document:
    """A document with its content and metadata."""
    id: str
    content: str
    metadata: Dict[str, Any]


class TurboVecRetriever:
    """
    RAG retriever using TurboVec for fast semantic search.
    """

    def __init__(
        self,
        dim: int = 1024,
        bit_width: int = 4,
        index_path: Optional[str] = None,
    ):
        self.dim = dim
        self.bit_width = bit_width
        self.index_path = index_path

        # Initialize TurboVec index
        self.index = TurboQuantIndex(dim=dim, bit_width=bit_width)

        # Document storage (maps internal index -> document)
        self.documents: List[Document] = []

        # Load existing index if path provided
        if index_path and os.path.exists(index_path):
            self.load(index_path)

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> int:
        """
        Add documents to the index.

        Args:
            documents: List of dicts with 'id', 'content', 'metadata'
            embeddings: Numpy array of shape (n_docs, dim)

        Returns:
            Number of documents added
        """
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have same length")

        added = 0
        embeddings_to_add = []

        for i, doc in enumerate(documents):
            doc_id = doc.get("id", f"doc_{hashlib.md5(doc.get('content', '').encode()).hexdigest()[:12]}")

            # Create document
            document = Document(
                id=doc_id,
                content=doc.get("content", ""),
                metadata=doc.get("metadata", {}),
            )

            embeddings_to_add.append(embeddings[i].astype(np.float32))
            self.documents.append(document)
            added += 1

        # Add all embeddings at once
        if embeddings_to_add:
            embeddings_array = np.array(embeddings_to_add, dtype=np.float32)
            self.index.add(embeddings_array)

        return added

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector of shape (dim,) or (1, dim)
            k: Number of results to return

        Returns:
            List of results with content, metadata, and similarity score
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search - returns (scores, indices)
        scores, indices = self.index.search(query_embedding, k=k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue

            doc = self.documents[idx]
            results.append({
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
                "similarity": float(score),
            })

        return results

    def save(self, path: str) -> None:
        """Save index and documents to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save index
        self.index.write(str(path / "index.tq"))

        # Save documents metadata
        docs_data = [
            {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
            }
            for doc in self.documents
        ]
        with open(path / "documents.json", "w", encoding="utf-8") as f:
            json.dump(docs_data, f, indent=2)

        # Save config
        config = {
            "dim": self.dim,
            "bit_width": self.bit_width,
            "document_count": len(self.documents),
        }
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def load(self, path: str) -> int:
        """Load index and documents from disk."""
        path = Path(path)

        # Load index using TurboVec's load method
        index_path = str(path / "index.tq")
        self.index = TurboQuantIndex.load(index_path)

        # Load documents metadata
        with open(path / "documents.json", "r", encoding="utf-8") as f:
            docs_data = json.load(f)

        # Reconstruct documents
        self.documents = [
            Document(
                id=doc["id"],
                content=doc["content"],
                metadata=doc.get("metadata", {}),
            )
            for doc in docs_data
        ]

        return len(self.documents)

    def __len__(self) -> int:
        """Return number of documents in index."""
        return len(self.documents)


class KnowledgeRetriever:
    """
    High-level RAG retriever for trading knowledge base.
    Integrates with Ollama for embeddings and TurboVec for search.
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        embedding_model: str = "mxbai-embed-large",
        index_path: Optional[str] = None,
    ):
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.embedding_model = embedding_model

        # Get embedding dimension from model
        dim_map = {
            "mxbai-embed-large": 1024,
            "nomic-embed-text": 768,
            "all-minilm": 384,
        }
        dim = dim_map.get(embedding_model, 1024)

        self.turbovec = TurboVecRetriever(
            dim=dim,
            bit_width=4,
            index_path=index_path,
        )

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using Ollama."""
        import json
        from urllib.request import urlopen, Request

        url = f"{self.ollama_base_url}/api/embeddings"
        payload = {
            "model": self.embedding_model,
            "prompt": text,
        }

        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers={"Content-Type": "application/json"})

        with urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
            embedding = result.get("embedding", [])
            if not embedding:
                raise ValueError("Empty embedding from Ollama")
            return np.array(embedding, dtype=np.float32)

    def add_knowledge_base(
        self,
        concepts: List[Dict[str, Any]],
        batch_size: int = 50,
    ) -> int:
        """Add knowledge base concepts to the index."""
        import logging
        logger = logging.getLogger("knowledge_retriever")

        # Prepare documents
        documents = []
        texts_to_embed = []

        for concept in concepts:
            text = " ".join([
                concept.get("concept_name", ""),
                concept.get("description", ""),
                concept.get("conditions", ""),
                concept.get("market_context", ""),
            ])

            documents.append({
                "id": concept.get("id", hashlib.md5(text.encode()).hexdigest()[:12]),
                "content": text,
                "metadata": {
                    "concept_name": concept.get("concept_name", ""),
                    "category": concept.get("category", ""),
                    "source_book": concept.get("source_book", ""),
                    "interpretation": concept.get("interpretation", ""),
                },
            })
            texts_to_embed.append(text)

        # Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(texts_to_embed)} concepts...")
        embeddings = []

        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i + batch_size]
            batch_embeddings = []

            for text in batch:
                try:
                    emb = self.get_embedding(text)
                    batch_embeddings.append(emb)
                except Exception as e:
                    logger.warning(f"Embedding failed: {e}")
                    batch_embeddings.append(np.zeros(self.turbovec.dim))

            embeddings.extend(batch_embeddings)
            logger.debug(f"  Processed {min(i + batch_size, len(texts_to_embed))}/{len(texts_to_embed)}")

        embeddings = np.array(embeddings)

        # Add to index
        added = self.turbovec.add_documents(documents, embeddings)
        logger.info(f"Added {added} concepts to TurboVec index")

        return added

    def search(
        self,
        query: str,
        k: int = 10,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search knowledge base."""
        # Generate query embedding
        query_embedding = self.get_embedding(query)

        # Search
        results = self.turbovec.search(query_embedding, k=k)

        # Filter by category if specified
        if category:
            results = [
                r for r in results
                if r["metadata"].get("category") == category
            ]

        return results

    def save(self, path: str) -> None:
        """Save index to disk."""
        self.turbovec.save(path)

    def load(self, path: str) -> int:
        """Load index from disk."""
        return self.turbovec.load(path)

    def __len__(self) -> int:
        """Return number of concepts in index."""
        return len(self.turbovec)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    retriever = KnowledgeRetriever(
        ollama_base_url="http://localhost:11434",
        embedding_model="mxbai-embed-large",
        index_path="./data/turbovec_index",
    )

    # Sample concepts
    concepts = [
        {
            "concept_name": "Bullish Engulfing",
            "description": "A two-candle reversal pattern",
            "conditions": "Downtrend, second candle engulfs first",
            "market_context": "More reliable at support levels",
            "category": "Candlestick Pattern",
            "source_book": "Japanese Candlesticks",
        },
        {
            "concept_name": "RSI Divergence",
            "description": "Price makes lower low, RSI makes higher low",
            "conditions": "RSI below 30, divergence spans 5-10 bars",
            "market_context": "Most reliable after extended trends",
            "category": "Technical Pattern",
            "source_book": "Technical Analysis",
        },
    ]

    retriever.add_knowledge_base(concepts)
    results = retriever.search("bullish reversal patterns", k=5)

    print("\nSearch Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['metadata']['concept_name']}")
        print(f"   Category: {result['metadata']['category']}")
        print(f"   Similarity: {result['similarity']:.4f}")

    retriever.save("./data/turbovec_index")
    print(f"\nSaved {len(retriever)} concepts to TurboVec index")
