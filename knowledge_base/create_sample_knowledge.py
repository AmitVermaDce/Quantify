#!/usr/bin/env python3
"""
Create Sample Knowledge Base from PDF Text

Creates initial knowledge base by:
1. Chunking extracted PDF texts
2. Creating document embeddings with Ollama
3. Storing in TurboVec for semantic search

This is a simpler alternative to full LLM concept extraction.
Each chunk becomes a searchable document.

Usage:
    python create_sample_knowledge.py --input-dir ./data/extracted_text/
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import setup_logging, load_config
from turbovec_retriever import KnowledgeRetriever


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('. ')
            if last_period > chunk_size // 2:
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


def load_text_files(input_dir: Path, logger) -> List[Dict[str, Any]]:
    """Load all extracted text files."""
    documents = []
    txt_files = list(input_dir.glob("*.txt"))

    if not txt_files:
        logger.warning(f"No .txt files found in {input_dir}")
        return documents

    logger.info(f"Found {len(txt_files)} text files")

    for txt_file in txt_files:
        logger.info(f"Loading {txt_file.name}...")

        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract book title from first line
        first_line = content.split("\n")[1] if "\n" in content else txt_file.stem
        book_title = first_line.replace("Book:", "").strip() if "Book:" in first_line else txt_file.stem

        # Remove "Source:" line
        lines = content.split("\n")
        if lines[0].startswith("Source:"):
            content = "\n".join(lines[2:])

        documents.append({
            "file": str(txt_file),
            "book_title": book_title,
            "content": content,
        })

        logger.info(f"  {len(content):,} characters")

    return documents


def main():
    parser = argparse.ArgumentParser(
        description="Create knowledge base from PDF text chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_sample_knowledge.py --input-dir ./data/extracted_text/
  python create_sample_knowledge.py --input-dir ./data/extracted_text/ --chunk-size 3000
        """
    )

    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory with extracted text files")
    parser.add_argument("--output", type=str, default="./data/turbovec_index",
                        help="Output directory for TurboVec index")
    parser.add_argument("--chunk-size", type=int, default=2000,
                        help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=200,
                        help="Chunk overlap in characters")
    parser.add_argument("--config", type=str, help="Path to config.json")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    config = load_config(args.config)
    log_level = "DEBUG" if args.verbose else config.get("logging", {}).get("level", "INFO")
    logger = setup_logging(name="create_sample_knowledge", level=log_level)

    llm_config = config.get("llm", {})

    # Load text files
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        sys.exit(1)

    text_documents = load_text_files(input_path, logger)

    if not text_documents:
        logger.error("No documents loaded")
        sys.exit(1)

    # Initialize retriever
    logger.info("Initializing TurboVec retriever...")
    retriever = KnowledgeRetriever(
        ollama_base_url=llm_config.get("base_url", "http://localhost:11434"),
        embedding_model=llm_config.get("embedding_model", "mxbai-embed-large"),
        index_path=None,
    )

    # Process documents
    logger.info("")
    logger.info("Creating knowledge base from text chunks...")

    all_chunks = []
    total_chunks = 0

    for doc in text_documents:
        chunks = chunk_text(doc["content"], args.chunk_size, args.chunk_overlap)

        for chunk in chunks:
            if len(chunk) < 100:  # Skip very small chunks
                continue

            chunk_id = hashlib.md5(chunk.encode()).hexdigest()[:12]
            all_chunks.append({
                "id": chunk_id,
                "content": chunk,
                "metadata": {
                    "book_title": doc["book_title"],
                    "source_file": doc["file"],
                    "chunk_size": len(chunk),
                },
            })
            total_chunks += 1

    logger.info(f"Created {total_chunks} chunks from {len(text_documents)} documents")

    # Generate embeddings and add to index
    logger.info("")
    logger.info("Generating embeddings and adding to TurboVec index...")

    documents_for_index = []
    texts_to_embed = []

    for chunk in all_chunks:
        documents_for_index.append({
            "id": chunk["id"],
            "content": chunk["content"],
            "metadata": chunk["metadata"],
        })
        texts_to_embed.append(chunk["content"])

    # Generate embeddings in batches
    embeddings = []
    batch_size = 20

    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i:i + batch_size]
        batch_embeddings = []

        for text in batch:
            try:
                emb = retriever.get_embedding(text)
                batch_embeddings.append(emb)
            except Exception as e:
                logger.warning(f"Embedding failed: {e}")
                batch_embeddings.append(np.zeros(retriever.turbovec.dim))

        embeddings.extend(batch_embeddings)
        logger.debug(f"  Processed {min(i + batch_size, len(texts_to_embed))}/{len(texts_to_embed)}")

    embeddings = np.array(embeddings)

    # Add to index
    added = retriever.turbovec.add_documents(documents_for_index, embeddings)

    # Save index
    output_path = Path(args.output)
    logger.info(f"Saving index to {output_path}...")
    retriever.save(str(output_path))

    logger.info("")
    logger.info("=" * 50)
    logger.info("Knowledge Base Creation Complete")
    logger.info("=" * 50)
    logger.info(f"Documents processed: {len(text_documents)}")
    logger.info(f"Chunks created: {total_chunks}")
    logger.info(f"Chunks indexed: {added}")
    logger.info(f"Index path: {output_path}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Start the API: python knowledge_api_turbovec.py --port 3001")
    logger.info("  2. Search knowledge:")
    logger.info(f"     curl -X POST http://localhost:3001/knowledge/search \\")
    logger.info(f"       -H 'Content-Type: application/json' \\")
    logger.info(f"       -d '{{\"query\": \"value investing principles\", \"limit\": 5}}'")
    logger.info("")


if __name__ == "__main__":
    main()
