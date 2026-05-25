#!/usr/bin/env python3
"""
Load Extracted Concepts into TurboVec Index

Reads extracted concepts from JSON files and loads them into TurboVec
for fast semantic search.

Usage:
    python load_turbovec_index.py --input ./data/extracted_concepts/all_concepts.json
    python load_turbovec_index.py --input-dir ./data/extracted_concepts/  # Load all JSON files
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import setup_logging, load_config
from turbovec_retriever import KnowledgeRetriever


def load_concepts_from_file(file_path: Path, logger) -> List[Dict[str, Any]]:
    """Load concepts from a JSON file."""
    logger.info(f"Loading concepts from {file_path}...")

    with open(file_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)

    logger.info(f"  Found {len(concepts)} concepts")
    return concepts


def load_all_concepts(input_dir: Path, logger) -> List[Dict[str, Any]]:
    """Load concepts from all JSON files in a directory."""
    all_concepts = []
    json_files = list(input_dir.glob("*_concepts.json"))

    if not json_files:
        logger.warning(f"No *_concepts.json files found in {input_dir}")
        return all_concepts

    logger.info(f"Found {len(json_files)} concept files")

    for json_file in json_files:
        concepts = load_concepts_from_file(json_file, logger)
        all_concepts.extend(concepts)

    return all_concepts


def main():
    parser = argparse.ArgumentParser(
        description="Load concepts into TurboVec index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python load_turbovec_index.py --input ./data/extracted_concepts/all_concepts.json
  python load_turbovec_index.py --input-dir ./data/extracted_concepts/
        """
    )

    parser.add_argument("--input", type=str, help="Input JSON file with concepts")
    parser.add_argument("--input-dir", type=str, help="Input directory with JSON files")
    parser.add_argument("--output", type=str, default="./data/turbovec_index",
                        help="Output directory for TurboVec index")
    parser.add_argument("--config", type=str, help="Path to config.json")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    config = load_config(args.config)
    log_level = "DEBUG" if args.verbose else config.get("logging", {}).get("level", "INFO")
    logger = setup_logging(name="load_turbovec", level=log_level)

    llm_config = config.get("llm", {})

    # Load concepts
    all_concepts = []

    if args.input:
        all_concepts = load_concepts_from_file(Path(args.input), logger)
    elif args.input_dir:
        all_concepts = load_all_concepts(Path(args.input_dir), logger)
    else:
        parser.print_help()
        print("\nError: Please specify --input or --input-dir")
        sys.exit(1)

    if not all_concepts:
        logger.error("No concepts loaded. Exiting.")
        sys.exit(1)

    logger.info(f"Total concepts to index: {len(all_concepts)}")

    # Initialize retriever
    logger.info("Initializing TurboVec retriever...")
    retriever = KnowledgeRetriever(
        ollama_base_url=llm_config.get("base_url", "http://localhost:11434"),
        embedding_model=llm_config.get("embedding_model", "mxbai-embed-large"),
        index_path=None,  # Start fresh
    )

    # Add concepts to index
    logger.info("Adding concepts to TurboVec index...")
    added = retriever.add_knowledge_base(all_concepts, batch_size=50)

    # Save index
    output_path = Path(args.output)
    logger.info(f"Saving index to {output_path}...")
    retriever.save(str(output_path))

    logger.info("")
    logger.info("=" * 50)
    logger.info("TurboVec Index Creation Complete")
    logger.info("=" * 50)
    logger.info(f"Concepts added: {added}")
    logger.info(f"Total in index: {len(retriever)}")
    logger.info(f"Index path: {output_path}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Start the API: python knowledge_api_turbovec.py --port 3001")
    logger.info("  2. Search knowledge:")
    logger.info(f"     curl -X POST http://localhost:3001/knowledge/search \\")
    logger.info(f"       -H 'Content-Type: application/json' \\")
    logger.info(f"       -d '{{\"query\": \"bullish reversal patterns\", \"limit\": 5}}'")
    logger.info("")


if __name__ == "__main__":
    main()
