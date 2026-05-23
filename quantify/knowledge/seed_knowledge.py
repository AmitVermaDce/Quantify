#!/usr/bin/env python3
"""
Seed Knowledge Database - Loads extracted concepts into PostgreSQL with embeddings.

Usage:
    python scripts/seed_knowledge.py --input ./data/extracted_concepts/all_concepts.json
    python scripts/seed_knowledge.py --sample  # Load sample data for testing
"""

import json
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
from urllib.request import urlopen, Request

import psycopg2
from psycopg2.extras import execute_values

from utils import setup_logging, load_config


def get_embedding(text: str, config: dict, logger=None) -> List[float]:
    """
    Generate embedding using Ollama.

    Args:
        text: Text to embed
        config: Configuration dictionary
        logger: Optional logger instance

    Returns:
        List of embedding floats
    """
    if logger is None:
        logger = setup_logging(name="embeddings")

    ollama_config = config.get("llm", {})
    url = f"{ollama_config.get('base_url', 'http://localhost:11434')}/api/embeddings"
    payload = {
        "model": ollama_config.get("embedding_model", "mxbai-embed-large"),
        "prompt": text,
    }

    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        with urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("embedding", [])
    except Exception as e:
        logger.error(f"Embedding failed for '{text[:50]}...': {e}")
        raise


def get_db_connection(config: dict, logger=None):
    """
    Create database connection.

    Args:
        config: Configuration dictionary
        logger: Optional logger instance

    Returns:
        psycopg2 connection
    """
    if logger is None:
        logger = setup_logging(name="database")

    db_url = config.get("database", {}).get("url", "postgresql://localhost:5432/trading_knowledge")
    logger.debug(f"Connecting to database: {db_url.split('@')[-1]}")
    return psycopg2.connect(db_url)


def seed_concepts(
    input_file: str,
    config: dict,
    batch_size: int = 50,
    logger=None,
) -> int:
    """
    Load concepts from JSON into PostgreSQL.

    Args:
        input_file: Path to JSON file with concepts
        config: Configuration dictionary
        batch_size: Number of concepts to insert per batch
        logger: Optional logger instance

    Returns:
        Number of concepts inserted
    """
    if logger is None:
        logger = setup_logging(name="seed")

    with open(input_file, "r", encoding="utf-8") as f:
        concepts = json.load(f)

    logger.info(f"Loaded {len(concepts)} concepts from {input_file}")

    conn = get_db_connection(config, logger)
    cursor = conn.cursor()
    dim = config.get("llm", {}).get("embedding_dimension", 1024)

    insert_data = []
    inserted_count = 0

    logger.info(f"Generating embeddings and preparing {len(concepts)} concepts...")
    for i, concept in enumerate(concepts):
        embedding_text = f"{concept.get('concept_name', '')} {concept.get('description', '')} {concept.get('conditions', '')}"

        try:
            embedding = get_embedding(embedding_text, config, logger)
        except Exception as e:
            logger.warning(f"Embedding failed for '{concept.get('concept_name')}': {e}")
            embedding = [0.0] * dim

        timeframes = concept.get("timeframes", [])
        if isinstance(timeframes, str):
            try:
                timeframes = json.loads(timeframes)
            except json.JSONDecodeError:
                timeframes = [timeframes]
        if not isinstance(timeframes, list):
            timeframes = [str(timeframes)]

        insert_data.append(
            (
                concept.get("concept_name", "Unknown"),
                concept.get("category", "Technical Pattern"),
                concept.get("description", ""),
                concept.get("interpretation", "neutral"),
                concept.get("conditions", ""),
                concept.get("market_context", ""),
                timeframes,
                concept.get("source_book", ""),
                concept.get("source_chapter", ""),
                concept.get("source_page", 0),
                concept.get("confidence", "medium"),
                embedding,
            )
        )

        if (i + 1) % batch_size == 0 or i == len(concepts) - 1:
            logger.info(f"  Inserting batch {i // batch_size + 1}... ({i + 1}/{len(concepts)} concepts)")
            insert_sql = """
                INSERT INTO concepts (
                    concept_name, category, description, interpretation, conditions,
                    market_context, timeframes, source_book, source_chapter, source_page,
                    confidence, embedding
                ) VALUES %s
                RETURNING id, concept_name
            """
            results = execute_values(cursor, insert_sql, insert_data, fetch=True)
            conn.commit()
            inserted_count += len(results)
            insert_data = []

        if (i + 1) % 10 == 0:
            logger.debug(f"  Processed {i + 1}/{len(concepts)} concepts...")

    logger.info(f"Successfully inserted {inserted_count} concepts")

    cursor.close()
    conn.close()
    return inserted_count


def seed_sample_data(config: dict, logger=None) -> int:
    """
    Insert sample concepts for testing.

    Args:
        config: Configuration dictionary
        logger: Optional logger instance

    Returns:
        Number of concepts inserted
    """
    if logger is None:
        logger = setup_logging(name="seed_sample")

    sample_concepts = [
        {
            "concept_name": "Bullish Engulfing Pattern",
            "category": "Candlestick Pattern",
            "description": "A two-candle reversal pattern where a small bearish candle is completely engulfed by a larger bullish candle.",
            "interpretation": "bullish reversal",
            "conditions": "Occurs in a downtrend; second candle opens below prior close and closes above prior open; volume increases.",
            "market_context": "More reliable at support levels or after panic selling.",
            "timeframes": ["Daily", "Weekly"],
            "source_book": "Japanese Candlestick Charting Techniques",
            "source_chapter": "Chapter 5",
            "source_page": 42,
            "confidence": "high when confirmed by volume",
        },
        {
            "concept_name": "RSI Bullish Divergence",
            "category": "Technical Pattern",
            "description": "Price makes a lower low while RSI makes a higher low, signaling weakening selling momentum.",
            "interpretation": "bullish reversal warning",
            "conditions": "RSI below 30; divergence spans 5-10 bars; volume declines into low then expands.",
            "market_context": "Most reliable after extended trends. Requires price action confirmation.",
            "timeframes": ["Daily", "Weekly", "4H"],
            "source_book": "Technical Analysis of the Financial Markets",
            "source_chapter": "Chapter 10",
            "source_page": 189,
            "confidence": "medium",
        },
        {
            "concept_name": "Morning Star Pattern",
            "category": "Candlestick Pattern",
            "description": "A three-candle bullish reversal pattern with a long bearish candle, small-bodied star, and long bullish candle.",
            "interpretation": "bullish reversal",
            "conditions": "Downtrend preceding; star gaps below first candle; third candle closes 50%+ into first candle's body.",
            "market_context": "One of the most reliable reversal patterns on weekly timeframes.",
            "timeframes": ["Daily", "Weekly"],
            "source_book": "Japanese Candlestick Charting Techniques",
            "source_chapter": "Chapter 7",
            "source_page": 55,
            "confidence": "high",
        },
    ]

    conn = get_db_connection(config, logger)
    cursor = conn.cursor()
    dim = config.get("llm", {}).get("embedding_dimension", 1024)

    insert_data = []
    for concept in sample_concepts:
        embedding_text = f"{concept['concept_name']} {concept['description']} {concept['conditions']}"
        try:
            embedding = get_embedding(embedding_text, config, logger)
        except Exception:
            logger.warning(f"Using zero embedding for {concept['concept_name']}")
            embedding = [0.0] * dim

        insert_data.append(
            (
                concept["concept_name"],
                concept["category"],
                concept["description"],
                concept["interpretation"],
                concept["conditions"],
                concept["market_context"],
                concept["timeframes"],
                concept["source_book"],
                concept["source_chapter"],
                concept["source_page"],
                concept["confidence"],
                embedding,
            )
        )

    insert_sql = """
        INSERT INTO concepts (
            concept_name, category, description, interpretation, conditions,
            market_context, timeframes, source_book, source_chapter, source_page,
            confidence, embedding
        ) VALUES %s
        RETURNING id, concept_name
    """

    results = execute_values(cursor, insert_sql, insert_data, fetch=True)
    conn.commit()

    logger.info(f"Successfully inserted {len(results)} sample concepts:")
    for row in results:
        logger.info(f"  - {row[1]} (id={row[0]})")

    cursor.close()
    conn.close()
    return len(results)


def main():
    parser = argparse.ArgumentParser(description="Seed knowledge database")
    parser.add_argument("--input", type=str, help="Input JSON file from extraction")
    parser.add_argument("--sample", action="store_true", help="Load sample data")
    parser.add_argument("--config", type=str, help="Path to config.json")
    parser.add_argument("--db", type=str, help="Database URL (overrides config)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for insertion")

    args = parser.parse_args()

    config = load_config(args.config)
    log_level = "DEBUG" if args.verbose else config.get("logging", {}).get("level", "INFO")
    logger = setup_logging(name="seed_knowledge", level=log_level)

    if args.db:
        config["database"] = config.get("database", {})
        config["database"]["url"] = args.db

    if args.sample:
        seed_sample_data(config, logger)
    elif args.input:
        seed_concepts(args.input, config, batch_size=args.batch_size, logger=logger)
    else:
        print("Usage:")
        print("  python seed_knowledge.py --sample  # Load sample concepts")
        print("  python seed_knowledge.py --input ./data/extracted_concepts/all_concepts.json")


if __name__ == "__main__":
    main()
