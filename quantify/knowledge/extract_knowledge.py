#!/usr/bin/env python3
"""
Knowledge Extraction Pipeline for Trading Books
Extracts structured trading concepts from book text using Ollama LLM.

Usage:
    python scripts/extract_knowledge.py --book "Technical Analysis" --input chapter5.txt
    python scripts/extract_knowledge.py --batch --input-dir ./extracted_text/ --output ./data/concepts.json
"""

import argparse
import json
import time
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils import setup_logging, load_config, retry_with_backoff, ExtractionError


@dataclass
class TradingConcept:
    """Structured trading knowledge item."""
    concept_name: str
    category: str
    description: str
    interpretation: str
    conditions: str
    market_context: str
    timeframes: List[str]
    source_book: str
    source_chapter: str
    source_page: int
    confidence: str


EXTRACTION_PROMPT = """You are a financial knowledge engineer. Extract every distinct trading concept, pattern, indicator, or rule from the text.

For each concept, output JSON with:
- concept_name: Short label
- category: One of ["Technical Pattern", "Indicator", "Fundamental Metric", "Options Strategy", "Risk Rule", "Market Bias", "Candlestick Pattern"]
- description: Concise 1-2 sentence definition
- interpretation: What it signals (bullish/bearish/neutral/reversal/continuation)
- conditions: Precise criteria (e.g., "RSI < 30 and volume > 1.5x average")
- market_context: When it works best and when it fails
- timeframes: List of applicable periods (e.g., ["Daily", "Weekly"])
- confidence: Author's reliability (low/medium/high)

TEXT:
{chunk_text}

Output ONLY valid JSON with a "concepts" array. No commentary."""


class OllamaClient:
    """Simple Ollama API client."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3.5:cloud",
        logger=None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.logger = logger or setup_logging(name="ollama")

    def chat(self, prompt: str, temperature: float = 0.1, timeout: int = 300) -> str:
        """
        Send chat completion request to Ollama.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature (0.0-1.0)
            timeout: Request timeout in seconds

        Returns:
            Response text from the model
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 4096},
        }

        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers={"Content-Type": "application/json"})

        try:
            with urlopen(req, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                content = result.get("message", {}).get("content", "")
                self.logger.debug(f"Ollama response: {len(content)} chars")
                return content
        except HTTPError as e:
            self.logger.error(f"Ollama API error: {e.code} {e.reason}")
            raise ExtractionError(f"Ollama API error: {e.code} {e.reason}")
        except URLError as e:
            self.logger.error(f"Failed to connect to Ollama: {e.reason}")
            raise ExtractionError(f"Failed to connect to Ollama: {e.reason}")


class KnowledgeExtractor:
    """Extract structured trading knowledge from book text."""

    def __init__(self, config_path: str = None, logger=None):
        self.config = load_config(config_path)
        self.logger = logger or setup_logging(name="knowledge_extractor")
        self.ollama = OllamaClient(
            base_url=self.config.get("llm", {}).get("base_url", "http://localhost:11434"),
            model=self.config.get("llm", {}).get("model", "qwen3.5:cloud"),
            logger=self.logger,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("extraction", {}).get("chunk_size", 2000),
            chunk_overlap=self.config.get("extraction", {}).get("chunk_overlap", 200),
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.retry_attempts = self.config.get("extraction", {}).get("retry_attempts", 3)
        self.retry_delay = self.config.get("extraction", {}).get("retry_delay_seconds", 10)


    def chunk_text(self, text: str, book_title: str) -> List[Dict[str, str]]:
        """Split text into chunks with metadata."""
        chunks = self.text_splitter.split_text(text)
        return [{
            "content": chunk,
            "metadata": {"book": book_title, "chunk_id": hashlib.md5(chunk.encode()).hexdigest()[:8]}
        } for i, chunk in enumerate(chunks)]

    def extract_from_chunk(self, chunk: str, book_title: str, chapter: str = "") -> List[TradingConcept]:
        """
        Extract concepts from a single chunk with retry logic.

        Args:
            chunk: Text chunk to extract from
            book_title: Source book title
            chapter: Source chapter name

        Returns:
            List of extracted TradingConcept objects
        """
        prompt = EXTRACTION_PROMPT.format(chunk_text=chunk)

        def parse_response():
            response_text = self.ollama.chat(prompt)
            response_text = response_text.strip()

            # Handle markdown code blocks
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            if not response_text:
                raise json.JSONDecodeError("Empty response", "", 0)

            result = json.loads(response_text)
            concepts = []

            for c in result.get("concepts", []):
                concept = TradingConcept(
                    concept_name=c.get("concept_name", "Unknown"),
                    category=c.get("category", "Technical Pattern"),
                    description=c.get("description", ""),
                    interpretation=c.get("interpretation", "neutral"),
                    conditions=c.get("conditions", ""),
                    market_context=c.get("market_context", ""),
                    timeframes=c.get("timeframes", ["Daily"]),
                    source_book=book_title,
                    source_chapter=chapter,
                    source_page=0,
                    confidence=c.get("confidence", "medium"),
                )
                concepts.append(concept)

            if not concepts:
                raise ExtractionError("No concepts extracted from chunk")

            return concepts

        try:
            return retry_with_backoff(
                parse_response,
                max_attempts=self.retry_attempts,
                delay_seconds=self.retry_delay,
                exceptions=(json.JSONDecodeError, ExtractionError, Exception),
            )
        except ExtractionError:
            self.logger.debug(f"Failed to extract concepts after {self.retry_attempts} attempts")
            return []

    def extract_from_text(self, text: str, book_title: str, chapter: str = "") -> List[TradingConcept]:
        """
        Extract concepts from full text.

        Args:
            text: Full text to extract from
            book_title: Source book title
            chapter: Source chapter name

        Returns:
            List of all extracted TradingConcept objects
        """
        chunks = self.chunk_text(text, book_title)
        all_concepts = []

        self.logger.info(f"Processing {len(chunks)} chunks from '{book_title}'...")
        for i, chunk in enumerate(chunks):
            self.logger.debug(f"  Chunk {i + 1}/{len(chunks)}")
            concepts = self.extract_from_chunk(chunk["content"], book_title, chapter)
            all_concepts.extend(concepts)
            self.logger.debug(f"    Found {len(concepts)} concepts")

        self.logger.info(f"Extracted {len(all_concepts)} total concepts from '{book_title}'.")
        return all_concepts

    def extract_from_file(self, file_path: str, book_title: str) -> List[TradingConcept]:
        """
        Extract concepts from a file.

        Args:
            file_path: Path to text file
            book_title: Source book title

        Returns:
            List of extracted TradingConcept objects
        """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        chapter = Path(file_path).stem
        return self.extract_from_text(text, book_title, chapter)


def save_concepts(concepts: List[TradingConcept], output_path: str, logger=None) -> str:
    """
    Save concepts to JSON file.

    Args:
        concepts: List of TradingConcept objects
        output_path: Path to output JSON file
        logger: Optional logger instance

    Returns:
        Path to saved file
    """
    if logger is None:
        logger = setup_logging(name="save_concepts")

    data = [asdict(c) for c in concepts]
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(concepts)} concepts to {output_path}")
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(description="Extract trading knowledge from books using Ollama")
    parser.add_argument("--book", type=str, help="Book title")
    parser.add_argument("--input", type=str, help="Input text file")
    parser.add_argument("--output", type=str, default="./data/extracted_concepts.json", help="Output JSON file")
    parser.add_argument("--batch", action="store_true", help="Batch mode")
    parser.add_argument("--input-dir", type=str, help="Input directory for batch mode")
    parser.add_argument("--config", type=str, help="Path to config.json")
    parser.add_argument("--model", type=str, help="Override model from config")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    config = load_config(args.config)
    log_level = "DEBUG" if args.verbose else config.get("logging", {}).get("level", "INFO")
    logger = setup_logging(name="extract_knowledge", level=log_level)

    # Check Ollama connection
    try:
        client = OllamaClient(
            base_url=config.get("llm", {}).get("base_url", "http://localhost:11434"),
            model=config.get("llm", {}).get("model", "qwen3.5:cloud"),
            logger=logger,
        )
        client.chat("Hello")
        logger.info("Ollama connection successful")
    except Exception as e:
        logger.error(f"Cannot connect to Ollama: {e}")
        logger.error("Make sure Ollama is running at localhost:11434")
        return

    extractor = KnowledgeExtractor(config_path=args.config, logger=logger)
    if args.model:
        extractor.ollama.model = args.model
        logger.info(f"Using model: {args.model}")

    if args.batch and args.input_dir:
        input_path = Path(args.input_dir)
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        all_concepts = []
        for txt_file in input_path.glob("*.txt"):
            logger.info(f"\nProcessing {txt_file.name}...")
            concepts = extractor.extract_from_file(str(txt_file), txt_file.stem)
            all_concepts.extend(concepts)
            save_concepts(concepts, str(output_dir / f"{txt_file.stem}_concepts.json"), logger)

        save_concepts(all_concepts, str(output_dir / "all_concepts.json"), logger)

    elif args.book and args.input:
        concepts = extractor.extract_from_file(args.input, args.book)
        save_concepts(concepts, args.output, logger)

    else:
        print("Usage examples:")
        print("  python extract_knowledge.py --book 'Technical Analysis' --input chapter5.txt")
        print("  python extract_knowledge.py --batch --input-dir ./extracted_text/ --output ./data/concepts.json")


if __name__ == "__main__":
    main()
