#!/usr/bin/env python3
"""
Command-line interface for the Knowledge Base pipeline.
Orchestrates PDF extraction, knowledge extraction, and database seeding.

Usage:
    python scripts/cli.py pipeline --input-dir ../resources/books --output ./data
    python scripts/cli.py extract --book "Technical Analysis" --input chapter5.txt
    python scripts/cli.py seed --input ./data/concepts.json
    python scripts/cli.py status
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from utils import setup_logging, load_config, format_duration, ensure_directory, ExtractionError

# Import pipeline modules
from extract_pdf import extract_text_from_pdf, clean_text
from extract_knowledge import KnowledgeExtractor, save_concepts
from seed_knowledge import seed_concepts, get_db_connection


class PipelineOrchestrator:
    """Orchestrates the full knowledge extraction pipeline."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.logger = setup_logging(
            name="pipeline",
            level=self.config.get("logging", {}).get("level", "INFO"),
            log_file=self.config.get("logging", {}).get("file"),
        )
        self.pdf_extractor = None
        self.knowledge_extractor = None

    def run_full_pipeline(
        self,
        input_dir: str,
        output_dir: str,
        file_pattern: str = "*.pdf",
        skip_existing: bool = True,
    ) -> dict:
        """
        Run the complete pipeline: PDF → Text → Concepts → Database

        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory for output files
            file_pattern: Glob pattern for PDF files
            skip_existing: Skip files that already have extracted concepts

        Returns:
            Statistics dictionary with counts and timing
        """
        start_time = time.time()
        stats = {
            "started_at": datetime.now().isoformat(),
            "pdfs_processed": 0,
            "concepts_extracted": 0,
            "concepts_seeded": 0,
            "errors": [],
        }

        self.logger.info("=" * 60)
        self.logger.info("Starting Knowledge Extraction Pipeline")
        self.logger.info("=" * 60)
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Model: {self.config['llm']['model']}")
        self.logger.info(f"Chunk size: {self.config['extraction']['chunk_size']}")
        self.logger.info("")

        # Ensure output directories exist
        text_dir = ensure_directory(f"{output_dir}/extracted_text")
        concepts_dir = ensure_directory(f"{output_dir}/concepts")

        # Find all PDF files
        input_path = Path(input_dir)
        pdf_files = list(input_path.glob(file_pattern))

        if not pdf_files:
            self.logger.warning(f"No PDF files found matching '{file_pattern}' in {input_dir}")
            return stats

        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        self.logger.info("")

        # Initialize extractors
        self.knowledge_extractor = KnowledgeExtractor(config_path=self.config_path)

        # Process each PDF
        for i, pdf_file in enumerate(pdf_files, 1):
            book_name = pdf_file.stem
            text_output = text_dir / f"{book_name}.txt"
            concepts_output = concepts_dir / f"{book_name}_concepts.json"

            self.logger.info(f"[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
            self.logger.info("-" * 40)

            try:
                # Step 1: Extract text from PDF
                if skip_existing and text_output.exists():
                    self.logger.info(f"  Skipping text extraction (exists): {text_output.name}")
                else:
                    self._extract_pdf_text(str(pdf_file), str(text_output))

                # Step 2: Extract knowledge concepts
                if skip_existing and concepts_output.exists():
                    self.logger.info(f"  Skipping concept extraction (exists): {concepts_output.name}")
                else:
                    concepts = self._extract_concepts(str(text_output), book_name)
                    stats["concepts_extracted"] += len(concepts)
                    save_concepts(concepts, str(concepts_output))

                # Step 3: Seed to database
                # self.logger.info("  Seeding to database...")
                # seeded = seed_concepts(str(concepts_output), self.config)
                # stats["concepts_seeded"] += seeded

                stats["pdfs_processed"] += 1
                self.logger.info("")

            except Exception as e:
                self.logger.error(f"  ERROR: {type(e).__name__} - {e}")
                stats["errors"].append({"file": str(pdf_file), "error": str(e)})
                self.logger.info("")

        # Summary
        elapsed = time.time() - start_time
        stats["completed_at"] = datetime.now().isoformat()
        stats["elapsed_seconds"] = elapsed

        self.logger.info("=" * 60)
        self.logger.info("Pipeline Complete")
        self.logger.info("=" * 60)
        self.logger.info(f"PDFs processed: {stats['pdfs_processed']}/{len(pdf_files)}")
        self.logger.info(f"Concepts extracted: {stats['concepts_extracted']}")
        self.logger.info(f"Concepts seeded: {stats['concepts_seeded']}")
        self.logger.info(f"Errors: {len(stats['errors'])}")
        self.logger.info(f"Total time: {format_duration(elapsed)}")

        if stats["errors"]:
            self.logger.warning("Files with errors:")
            for err in stats["errors"]:
                self.logger.warning(f"  - {err['file']}: {err['error']}")

        return stats

    def _extract_pdf_text(self, pdf_path: str, output_path: str) -> None:
        """Extract text from a single PDF file."""
        self.logger.info(f"  Extracting text from PDF...")
        text = extract_text_from_pdf(pdf_path)
        cleaned = clean_text(text)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        self.logger.info(f"  Saved: {output_path} ({len(cleaned):,} chars)")

    def _extract_concepts(self, text_path: str, book_name: str):
        """Extract concepts from a text file."""
        self.logger.info(f"  Extracting concepts using LLM...")

        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        return self.knowledge_extractor.extract_from_text(text, book_name)


def cmd_pipeline(args):
    """Run the full pipeline command."""
    orchestrator = PipelineOrchestrator(config_path=args.config)
    stats = orchestrator.run_full_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_pattern=args.pattern,
        skip_existing=args.skip_existing,
    )
    sys.exit(0 if not stats["errors"] else 1)


def cmd_extract(args):
    """Run knowledge extraction on a single file."""
    config = load_config(args.config)
    logger = setup_logging(name="extract", level=config.get("logging", {}).get("level", "INFO"))

    extractor = KnowledgeExtractor(config_path=args.config)

    logger.info(f"Extracting concepts from: {args.input}")
    concepts = extractor.extract_from_file(args.input, args.book)

    save_concepts(concepts, args.output)
    logger.info(f"Extraction complete: {len(concepts)} concepts")


def cmd_seed(args):
    """Seed concepts to the database."""
    config = load_config(args.config)
    logger = setup_logging(name="seed", level=config.get("logging", {}).get("level", "INFO"))

    logger.info(f"Seeding concepts from: {args.input}")
    count = seed_concepts(args.input, config)
    logger.info(f"Seeded {count} concepts to database")


def cmd_status(args):
    """Check pipeline status and configuration."""
    config = load_config(args.config)
    logger = setup_logging(name="status", level="INFO", log_file=None)

    logger.info("Knowledge Base Configuration")
    logger.info("=" * 40)
    logger.info(f"LLM Model: {config['llm']['model']}")
    logger.info(f"LLM Base URL: {config['llm']['base_url']}")
    logger.info(f"Embedding Model: {config['llm']['embedding_model']}")
    logger.info(f"Embedding Dimension: {config['llm']['embedding_dimension']}")
    logger.info(f"Database URL: {config['database']['url']}")
    logger.info(f"Chunk Size: {config['extraction']['chunk_size']}")
    logger.info(f"Chunk Overlap: {config['extraction']['chunk_overlap']}")
    logger.info(f"Retry Attempts: {config['extraction']['retry_attempts']}")
    logger.info(f"Retry Delay: {config['extraction']['retry_delay_seconds']}s")

    # Check Ollama connection
    logger.info("")
    logger.info("Checking Ollama connection...")
    try:
        from extract_knowledge import OllamaClient
        client = OllamaClient(
            base_url=config["llm"]["base_url"],
            model=config["llm"]["model"],
        )
        client.chat("Hello")
        logger.info("  ✓ Ollama is reachable")
    except Exception as e:
        logger.error(f"  ✗ Ollama connection failed: {e}")

    # Check database connection
    logger.info("Checking database connection...")
    try:
        conn = get_db_connection(config)
        conn.close()
        logger.info("  ✓ Database is reachable")
    except Exception as e:
        logger.error(f"  ✗ Database connection failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Base Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s pipeline --input-dir ./books --output ./data
  %(prog)s extract --book "Technical Analysis" --input chapter5.txt --output concepts.json
  %(prog)s seed --input concepts.json
  %(prog)s status
        """,
    )

    parser.add_argument("--config", type=str, help="Path to config.json")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full extraction pipeline")
    pipeline_parser.add_argument("--input-dir", type=str, required=True, help="Directory with PDF files")
    pipeline_parser.add_argument("--output-dir", type=str, default="./data", help="Output directory")
    pipeline_parser.add_argument("--pattern", type=str, default="*.pdf", help="File pattern to match")
    pipeline_parser.add_argument("--skip-existing", action="store_true", help="Skip already processed files")
    pipeline_parser.set_defaults(func=cmd_pipeline)

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract concepts from a single file")
    extract_parser.add_argument("--book", type=str, required=True, help="Book title")
    extract_parser.add_argument("--input", type=str, required=True, help="Input text file")
    extract_parser.add_argument("--output", type=str, default="./data/concepts.json", help="Output JSON file")
    extract_parser.set_defaults(func=cmd_extract)

    # Seed command
    seed_parser = subparsers.add_parser("seed", help="Seed concepts to database")
    seed_parser.add_argument("--input", type=str, required=True, help="Input JSON file with concepts")
    seed_parser.set_defaults(func=cmd_seed)

    # Status command
    status_parser = subparsers.add_parser("status", help="Check configuration and connections")
    status_parser.set_defaults(func=cmd_status)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
