#!/usr/bin/env python3
"""
PDF Text Extractor for Trading Books
Extracts text from PDF files for knowledge extraction pipeline.

Usage:
    python scripts/extract_pdf.py --input-dir ./data/Trading_Books --output-dir ./data/extracted_text
"""

import os
import re
import argparse
from pathlib import Path
from typing import Optional

try:
    import pymupdf
except ImportError:
    os.system("pip3 install pymupdf")
    import pymupdf

from utils import setup_logging, load_config


def extract_text_from_pdf(pdf_path: str, logger=None) -> str:
    """
    Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file
        logger: Optional logger instance

    Returns:
        Extracted text content
    """
    if logger is None:
        logger = setup_logging(name="pdf_extractor")

    doc = pymupdf.open(pdf_path)
    text_content = []

    logger.debug(f"Extracting {len(doc)} pages from {Path(pdf_path).name}...")
    for i, page in enumerate(doc):
        text_content.append(f"\n--- PAGE {i + 1}/{len(doc)} ---\n")
        text_content.append(page.get_text("text"))
        if (i + 1) % 50 == 0:
            logger.debug(f"  Processed {i + 1}/{len(doc)} pages...")

    doc.close()
    return "".join(text_content)


def clean_text(text: str) -> str:
    """
    Clean up extracted text by removing artifacts.

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    logger=None,
) -> Optional[dict]:
    """
    Process a single PDF file.

    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory for text file
        logger: Optional logger instance

    Returns:
        Statistics dictionary or None on failure
    """
    if logger is None:
        logger = setup_logging(name="pdf_processor")

    try:
        logger.info(f"Processing {pdf_path.name}...")
        text = extract_text_from_pdf(str(pdf_path), logger)
        cleaned = clean_text(text)

        output_file = output_dir / f"{pdf_path.stem}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Source: {pdf_path}\n\n")
            f.write(cleaned)

        stats = {
            "file": pdf_path.name,
            "pages": len(pymupdf.open(str(pdf_path))),
            "characters": len(cleaned),
            "output": str(output_file),
        }

        logger.info(f"  Saved: {output_file.name} ({len(cleaned) / 1024 / 1024:.1f} MB)")
        return stats

    except Exception as e:
        logger.error(f"  ERROR: {type(e).__name__} - {e}")
        return {"file": pdf_path.name, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDF trading books")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory with PDFs")
    parser.add_argument("--output-dir", type=str, default="./data/extracted_text", help="Output directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of PDFs (for testing)")
    parser.add_argument("--config", type=str, help="Path to config.json")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    config = load_config(args.config)
    log_level = "DEBUG" if args.verbose else config.get("logging", {}).get("level", "INFO")
    logger = setup_logging(name="extract_pdf", level=log_level)

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {args.input_dir}")
        return

    if args.limit > 0:
        pdf_files = pdf_files[: args.limit]
        logger.info(f"Processing first {len(pdf_files)} files (limit={args.limit})")

    logger.info(f"Found {len(pdf_files)} PDF files")
    logger.info("")

    stats = {"processed": 0, "errors": 0, "total_pages": 0, "total_chars": 0}

    for i, pdf_file in enumerate(pdf_files, 1):
        logger.info(f"[{i}/{len(pdf_files)}] {pdf_file.name}")
        result = process_pdf(pdf_file, output_path, logger)

        if result and "error" not in result:
            stats["processed"] += 1
            stats["total_pages"] += result.get("pages", 0)
            stats["total_chars"] += result.get("characters", 0)
        else:
            stats["errors"] += 1

        logger.info("")

    logger.info("=" * 40)
    logger.info("Extraction Complete")
    logger.info(f"PDFs processed: {stats['processed']}/{len(pdf_files)}")
    logger.info(f"Total pages: {stats['total_pages']}")
    logger.info(f"Total characters: {stats['total_chars']:,}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info("")
    logger.info(f"Next: python scripts/extract_knowledge.py --batch --input-dir {output_path}")


if __name__ == "__main__":
    main()
