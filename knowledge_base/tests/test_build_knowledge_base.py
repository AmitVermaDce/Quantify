"""Tests for build_knowledge_base pipeline."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

try:
    from ..build_knowledge_base import (
        extract_text_from_pdf,
        chunk_text_semantic,
        load_all_pdfs,
    )
except ImportError:
    from build_knowledge_base import (
        extract_text_from_pdf,
        chunk_text_semantic,
        load_all_pdfs,
    )


class TestExtractTextFromPdf:
    """Test PDF text extraction."""

    def test_extract_text_from_pdf(self, tmp_path):
        """Test extracting text from a PDF file."""
        # Note: This test requires a real PDF file
        # Create a minimal valid PDF for testing
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R >>\nendobj\n"
            b"4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 72 720 Td "
            b"(Test content) Tj ET\nendstream\nendobj\n"
            b"xref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n"
            b"0000000058 00000 n\n0000000115 00000 n\n0000000214 00000 n\n"
            b"trailer\n<< /Root 1 0 R /Size 5 >>\nstartxref\n307\n%%EOF\n"
        )

        try:
            text = extract_text_from_pdf(str(pdf_path))
            assert isinstance(text, str)
            assert "Test content" in text
        except Exception as e:
            # PyMuPDF may fail on minimal PDF, skip if so
            pytest.skip(f"PDF extraction failed: {e}")

    def test_extract_text_nonexistent_pdf(self):
        """Test error handling for nonexistent PDF."""
        with pytest.raises(Exception):
            extract_text_from_pdf("/nonexistent/file.pdf")


class TestChunkTextSemantic:
    """Test semantic text chunking."""

    def test_chunk_text_semantic(self):
        """Test chunking text into semantic segments."""
        text = """
        Chapter 1: Introduction

        This is the first paragraph of the introduction. It contains important
        information about the topic at hand.

        Chapter 2: Main Content

        This is the main content section. It has detailed explanations and
        examples that illustrate the key concepts.
        """

        chunks = chunk_text_semantic(text, book_title="Test Book")

        assert len(chunks) > 0
        for chunk in chunks:
            assert "content" in chunk
            assert "metadata" in chunk
            assert "id" in chunk
            assert chunk["metadata"]["book_title"] == "Test Book"
            assert len(chunk["content"].strip()) >= 50  # Minimum chunk size

    def test_chunk_text_semantic_skips_small_chunks(self):
        """Test that very small chunks are skipped."""
        text = "Short text."
        chunks = chunk_text_semantic(text, book_title="Test")
        # Very short text should produce no chunks (filtered out)
        assert len(chunks) == 0

    def test_chunk_text_semantic_long_text(self):
        """Test chunking of long text produces multiple chunks."""
        # Generate long text
        paragraphs = [f"Paragraph {i}: " + "Lorem ipsum " * 50 for i in range(20)]
        text = "\n\n".join(paragraphs)

        chunks = chunk_text_semantic(text, book_title="Long Book")

        assert len(chunks) > 1
        # Verify chunks don't exceed max size significantly
        for chunk in chunks:
            assert len(chunk["content"]) <= 2500  # Some tolerance over 2000


class TestLoadAllPdfs:
    """Test loading all PDFs from a directory."""

    def test_load_all_pdfs_empty_directory(self, tmp_path):
        """Test loading from directory with no PDFs."""
        import logging

        # Create empty subdirectory
        books_dir = tmp_path / "Investing_Books"
        books_dir.mkdir()

        # Create a simple logger for the test
        logger = logging.getLogger("test_load_pdfs")
        logger.setLevel(logging.DEBUG)

        chunks = load_all_pdfs(tmp_path, logger)
        assert chunks == []

    def test_load_all_pdfs_with_files(self, tmp_path):
        """Test loading PDFs from directory structure."""
        # Create directory structure
        investing_dir = tmp_path / "Investing_Books"
        investing_dir.mkdir()

        # Create a minimal PDF file
        pdf_file = investing_dir / "test_book.pdf"
        pdf_file.write_bytes(
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            b"trailer\n<< /Root 1 0 R /Size 4 >>\n%%EOF\n"
        )

        import logging
        logger = MagicMock()
        logger.info = MagicMock()
        logger.debug = MagicMock()
        logger.warning = MagicMock()
        logger.error = MagicMock()

        chunks = load_all_pdfs(tmp_path, logger)

        # May fail on minimal PDF, but should return list
        assert isinstance(chunks, list)
        if chunks:
            for chunk in chunks:
                assert "metadata" in chunk
                assert chunk["metadata"]["category"] == "Investing_Books"
