"""Pytest configuration and fixtures for knowledge_base tests."""

import pytest
from pathlib import Path
import sys

# Add knowledge_base to path for imports
KB_DIR = Path(__file__).parent.parent
if str(KB_DIR) not in sys.path:
    sys.path.insert(0, str(KB_DIR))


@pytest.fixture(scope="session")
def kb_dir():
    """Return the knowledge_base directory path."""
    return KB_DIR


@pytest.fixture(scope="session")
def sample_pdf_path(tmp_path_factory):
    """Create a sample PDF file for testing."""
    pdf_path = tmp_path_factory.mktemp("pdfs") / "sample.pdf"
    # Minimal valid PDF structure
    pdf_path.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 50 >>\nstream\nBT /F1 12 Tf 72 720 Td "
        b"(Sample PDF content for testing) Tj ET\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f\n0000000009 00000 n\n"
        b"0000000058 00000 n\n0000000115 00000 n\n0000000214 00000 n\n"
        b"0000000314 00000 n\n"
        b"trailer\n<< /Root 1 0 R /Size 6 >>\nstartxref\n394\n%%EOF\n"
    )
    return pdf_path


@pytest.fixture
def sample_text():
    """Return sample text for chunking tests."""
    return """
    Chapter 1: Introduction to Investing

    Value investing is an investment paradigm that derives from the ideas
    that Benjamin Graham and David Dodd began teaching at Columbia Business
    School in 1928 and subsequently developed in their 1934 text Security
    Analysis. Although value investing has taken many forms since its
    inception, it broadly involves buying securities that appear underpriced
    by some form of fundamental analysis.

    Chapter 2: Margin of Safety

    The concept of margin of safety is central to value investing. It refers
    to the difference between the intrinsic value of a security and its
    market price. This difference provides a cushion against errors in
    estimation or unforeseen market conditions.

    Chapter 3: Risk Management

    Risk management is the process of identification, analysis, and
    acceptance or mitigation of uncertainty in investment decisions.
    Essentially, risk management occurs anytime an investor or fund manager
    analyzes and attempts to reduce the potential risks of an investment.
    """


@pytest.fixture
def sample_documents():
    """Return sample documents for indexing tests."""
    return [
        {
            "id": "doc1",
            "content": "Value investing involves buying undervalued securities",
            "metadata": {"category": "Investing_Books", "book_title": "Test Book 1"}
        },
        {
            "id": "doc2",
            "content": "Risk management is crucial for long-term success",
            "metadata": {"category": "Investing_Books", "book_title": "Test Book 2"}
        },
        {
            "id": "doc3",
            "content": "Market psychology affects investment decisions",
            "metadata": {"category": "Psychological_Books", "book_title": "Test Book 3"}
        },
    ]
