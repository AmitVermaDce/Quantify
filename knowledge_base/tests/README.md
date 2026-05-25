# Knowledge Base Tests

This directory contains tests for the `knowledge_base` module.

## Running Tests

### Run all knowledge base tests
```bash
python -m pytest knowledge_base/tests/ -v -o testpaths=knowledge_base/tests -o addopts=""
```

### Run specific test file
```bash
# Test KnowledgeService
python -m pytest knowledge_base/tests/test_service.py -v -o addopts=""

# Test TurboVecRetriever
python -m pytest knowledge_base/tests/test_turbovec_retriever.py -v -o addopts=""

# Test build pipeline
python -m pytest knowledge_base/tests/test_build_knowledge_base.py -v -o addopts=""
```

### Run with coverage
```bash
python -m pytest knowledge_base/tests/ --cov=knowledge_base --cov-report=html -o addopts=""
```

### Run specific test
```bash
python -m pytest knowledge_base/tests/test_service.py::TestKnowledgeService::test_search -v -o addopts=""
```

## Test Files

| File | Description |
|------|-------------|
| `conftest.py` | Pytest fixtures and configuration |
| `test_service.py` | Tests for `KnowledgeService` class |
| `test_turbovec_retriever.py` | Tests for `TurboVecRetriever` and `KnowledgeRetriever` |
| `test_build_knowledge_base.py` | Tests for PDF extraction and chunking pipeline |

## Test Categories

### Unit Tests (`test_service.py`)
- `KnowledgeService` initialization and configuration
- Search functionality with filters (category, similarity)
- Context generation for prompts
- Knowledge injection into prompts
- Error handling

### Retriever Tests (`test_turbovec_retriever.py`)
- `TurboVecRetriever` core operations (add, search, save, load)
- `KnowledgeRetriever` high-level RAG operations
- `Document` dataclass

### Pipeline Tests (`test_build_knowledge_base.py`)
- PDF text extraction with PyMuPDF
- Semantic text chunking with LangChain
- Batch PDF loading from directories

## Fixtures

- `kb_dir`: Path to knowledge_base directory
- `sample_pdf_path`: Temporary PDF file for testing
- `sample_text`: Sample text for chunking tests
- `sample_documents`: Sample documents for indexing tests

## Requirements

Tests require:
- Ollama running with `mxbai-embed-large` model (for embedding tests)
- PyMuPDF installed
- TurboVec installed
- Existing knowledge base index at `knowledge_base/data/knowledge_base/`

Some tests will skip if Ollama is not available.
