# Knowledge Base Pipeline for TradingAgents

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    market_data/                                  │
│  ├── Investing_Books/    (PDFs: Security Analysis, etc.)        │
│  ├── Psychological_Books/ (PDFs: Trading in the Zone, etc.)     │
│  └── Historical_Data/     (future: market data)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Step 1: Document Parsing (dots.mocr)               │
│  - Python 3.12 environment                                      │
│  - vLLM server for inference                                    │
│  - Multilingual OCR + Layout Analysis                           │
│  - Output: Markdown with structure preserved                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│           Step 2: Semantic Chunking                             │
│  - Intelligent splitting by section/topic                       │
│  - Preserves semantic coherence                                 │
│  - Metadata: book, chapter, section, page                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│          Step 3: Embedding + TurboVec Index                     │
│  - Ollama: mxbai-embed-large (1024 dim)                         │
│  - TurboVec: 4-bit quantization (4x memory efficiency)          │
│  - Persistent storage: data/turbovec_index/                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         Step 4: RAG Integration in TradingAgents                │
│  - Each agent queries knowledge base                            │
│  - Semantic search for relevant context                         │
│  - Knowledge-enhanced prompts                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Knowledge-Enhanced Trading Decisions               │
│  - Portfolio Manager: references value investing principles     │
│  - Risk Manager: cites risk management frameworks               │
│  - Analysts: use technical/fundamental patterns from books      │
└─────────────────────────────────────────────────────────────────┘
```

## Setup Instructions

### Prerequisites

1. **Ollama** (already running):
   ```bash
   ollama list  # Should show mxbai-embed-large
   ```

2. **Python 3.12** for dots.mocr:
   ```bash
   conda create -n dots_mocr python=3.12
   conda activate dots_mocr
   ```

### Step 1: Install dots.mocr

```bash
# Clone and install
git clone https://github.com/rednote-hilab/dots.ocr
cd dots.ocr
pip install -e .

# Download model weights
python3 tools/download_model.py
```

### Step 2: Start vLLM Server

```bash
# In dots.mocr environment
vllm serve rednote-hilab/dots.mocr --tensor-parallel-size 1
```

### Step 3: Parse PDFs with dots.mocr

```bash
cd quantify/knowledge
python dots_ocr_parser.py \
  --input-dir ../market_data/Investing_Books/ \
  --output-dir ./data/extracted_text/ \
  --mode vllm \
  --vllm-url http://localhost:8000/v1
```

### Step 4: Create TurboVec Index

```bash
python create_sample_knowledge.py \
  --input-dir ./data/extracted_text/ \
  --output ./data/turbovec_index
```

### Step 5: Start Knowledge API

```bash
python knowledge_api_turbovec.py --port 3001
```

### Step 6: Query Knowledge

```bash
curl -X POST http://localhost:3001/knowledge/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "value investing principles", "limit": 5}'
```

## File Structure

```
quantify/knowledge/
├── dots_ocr_parser.py          # dots.mocr PDF parser
├── turbovec_retriever.py       # TurboVec + Ollama integration
├── create_sample_knowledge.py  # Build knowledge base
├── knowledge_api_turbovec.py   # FastAPI server
├── load_all_pdfs.py            # Legacy pipeline (PyMuPDF)
├── config.json                 # Configuration
└── data/
    ├── extracted_text/         # Parsed markdown from PDFs
    └── turbovec_index/         # Vector index + documents
```

## RAG Integration with Agents

Each agent can query the knowledge base:

```python
from quantify.knowledge.turbovec_retriever import KnowledgeRetriever

# Initialize
retriever = KnowledgeRetriever(
    ollama_base_url="http://localhost:11434",
    embedding_model="mxbai-embed-large",
    index_path="./data/turbovec_index",
)

# Search for relevant knowledge
results = retriever.search(
    query="bullish reversal patterns with volume confirmation",
    k=5,
    category="Technical Pattern",
)

# Inject into agent prompt
context = "\n\n".join([r["content"] for r in results])
prompt = f"Use this knowledge: {context}\n\nAnalyze the stock..."
```

## Benefits

1. **Better Parsing**: dots.mocr understands layout, tables, figures
2. **Semantic Coherence**: Chunks are topically unified, not arbitrary splits
3. **Efficient Storage**: TurboVec 4-bit = 4x less memory
4. **Fast Retrieval**: Sub-millisecond semantic search
5. **Knowledge-Enhanced Agents**: Decisions backed by book knowledge
