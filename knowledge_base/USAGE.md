# Knowledge Base Usage Guide

## Directory Structure

```
Quantify/
├── knowledge_base/          # Knowledge base code, data, and source PDFs
│   ├── build_knowledge_base.py   # Build/rebuild knowledge base
│   ├── service.py                # Main API for agents
│   ├── turbovec_retriever.py     # TurboVec integration
│   ├── knowledge_api_simple.py   # REST API server
│   ├── market_data/              # Source PDFs (single location)
│   │   ├── Investing_Books/
│   │   └── Psychological_Books/
│   ├── data/
│   │   ├── extracted_text/       # Extracted text from PDFs
│   │   └── knowledge_base/       # TurboVec index (18K+ documents)
│   └── tests/                    # Test suite
│
├── quantify/
│   └── agents/
│       └── knowledge_mixin.py    # Agent integration mixin
│
└── quantify/tests/
    └── test_knowledge_integration.py  # End-to-end integration test
```

## Quick Start

### 1. Build Knowledge Base (Initial)

```bash
cd knowledge_base
python build_knowledge_base.py \
  --market-data-dir ./market_data/ \
  --output ./data/knowledge_base
```

### 2. Add New PDFs (Incremental)

```python
from knowledge_base import KnowledgeService

kb = KnowledgeService()

# Add new PDFs to existing index
kb.add_pdfs([
    "new_investing_book.pdf",
    "market_research.pdf",
])

print(f"Total documents: {len(kb)}")
```

### 3. Use in TradingAgents

```python
from quantify.agents.knowledge_mixin import KnowledgeMixin, get_knowledge_context

# Option A: Direct lookup
context = get_knowledge_context("margin of safety", k=3)
print(context)

# Option B: Mixin class (recommended for agents)
class MyTradingAgent(KnowledgeMixin):
    USE_KNOWLEDGE = True
    DEFAULT_KNOWLEDGE_QUERY = "technical analysis patterns"
    KNOWLEDGE_K = 5

    def __call__(self, state):
        # Inject knowledge into state
        self._inject_knowledge(state, query="bullish reversal patterns")

        # Build prompt with knowledge section
        system_msg = self._build_system_message(state)
        system_msg += self._build_knowledge_section(state)

        # ... rest of agent logic
```

### 4. Start REST API (Optional)

```bash
cd knowledge_base
python knowledge_api_simple.py --port 3001 --index-path data/knowledge_base

# Query from anywhere
curl -X POST http://localhost:3001/knowledge/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "value investing", "limit": 5}'
```

## API Reference

### KnowledgeService

```python
from knowledge_base import KnowledgeService

kb = KnowledgeService(
    index_path="knowledge_base/data/knowledge_base",  # Auto-detected
    ollama_url="http://localhost:11434",
    embedding_model="mxbai-embed-large",
)

# Search
results = kb.search("margin of safety", k=5)

# Get formatted context for prompts
context = kb.get_context("value investing", k=3, max_tokens=1500)

# Inject into prompt
prompt = kb.inject_knowledge(
    base_prompt,
    query="risk management",
    section_name="Risk Management Knowledge"
)

# Add new PDFs
kb.add_pdfs(["new_book.pdf"])

# Stats
stats = kb.stats()
print(f"Documents: {stats['total_documents']}")
```

### KnowledgeMixin (for Agents)

```python
from quantify.agents.knowledge_mixin import KnowledgeMixin

class BullResearcher(KnowledgeMixin):
    USE_KNOWLEDGE = True
    DEFAULT_KNOWLEDGE_QUERY = "bullish patterns growth investing"
    KNOWLEDGE_K = 3

    def __call__(self, state):
        # Auto-inject based on agent role
        self._inject_knowledge(state)

        # Build prompt with knowledge
        prompt = self._build_prompt(state)
        prompt += self._build_knowledge_section(state)

        # Invoke LLM
        response = self.llm.invoke(prompt)
        return {"messages": [response]}
```

## Configuration

### Chunk Settings

```bash
python build_knowledge_base.py \
  --chunk-size 2000 \      # Characters per chunk
  --chunk-overlap 200 \    # Overlap between chunks
  --verbose
```

### Category Filtering

```python
# Only search investing books
kb.search("value investing", category="Investing_Books")

# Only search psychology books
kb.search("loss aversion", category="Psychological_Books")
```

## Adding New PDFs

When you add new PDFs:

1. **Place PDF in correct category:**
   - `knowledge_base/market_data/Investing_Books/` for investment books
   - `knowledge_base/market_data/Psychological_Books/` for psychology books

2. **Run incremental update:**
   ```bash
   cd knowledge_base
   python -c "
   from service import KnowledgeService
   kb = KnowledgeService()
   kb.add_pdfs(['market_data/Investing_Books/new_book.pdf'])
   "
   ```

3. **Verify:**
   ```python
   print(kb.stats())
   ```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  market_data/ (PDFs)                                    │
│  ├── Investing_Books/                                   │
│  └── Psychological_Books/                               │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  knowledge_base/build_knowledge_base.py                 │
│  1. Extract text (PyMuPDF)                              │
│  2. Chunk with semantic boundaries                      │
│  3. Generate embeddings (Ollama)                        │
│  4. Build TurboVec index (4-bit quantized)              │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  knowledge_base/data/knowledge_base/                    │
│  ├── index.tq (TurboVec 4-bit index)                    │
│  ├── documents.json (document metadata)                 │
│  └── config.json (configuration)                        │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  TradingAgents with KnowledgeMixin                      │
│  ├── Portfolio Manager                                  │
│  ├── Risk Manager                                       │
│  ├── Bull/Bear Researchers                              │
│  └── Technical Analyst                                  │
│                                                          │
│  Each agent queries knowledge base during decision      │
│  making, injecting relevant book knowledge into prompts │
└─────────────────────────────────────────────────────────┘
```

## Troubleshooting

### "Knowledge base not found"
```bash
# Check index exists
ls -la knowledge_base/data/knowledge_base/

# Should see:
# - index.tq (~10MB)
# - documents.json (~40MB)
# - config.json
```

### "Ollama not responding"
```bash
# Check Ollama is running
ollama list

# Pull embedding model if missing
ollama pull mxbai-embed-large
```

### "Import errors"
```bash
# Ensure knowledge_base is in path
cd /path/to/Quantify
python -c "import sys; print('\\n'.join(sys.path))"

# Should include path to knowledge_base
```
