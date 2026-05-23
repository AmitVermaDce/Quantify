# Knowledge Base Service

Production-ready knowledge extraction and semantic search system for trading expertise. Extracts structured trading concepts from books using LLM (Ollama) and provides a REST API for semantic search. Now includes real-time market data layer and TradingAgents RAG integration.

## Features

### Knowledge Extraction
- **PDF Text Extraction**: Batch extract text from trading books
- **LLM-Powered Knowledge Extraction**: Extract structured trading concepts using qwen3.5:cloud
- **Vector Embeddings**: Generate embeddings with mxbai-embed-large (1024 dimensions)
- **Semantic Search**: PostgreSQL + pgvector for similarity search
- **REST API**: FastAPI service with health checks and structured logging
- **CLI Pipeline**: Orchestrated pipeline with progress tracking
- **RAG Integration**: Knowledge client for TradingAgents prompt injection

### Market Data (New)
- **Yahoo Finance**: OHLCV data, technical indicators, fundamentals, news (no API key)
- **Alpha Vantage**: Alternative data source (requires API key)
- **Social Sentiment**: Reddit and StockTwits sentiment analysis
- **Unified API**: Single REST endpoint for all market data

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Check connections
make check-db
make check-ollama

# 3. Extract text from PDFs
make extract INPUT_DIR=../resources/books

# 4. Extract knowledge concepts
make knowledge INPUT_DIR=./data/extracted_text

# 5. Seed database
make seed INPUT=./data/concepts/all_concepts.json

# 6. Start Knowledge API server (FastAPI - Python)
make api

# 7. Start Market Data service
make data-service
```

## API Endpoints

### Knowledge Service (Port 3001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/knowledge/search` | POST | Semantic search for concepts |
| `/knowledge/:id` | GET | Get concept by ID with relationships |
| `/knowledge` | GET | List concepts with pagination |
| `/strategies` | GET | Find strategies by conditions |
| `/health` | GET | Health check with status |

### Market Data Service (Port 3002)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stock/<symbol>` | GET | OHLCV stock data |
| `/indicators/<symbol>` | GET | Technical indicators (RSI, MACD, SMA, etc.) |
| `/fundamentals/<symbol>` | GET | Fundamental data (ratios, financials) |
| `/news` | GET | News articles |
| `/sentiment/<symbol>` | GET | Social media sentiment |
| `/symbol/<symbol>/full` | GET | Complete analysis package |
| `/health` | GET | Health check |

### Example: Search for Concepts

```bash
curl -X POST http://localhost:3001/knowledge/search \
  -H "Content-Type: application/json" \
  -d '{"query": "bullish reversal patterns", "limit": 5}'
```

### Example: Get Stock Data

```bash
curl "http://localhost:3002/stock/AAPL?start_date=2026-01-01&end_date=2026-05-20"
```

### Example: Get Technical Indicators

```bash
curl "http://localhost:3002/indicators/NVDA?indicators=rsi,macd,sma_50,sma_200"
```

### Example: Get Full Analysis

```bash
curl http://localhost:3002/symbol/TSLA/full
```

## Project Structure

```
knowledge-base/
├── config.json           # Central configuration
├── .env.example          # Environment variables template
├── requirements.txt      # Python dependencies
├── package.json          # Node.js dependencies (optional)
├── Makefile              # Build commands
├── Dockerfile            # Container image
├── schema/
│   └── knowledge_graph.sql  # PostgreSQL schema with pgvector
├── scripts/
│   ├── cli.py            # CLI orchestrator
│   ├── extract_pdf.py    # PDF text extraction
│   ├── extract_knowledge.py  # LLM concept extraction
│   ├── seed_knowledge.py # Database seeding
│   ├── knowledge_api.py  # FastAPI REST API (knowledge)
│   ├── knowledge_client.py   # RAG client for TradingAgents
│   ├── data_service.py   # Flask REST API (market data)
│   ├── utils.py          # Shared utilities
│   └── data/             # Market data layer
│       ├── interface.py      # Vendor abstraction
│       ├── yfinance_data.py  # Yahoo Finance client
│       ├── alpha_vantage.py  # Alpha Vantage client
│       └── sentiment.py      # Reddit/StockTwits
├── tests/
│   └── test_utils.py     # Unit tests
└── data/
    ├── extracted_text/   # Extracted PDF text
    └── concepts/         # Extracted concepts JSON
```

## Configuration

Edit `config.json` or use environment variables:

```json
{
  "llm": {
    "provider": "ollama",
    "base_url": "http://localhost:11434",
    "model": "qwen3.5:cloud",
    "embedding_model": "mxbai-embed-large",
    "embedding_dimension": 1024
  },
  "database": {
    "url": "postgresql://localhost:5432/trading_knowledge"
  },
  "extraction": {
    "chunk_size": 2000,
    "chunk_overlap": 200,
    "retry_attempts": 3,
    "retry_delay_seconds": 10
  },
  "knowledge_service": {
    "url": "http://localhost:3001"
  }
}
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | Ollama API endpoint | `http://localhost:11434` |
| `OLLAMA_MODEL` | LLM model name | `qwen3.5:cloud` |
| `EMBEDDING_MODEL` | Embedding model | `mxbai-embed-large` |
| `DATABASE_URL` | PostgreSQL connection | `postgresql://localhost:5432/trading_knowledge` |
| `KNOWLEDGE_SERVICE_URL` | Knowledge API URL | `http://localhost:3001` |
| `KNOWLEDGE_QUERY_TIMEOUT` | Query timeout (seconds) | `2.0` |
| `DATA_SERVICE_PORT` | Market Data API port | `3002` |
| `DATA_VENDOR` | Primary data vendor | `yfinance` |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage API key | (required for Alpha Vantage) |
| `LOG_LEVEL` | Logging level | `INFO` |

## Integration with TradingAgents (RAG)

The knowledge base provides RAG-style context injection for TradingAgents analysts:

```python
from knowledge_client import query_knowledge, get_trading_context, format_knowledge_for_prompt

# Query for specific concept
results = query_knowledge("RSI bullish divergence", limit=3)

# Get context for analyst prompt
context = get_trading_context("NVDA", "2026-05-20", analyst_type="market")

# Inject into analyst system prompt
system_prompt = """You are a Technical Analyst. Analyze the market and provide insights.

""" + context + """

Use the provided tools to gather data and make your analysis."""

# Or format manually
knowledge_text = format_knowledge_for_prompt(results, max_items=5)
```

### Analyst-Specific Queries

```python
# Market Analyst - technical patterns
context = get_trading_context("AAPL", "2026-05-20", analyst_type="market")

# News Analyst - sentiment and macro
context = get_trading_context("SPY", "2026-05-20", analyst_type="news")

# Fundamentals Analyst - valuation metrics
context = get_trading_context("MSFT", "2026-05-20", analyst_type="fundamentals")

# Risk Management - position sizing rules
context = get_trading_context("", "2026-05-20", analyst_type="risk")
```

### Full Example in TradingAgents

```python
# In tradingagents/agents/analysts/market_analyst.py
from knowledge_client import get_trading_context

def analyze_market(ticker: str, date: str):
    # Get relevant trading knowledge
    knowledge_context = get_trading_context(ticker, date, analyst_type="market")
    
    # Build system prompt with injected knowledge
    system_prompt = BASE_SYSTEM_PROMPT
    if knowledge_context:
        system_prompt += f"\n\n{knowledge_context}"
    
    # Continue with analysis...
```

## Market Data Providers

### Yahoo Finance (Default)
- **No API key required**
- OHLCV data with multiple intervals
- Technical indicators via stockstats
- Company fundamentals
- News articles
- Fallback when Alpha Vantage rate limited

### Alpha Vantage
- **Requires API key** (free at alphavantage.co)
- Higher quality fundamental data
- News with sentiment scores
- Insider transactions
- Economic data

### Social Sentiment
- **Reddit**: r/wallstreetbets, r/stocks, r/investing
- **StockTwits**: Real-time trader sentiment
- Keyword-based sentiment analysis
- Bullish/bearish/neutral classification

## Docker Deployment

```bash
# Build image
docker build -t knowledge-base:latest .

# Run Knowledge Service only
docker run -d -p 3001:3001 \
  -e DATABASE_URL=postgresql://user:pass@db:5432/trading_knowledge \
  -e OLLAMA_BASE_URL=http://ollama:11434 \
  knowledge-base:latest

# Run both services
docker run -d -p 3001:3001 -p 3002:3002 \
  -e DATABASE_URL=postgresql://user:pass@db:5432/trading_knowledge \
  -e OLLAMA_BASE_URL=http://ollama:11434 \
  -e DATA_VENDOR=yfinance \
  knowledge-base:latest
```

## Development

```bash
# Run tests
make test

# Check code style
pip install ruff black
ruff check scripts/
black --check scripts/

# Run with verbose logging
python scripts/cli.py pipeline --input-dir ./books --verbose

# Test knowledge client
python scripts/knowledge_client.py --query "bullish reversal patterns"

# Test knowledge API directly
python scripts/knowledge_api.py --reload
```

## Database Schema

The system uses PostgreSQL with pgvector extension:

- **concepts**: Core trading concepts with embeddings
- **indicators**: Technical indicators
- **patterns**: Chart patterns
- **strategies**: Trading strategies
- **relationships**: Graph relationships between concepts
- **market_regimes**: Market conditions

Run schema setup:
```bash
psql -d trading_knowledge -f schema/knowledge_graph.sql
```

## Troubleshooting

### Ollama Connection Failed
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull required models
ollama pull qwen3.5:cloud
ollama pull mxbai-embed-large
```

### Database Connection Failed
```bash
# Create database
createdb trading_knowledge

# Install pgvector extension
psql -d trading_knowledge -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Alpha Vantage Rate Limit
- Free tier: 5 requests/minute, 500 requests/day
- Upgrade to paid plan for higher limits
- System automatically falls back to Yahoo Finance

### Knowledge Service Not Returning Results
1. Ensure the knowledge service is running: `make api`
2. Check health: `curl http://localhost:3001/health`
3. Seed data first: `make seed-sample` or `make seed INPUT=./data/concepts/all_concepts.json`

### Extraction Taking Too Long
- qwen3.5:cloud is a large model (~400B params)
- Each chunk takes 10-30 seconds
- A 1000-page book = 400+ chunks = 2-4 hours
- Use `--verbose` for progress tracking

## License

Internal use only.
