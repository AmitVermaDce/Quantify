# Quantify - AI-Powered Trading Analysis Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready trading analysis system combining **multi-agent AI reasoning** with real-time market data and semantic knowledge retrieval. Quantify uses a team of specialized AI analysts to provide comprehensive investment research and trade recommendations.

## Features

- **Multi-Agent Analysis** - Four specialized analyst agents (Market, News, Fundamentals, Sentiment) collaborate to analyze stocks
- **Debate-Style Reasoning** - Bull and Bear researchers debate investment thesis before final decision
- **Risk Management** - Three risk perspectives (Aggressive, Conservative, Neutral) evaluate position sizing
- **Portfolio Manager** - Final synthesis produces actionable trade recommendations with confidence ratings
- **Knowledge-Enhanced Decisions** - RAG-powered knowledge base injects insights from 22+ financial books into every agent's analysis
- **Multiple Data Sources** - Yahoo Finance, Alpha Vantage, StockTwits, Reddit sentiment
- **Checkpoint/Resume** - Long-running analyses can resume from crashes
- **Local LLM Support** - Run with Ollama for privacy and cost savings

## Quick Start

```bash
# Clone the repository
git clone https://github.com/AmitVermaDce/Quantify.git
cd Quantify

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment (optional - works with Ollama out of the box)
cp .env.example .env  # Or create .env with your API keys

# Run analysis
python -m quantify.cli.main
```

## Usage

### Interactive Mode

```bash
python -m quantify.cli.main
```

The CLI will guide you through:
1. **Ticker Symbol** - Stock to analyze (e.g., AAPL, TSLA, NVDA)
2. **Analysis Date** - Trading date for historical analysis
3. **Output Language** - Language for reports
4. **Analyst Selection** - Choose which analysts to run
5. **Research Depth** - Quick or Deep analysis mode
6. **LLM Provider** - OpenAI, Anthropic, Google, or Ollama (local)
7. **Model Selection** - Choose models for quick and deep thinking

### Command Options

```bash
# With checkpointing (resumable if crash)
python -m quantify.cli.main analyze --checkpoint

# Clear old checkpoints
python -m quantify.cli.main analyze --clear-checkpoints
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     TradingAgents Graph                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Market    │  │    News     │  │Fundamentals │             │
│  │  Analyst    │  │  Analyst    │  │  Analyst    │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┴────────────────┘                     │
│                          │                                      │
│                    ┌─────▼─────┐                                │
│                    │ Sentiment │                                │
│                    │  Analyst  │                                │
│                    └─────┬─────┘                                │
│                          │                                      │
│         ┌────────────────▼────────────────┐                     │
│         │     Bull vs Bear Debate         │                     │
│         │   (Research Manager judges)     │                     │
│         └────────────────┬────────────────┘                     │
│                          │                                      │
│                    ┌─────▼─────┐                                │
│                    │  Trader   │                                │
│                    └─────┬─────┘                                │
│                          │                                      │
│         ┌────────────────▼────────────────┐                     │
│         │   Risk Debate (3 perspectives)  │                     │
│         │  Aggressive │ Neutral │ Conservative                  │
│         └────────────────┬────────────────┘                     │
│                          │                                      │
│                    ┌─────▼─────┐                                │
│         ┌─────────►│ Portfolio │◄─────────┐                     │
│         │          │  Manager  │          │                     │
│         │          └─────┬─────┘          │                     │
│         │                │                │                     │
│         │          ┌─────▼─────┐          │                     │
│         │          │  FINAL    │          │                     │
│         │          │ DECISION  │          │                     │
│         │          └───────────┘          │                     │
│         └─────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
Quantify/
├── quantify/              # Main Python package
│   ├── __init__.py
│   ├── agents/            # Trading agent implementations
│   │   ├── base_analyst.py      # Base class for analyst nodes (with KB injection)
│   │   ├── base_researcher.py   # Base class for bull/bear researchers (with KB)
│   │   ├── base_debator.py      # Base class for risk debators (with KB)
│   │   ├── knowledge_mixin.py   # Knowledge base integration mixin
│   │   ├── analysts/            # Market, news, fundamentals, sentiment analysts
│   │   ├── researchers/         # Bull and Bear researchers
│   │   ├── risk_mgmt/           # Aggressive, Conservative, Neutral debators
│   │   ├── managers/            # Research Manager, Portfolio Manager
│   │   └── trader/              # Trader agent
│   ├── cli/               # Command-line interface
│   │   └── main.py        # Main CLI entry point
│   ├── dataflows/         # Market data vendors
│   │   ├── y_finance.py   # Yahoo Finance (primary)
│   │   ├── alpha_vantage.py  # Alpha Vantage (fallback)
│   │   ├── yfinance_news.py  # News from Yahoo Finance
│   │   ├── stocktwits.py  # StockTwits sentiment
│   │   └── reddit.py      # Reddit sentiment (r/wallstreetbets)
│   ├── graph/             # LangGraph workflow
│   │   ├── trading_graph.py   # Main graph orchestration
│   │   ├── setup.py       # Graph setup
│   │   ├── propagation.py # State propagation
│   │   ├── checkpointer.py# Checkpoint management
│   │   └── analyst_execution.py  # Analyst execution planning
│   ├── llm_clients/       # LLM provider clients
│   │   ├── openai_client.py
│   │   ├── anthropic_client.py
│   │   ├── google_client.py
│   │   └── ollama_client.py
│   └── default_config.py  # Default configuration
├── knowledge_base/        # RAG Knowledge Base (separate module)
│   ├── service.py         # KnowledgeService - main API for agents
│   ├── turbovec_retriever.py  # TurboVec 4-bit quantized retrieval
│   ├── build_knowledge_base.py  # PDF→chunks→embeddings pipeline
│   ├── knowledge_api_simple.py  # REST API server
│   ├── market_data/       # Source PDFs (single location for all books)
│   │   ├── Investing_Books/
│   │   └── Psychological_Books/
│   ├── tests/             # Test suite (33 passing tests)
│   └── data/
│       ├── extracted_text/    # Extracted text from PDFs
│       └── knowledge_base/    # Indexed knowledge (18,443 chunks)
│           ├── index.tq       # TurboVec index (~10MB)
│           ├── documents.json # Document metadata
│           └── config.json    # Configuration
├── quantify/tests/
│   └── test_knowledge_integration.py  # End-to-end integration test
├── .env                   # Environment variables (create from .env.example)
├── .gitignore
├── pyproject.toml         # Project metadata and dependencies
├── requirements.txt       # Python dependencies
├── CHANGELOG.md           # Version history
└── README.md              # This file
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# LLM Provider (options: openai, anthropic, google, ollama)
TRADINGAGENTS_LLM_PROVIDER=ollama

# Model selection
TRADINGAGENTS_DEEP_THINK_LLM=qwen3.5
TRADINGAGENTS_QUICK_THINK_LLM=qwen3.5

# Ollama endpoint
TRADINGAGENTS_LLM_BACKEND_URL=http://localhost:11434

# Output language
TRADINGAGENTS_OUTPUT_LANGUAGE=English

# Debate rounds
TRADINGAGENTS_MAX_DEBATE_ROUNDS=1
TRADINGAGENTS_MAX_RISK_ROUNDS=1

# Checkpointing
TRADINGAGENTS_CHECKPOINT_ENABLED=false
```

### API Keys

For cloud LLM providers, set the appropriate API key:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Google
export GOOGLE_API_KEY=...
```

**Note:** Ollama (local LLM) requires no API keys.

## Supported LLM Providers

| Provider | Models | Notes |
|----------|--------|-------|
| **Ollama** | Any local model | Private, no API costs |
| **OpenAI** | GPT-4, GPT-4 Turbo | Best quality |
| **Anthropic** | Claude 3.5/3.7 | Strong reasoning |
| **Google** | Gemini 2.5 | Good value |

## Data Sources

### External API Calls

Quantify fetches real-time data from multiple external sources:

| Source | Data Type | API Key Required | Rate Limit |
|--------|-----------|-----------------|------------|
| **Yahoo Finance** | OHLCV, Fundamentals, Financial Statements, Insider Transactions, News | No | Unofficial |
| **Alpha Vantage** | OHLCV, Technical Indicators, Fundamentals, News | Yes (free tier) | 5/min, 500/day |
| **StockTwits** | Social Sentiment (Bullish/Bearish posts) | No | ~10/min |
| **Reddit** | Social Sentiment (r/wallstreetbets, r/stocks, r/investing) | No | ~10/min |
| **Ollama** | LLM Inference & Embeddings (local) | No (runs locally) | N/A |

### Default Configuration

By default, **yfinance** is used for all market data. Alpha Vantage is available as an alternative:

```bash
# In .env file, switch data vendors:
TRADINGAGENTS_DATA_VENDORS_CORE_STOCK_APIS=alpha_vantage
TRADINGAGENTS_DATA_VENDORS_TECHNICAL_INDICATORS=alpha_vantage
TRADINGAGENTS_DATA_VENDORS_FUNDAMENTAL_DATA=alpha_vantage
TRADINGAGENTS_DATA_VENDORS_NEWS_DATA=alpha_vantage

# Set your Alpha Vantage API key:
ALPHA_VANTAGE_API_KEY=your_key_here
```

Get a free Alpha Vantage API key at: https://www.alphavantage.co/support/#api-key

## Output

After analysis, Quantify produces:

1. **Market Report** - Technical analysis with indicators
2. **News Report** - Macroeconomic and company news summary
3. **Fundamentals Report** - Financial statement analysis
4. **Sentiment Report** - Social media sentiment breakdown
5. **Investment Plan** - Bull/Bear debate conclusion
6. **Risk Assessment** - Multi-perspective risk analysis
7. **Final Trade Decision** - BUY/HOLD/SELL recommendation with rating

Reports are saved to `~/.tradingagents/logs/<TICKER>/<DATE>/reports/`

## Knowledge Base

Quantify includes a RAG-powered knowledge base with insights from 22+ financial books including:

- **Value Investing**: Benjamin Graham (Intelligent Investor, Security Analysis), Seth Klarman (Margin of Safety)
- **Quality Investing**: Philip Fisher, Bruce Greenwald, Joel Greenblatt
- **Trading Psychology**: Daniel Kahneman, Mark Douglas, Tom Hougaard
- **Market Psychology**: Gustave Le Bon, Robert Cialdini, James Surowiecki

### Building the Knowledge Base

```bash
cd knowledge_base
python build_knowledge_base.py \
  --market-data-dir ../quantify/market_data/ \
  --output ./data/knowledge_base
```

### Adding New PDFs

```bash
cd knowledge_base
python -c "
from service import KnowledgeService
kb = KnowledgeService()
kb.add_pdfs(['../quantify/market_data/Investing_Books/new_book.pdf'])
print(f'Total documents: {len(kb)}')
"
```

### Knowledge Base Tests

```bash
# Run all knowledge base tests
python -m pytest knowledge_base/tests/ -v -o addopts=""

# Run end-to-end integration test
python test_knowledge_integration.py --test all
```

See [knowledge_base/USAGE.md](knowledge_base/USAGE.md) for complete documentation.

## Development

```bash
# Run tests
pytest quantify/tests/

# Lint
ruff check quantify/

# Type check
mypy quantify/
```

## Requirements

- Python 3.10+
- Ollama (for local LLM) or API keys for cloud providers
- Ollama with `mxbai-embed-large` model (for knowledge base embeddings)

### Install Ollama and Models

```bash
# Install Ollama (macOS)
brew install ollama

# Pull required models
ollama pull llama-3.2-3b      # Fast thinking model
ollama pull qwen2.5:7b        # Deep thinking model (or any preferred)
ollama pull mxbai-embed-large # Embeddings for knowledge base
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

- GitHub Issues: https://github.com/AmitVermaDce/Quantify/issues
- Discussions: https://github.com/AmitVermaDce/Quantify/discussions

## Acknowledgments

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) - Workflow orchestration
- [LangChain](https://github.com/langchain-ai/langchain) - LLM abstractions
- [yfinance](https://github.com/ranaroussi/yfinance) - Market data
- [Ollama](https://ollama.ai) - Local LLM runtime
