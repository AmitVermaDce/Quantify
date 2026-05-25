# Quantify Architecture

This document describes the architecture of the Quantify TradingAgents system.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRADING AGENTS SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    1. INPUT LAYER                                   │    │
│  │   • Ticker Symbol (e.g., AAPL, SPY)                                │    │
│  │   • Trade Date                                                     │    │
│  │   • Asset Type (stock/crypto)                                      │    │
│  │   • Selected Analysts (market, news, sentiment, fundamentals)      │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    2. ANALYST TEAM (Parallel/Sequential)            │    │
│  │   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │    │
│  │   │ Market Analyst  │  │ Sentiment       │  │ News Analyst    │   │    │
│  │   │ + Technical     │  │ Analyst         │  │ + Headlines     │   │    │
│  │   │ + Indicators    │  │ + StockTwits    │  │ + Global News   │   │    │
│  │   │ + SMA/EMA/MACD  │  │ + Reddit        │  │ + Insider       │   │    │
│  │   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘   │    │
│  │            │                    │                    │             │    │
│  │            ▼                    ▼                    ▼             │    │
│  │   ┌─────────────────────────────────────────────────────────────┐ │    │
│  │   │              Fundamentals Analyst                            │ │    │
│  │   │   + Financial Statements  + Balance Sheet                    │ │    │
│  │   │   + Cash Flow           + Income Statement                   │ │    │
│  │   └─────────────────────────────────────────────────────────────┘ │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    3. RESEARCH TEAM (Debate)                        │    │
│  │   ┌──────────────┐         ┌──────────────┐                        │    │
│  │   │ Bull         │◄───────►│ Bear         │  Multi-round debate    │    │
│  │   │ Researcher   │         │ Researcher   │  (configurable rounds) │    │
│  │   └──────┬───────┘         └──────┬───────┘                        │    │
│  │          │                        │                                 │    │
│  │          ▼                        ▼                                 │    │
│  │   ┌─────────────────────────────────────────────────────────────┐  │    │
│  │   │              Research Manager                                │  │    │
│  │   │         Synthesizes debate → Investment Plan                 │  │    │
│  │   └─────────────────────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    4. TRADING TEAM                                  │    │
│  │   ┌─────────────────────────────────────────────────────────────┐  │    │
│  │   │              Trader                                          │  │    │
│  │   │   Converts Investment Plan → Concrete Transaction Proposal   │  │    │
│  │   │   (BUY/SELL/HOLD with entry, target, stop-loss)              │  │    │
│  │   └─────────────────────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    5. RISK MANAGEMENT (Debate)                      │    │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │    │
│  │   │ Aggressive   │  │ Conservative │  │ Neutral      │             │    │
│  │   │ Analyst      │  │ Analyst      │  │ Analyst      │             │    │
│  │   │ High risk    │  │ Low risk     │  │ Balanced     │             │    │
│  │   │ High reward  │  │ Stability    │  │ Moderate     │             │    │
│  │   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │    │
│  │          │                 │                  │                     │    │
│  │          ▼                 ▼                  ▼                     │    │
│  │   ┌─────────────────────────────────────────────────────────────┐  │    │
│  │   │              Portfolio Manager                               │  │    │
│  │   │         Final Decision: BUY/HOLD/SELL + Position Size        │  │    │
│  │   └─────────────────────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    6. OUTPUT LAYER                                  │    │
│  │   • Final Trade Decision (structured: action, quantity, price)     │    │
│  │   • Complete Report (markdown with all analyst reports)            │    │
│  │   • Memory Log (for future reflection on outcomes)                 │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Knowledge Base Integration

The knowledge base is integrated at **every agent level**, automatically injecting relevant insights from financial books into each decision:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KNOWLEDGE BASE INTEGRATION                          │
│                         (Auto-injected at every step)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              knowledge_base/ (root level)                            │   │
│  │   ┌──────────────────────────────────────────────────────────────┐  │   │
│  │   │  service.py        - KnowledgeService API                    │  │   │
│  │   │  turbovec_retriever  - 4-bit quantized vector search         │  │   │
│  │   │  build_knowledge_base.py - PDF→chunks→embeddings pipeline    │  │   │
│  │   │  market_data/        - Source PDFs (single location)         │  │   │
│  │   └──────────────────────────────────────────────────────────────┘  │   │
│  │   ┌──────────────────────────────────────────────────────────────┐  │   │
│  │   │  data/knowledge_base/                                        │  │   │
│  │   │    • index.tq (~10MB) - TurboVec index                       │  │   │
│  │   │    • documents.json (~40MB) - 18,443 chunks                  │  │   │
│  │   │    • Investing_Books (7,769 docs)                            │  │   │
│  │   │    • Psychological_Books (10,674 docs)                       │  │   │
│  │   └──────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│         Auto-injected into every agent prompt via:                         │
│         • base_researcher.py (Bull/Bear)                                   │
│         • base_debator.py (Risk analysts)                                  │
│         • base_analyst.py (News/Fundamentals)                              │
│         • market_analyst.py, sentiment_analyst.py                          │
│         • portfolio_manager.py, trader.py                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## State Management

The system uses LangGraph's state management with the following state structure:

```python
AgentState:
  - messages: List[Message]           # LangChain message history
  - company_of_interest: str          # Ticker symbol
  - asset_type: str                   # "stock" or "crypto"
  - trade_date: str                   # YYYY-MM-DD
  - sender: str                       # Last agent to act
  - past_context: str                 # Memory log from prior decisions
  
  # Analyst Reports
  - market_report: str
  - sentiment_report: str
  - news_report: str
  - fundamentals_report: str
  
  # Research Team
  - investment_debate_state: InvestDebateState
  - investment_plan: str
  
  # Trader
  - trader_investment_plan: str
  
  # Risk Team
  - risk_debate_state: RiskDebateState
  - final_trade_decision: str
```

## Directory Structure

```
Quantify/
├── quantify/                    # Main Python package
│   ├── agents/                  # Agent implementations
│   │   ├── base_analyst.py      # Base class with KB injection
│   │   ├── base_researcher.py   # Base class with KB injection
│   │   ├── base_debator.py      # Base class with KB injection
│   │   ├── knowledge_mixin.py   # Optional mixin for custom agents
│   │   ├── analysts/            # 4 analyst types
│   │   ├── researchers/         # Bull/Bear researchers
│   │   ├── risk_mgmt/           # 3 risk debators
│   │   ├── managers/            # Research Manager, Portfolio Manager
│   │   └── trader/              # Trader agent
│   ├── cli/                     # Command-line interface
│   ├── dataflows/               # Data providers (yfinance, etc.)
│   ├── graph/                   # LangGraph workflow
│   │   ├── trading_graph.py     # Main orchestration
│   │   ├── analyst_execution.py # Analyst execution planning
│   │   ├── checkpointer.py      # Checkpoint/resume
│   │   └── ...
│   ├── llm_clients/             # LLM provider clients
│   └── default_config.py        # Default configuration
│
├── knowledge_base/              # RAG Knowledge Base
│   ├── service.py               # KnowledgeService API
│   ├── turbovec_retriever.py    # TurboVec retrieval
│   ├── build_knowledge_base.py  # Build pipeline
│   ├── market_data/             # Source PDFs (Investing_Books, Psychological_Books)
│   ├── tests/                   # 33 passing tests
│   ├── data/
│   │   ├── extracted_text/      # Extracted text from PDFs
│   │   └── knowledge_base/      # Indexed knowledge
│   └── USAGE.md                 # Usage documentation
│
├── quantify/tests/
│   └── test_knowledge_integration.py  # E2E test
└── ...
```

## Agent Communication Pattern

Agents communicate through LangGraph's state graph:

1. **Parallel Phase**: Analysts run concurrently (configurable)
2. **Sequential Debate**: Bull ↔ Bear debate (Research Manager judges)
3. **Trader**: Converts plan to transaction proposal
4. **Risk Debate**: Aggressive ↔ Conservative ↔ Neutral (Portfolio Manager judges)
5. **Final Decision**: Portfolio Manager produces BUY/HOLD/SELL

## Knowledge Injection Flow

Every agent automatically injects knowledge:

```python
# In base_researcher.py, base_debator.py, base_analyst.py:

def _build_prompt(self, state: Dict) -> str:
    prompt = self.PROMPT_TEMPLATE.format(...)
    
    # Auto-inject knowledge if enabled
    if self.USE_KNOWLEDGE:
        kb = _get_kb_service()
        if kb:
            ticker = state.get("ticker", "market")
            query = f"{ticker} investment analysis"
            context = kb.get_context(query, k=2)
            if context:
                prompt += f"\n\n== Relevant Financial Knowledge ==\n{context}"
    
    return prompt
```

## Configuration

### Default Config (`default_config.py`)

```python
DEFAULT_CONFIG = {
    "llm_provider": "openai",
    "quick_think_llm": "gpt-4o-mini",
    "deep_think_llm": "gpt-4o",
    "max_debate_rounds": 2,
    "max_risk_discuss_rounds": 2,
    "data_cache_dir": "~/.tradingagents/cache",
    "results_dir": "~/.tradingagents/results",
    ...
}
```

### Environment Variables (`.env`)

```bash
TRADINGAGENTS_LLM_PROVIDER=ollama
TRADINGAGENTS_DEEP_THINK_LLM=qwen2.5:7b
TRADINGAGENTS_QUICK_THINK_LLM=llama-3.2-3b
TRADINGAGENTS_LLM_BACKEND_URL=http://localhost:11434
TRADINGAGENTS_OUTPUT_LANGUAGE=English
```

## Testing

### Unit Tests

```bash
# Main package tests
pytest quantify/tests/

# Knowledge base tests
pytest knowledge_base/tests/ -v -o addopts=""
```

### Integration Tests

```bash
# Full pipeline test
python test_knowledge_integration.py --test all --ticker AAPL --date 2024-01-15
```

## LLM Providers

| Provider | Models | Configuration |
|----------|--------|---------------|
| Ollama | Local models | `TRADINGAGENTS_LLM_PROVIDER=ollama` |
| OpenAI | GPT-4, GPT-4 Turbo | `OPENAI_API_KEY` env var |
| Anthropic | Claude 3.5/3.7 | `ANTHROPIC_API_KEY` env var |
| Google | Gemini 2.5 | `GOOGLE_API_KEY` env var |

## Data Flow

```
User Input (CLI)
       │
       ▼
┌─────────────────┐
│ TradingAgentsGraph │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analyst Team   │ → market_report, sentiment_report, ...
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Research Team  │ → investment_plan
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Trader         │ → trader_investment_plan
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Risk Team      │ → final_trade_decision
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Portfolio Mgr  │ → BUY/HOLD/SELL
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Output (MD/JSON)│
└─────────────────┘
```

## Checkpoint/Resume

When `--checkpoint` is enabled:

1. Each node's output is saved to SQLite
2. On crash, resume from last successful node
3. Thread ID = `{ticker}:{trade_date}` ensures correct state

## Memory Log

The memory log (`TradingMemoryLog`) stores:

1. Past decisions for same ticker
2. Outcomes (when available)
3. Cross-ticker lessons

Injected as `past_context` at pipeline start.
