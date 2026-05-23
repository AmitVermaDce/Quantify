# TradingAgents Integration Guide

## Repository Overview

The TradingAgents framework is a comprehensive multi-agent trading system with:

- **91 Python files** across agents, dataflows, graph orchestration, and LLM clients
- **LangGraph-based** workflow orchestration
- **Multi-provider LLM support** (OpenAI, Google, Anthropic, xAI, DeepSeek, Qwen, GLM, MiniMax)
- **Real-time data integration** (Alpha Vantage, Yahoo Finance, StockTwits, Reddit)

## Architecture

```
tradingagents/
├── agents/              # Specialized trading agents
│   ├── analysts/        # Analysis specialists
│   │   ├── fundamentals_analyst.py
│   │   ├── market_analyst.py
│   │   ├── news_analyst.py
│   │   └── sentiment_analyst.py
│   ├── researchers/     # Bull/Bear debate team
│   │   ├── bull_researcher.py
│   │   └── bear_researcher.py
│   ├── managers/        # Decision makers
│   │   ├── portfolio_manager.py
│   │   └── research_manager.py
│   ├── risk_mgmt/       # Risk assessment
│   │   ├── aggressive_debator.py
│   │   ├── conservative_debator.py
│   │   └── neutral_debator.py
│   ├── trader/          # Trade execution
│   │   └── trader.py
│   └── schemas.py       # Pydantic output schemas
├── dataflows/           # Data collection layer
│   ├── alpha_vantage_*  # Market data, news, fundamentals
│   ├── y_finance.py     # Yahoo Finance integration
│   ├── stocktwits.py    # Social sentiment
│   ├── reddit.py        # Reddit posts
│   └── stockstats_utils.py  # Technical indicators
├── graph/               # LangGraph workflow
│   ├── trading_graph.py # Main orchestration
│   ├── analyst_execution.py
│   ├── conditional_logic.py
│   └── reflection.py
└── llm_clients/         # LLM provider support
    ├── openai_client.py
    ├── anthropic_client.py
    ├── google_client.py
    ├── azure_client.py
    └── model_catalog.py
```

## Agent Roles

### Analyst Team
| Agent | Role |
|-------|------|
| Fundamentals Analyst | Company financials, intrinsic value |
| Market Analyst | Technical indicators (RSI, MACD, Bollinger) |
| News Analyst | Macroeconomic events, market impact |
| Sentiment Analyst | StockTwits, Reddit, news sentiment |

### Researcher Team
| Agent | Role |
|-------|------|
| Bull Researcher | Argues for investment (growth potential) |
| Bear Researcher | Argues against (risks, concerns) |

### Decision Makers
| Agent | Role |
|-------|------|
| Research Manager | Synthesizes debate into investment plan |
| Trader | Converts plan to transaction proposal |
| Portfolio Manager | Final approve/reject decision |
| Risk Management | Assesses volatility, drawdown, position sizing |

## Integration with Knowledge Base

Your knowledge base serves as **expert memory** for TradingAgents:

### 1. Pre-Trade Research
```python
# Before analyst agents run, query knowledge base for relevant patterns
response = requests.post('http://localhost:3001/knowledge/search',
    json={"query": "bullish reversal patterns RSI divergence", "limit": 5})
knowledge_context = response.json()['results']

# Inject into analyst prompts
analyst_prompt = f"""
Relevant trading knowledge:
{format_knowledge(knowledge_context)}

Current market data: {market_data}
"""
```

### 2. Decision Validation
```python
# After Trader proposes action, validate against knowledge base
trader_proposal = "BUY AAPL at $150"
similar_cases = requests.post('http://localhost:3001/knowledge/search',
    json={"query": trader_proposal, "category": "Strategy"})

# Check if proposal aligns with established trading wisdom
```

### 3. Risk Management Reference
```python
# Risk agents query knowledge base for position sizing rules
risk_rules = requests.post('http://localhost:3001/knowledge/search',
    json={"query": "Kelly criterion position sizing maximum drawdown"})
```

## Dataflows Available

| Source | Data Type |
|--------|-----------|
| Alpha Vantage | Real-time quotes, technical indicators, fundamentals |
| Yahoo Finance | Historical prices, company info, news |
| StockTwits | Retail sentiment (Bullish/Bearish) |
| Reddit | r/wallstreetbets, r/stocks, r/investing posts |

## LLM Provider Support

The framework supports:
- **OpenAI**: GPT-5.x, GPT-4.x
- **Google**: Gemini 2.5, 2.0
- **Anthropic**: Claude 4.x, 3.7
- **xAI**: Grok 4.x, 3
- **DeepSeek**: DeepSeek-V3
- **Qwen**: Qwen 2.5 (via DashScope)
- **GLM**: GLM-4 (via Zhipu)
- **MiniMax**: M2, M2.5

## Next Steps

1. **Install TradingAgents**
   ```bash
   cd TheProject/tradingagents
   pip install -e .
   ```

2. **Configure Ollama** (your setup)
   ```bash
   # Update .env with Ollama endpoint
   export LLM_PROVIDER=ollama
   export OLLAMA_BASE_URL=http://localhost:11434
   export MODEL_NAME=qwen3.5:cloud
   ```

3. **Connect Knowledge Base**
   - Add knowledge base queries to agent prompts
   - Use retrieved concepts as few-shot examples
   - Log agent decisions back to knowledge base for continuous learning

4. **Run a Trade Analysis**
   ```bash
   python main.py --ticker AAPL --start-date 2026-01-01
   ```

## Code Quality Notes

- Uses **LangGraph** for state machine orchestration
- **Pydantic schemas** for structured agent output
- **Tool-calling** pattern for data access (Market Analyst)
- **Pre-fetch + inject** pattern for Sentiment Analyst
- **Debate-style** reasoning for Bull/Bear researchers
