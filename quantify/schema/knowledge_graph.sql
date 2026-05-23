-- Knowledge Graph Schema for Trading Expertise
-- PostgreSQL with pgvector extension
-- Configured for Ollama embeddings (mxbai-embed-large: 1024 dimensions)

-- Enable pgvector extension
-- Note: Requires superuser. Run: psql -U postgres trading_knowledge < this_file.sql
CREATE EXTENSION IF NOT EXISTS pgvector;

-- ============================================
-- CORE CONCEPTS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS concepts (
    id SERIAL PRIMARY KEY,
    concept_name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    interpretation TEXT,
    conditions TEXT,
    market_context TEXT,
    timeframes TEXT[],
    source_book VARCHAR(255),
    source_chapter VARCHAR(255),
    source_page INTEGER,
    confidence VARCHAR(50),
    embedding vector(1024),  -- mxbai-embed-large dimension (Ollama)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_concepts_category ON concepts(category);
CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts(concept_name);
-- ivfflat index with lists parameter (tune to sqrt(expected row count))
CREATE INDEX IF NOT EXISTS idx_concepts_embedding
    ON concepts USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
-- Composite index for filtered searches
CREATE INDEX IF NOT EXISTS idx_concepts_category_embedding
    ON concepts USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ============================================
-- INDICATORS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS indicators (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    formula TEXT NOT NULL,
    interpretation_overbought TEXT,
    interpretation_oversold TEXT,
    default_period INTEGER,
    data_requirements TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_indicators_name ON indicators(name);

-- ============================================
-- PATTERNS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS patterns (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    pattern_type VARCHAR(50),
    bullish_conditions JSONB,
    bearish_conditions JSONB,
    reliability_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_patterns_name ON patterns(name);
CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type);

-- ============================================
-- STRATEGIES TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS strategies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    asset_class VARCHAR(50),
    market_regime VARCHAR(100),
    entry_logic TEXT NOT NULL,
    exit_logic TEXT NOT NULL,
    stop_loss_rules TEXT,
    position_sizing_rules TEXT,
    risk_reward_ratio DECIMAL(5,2),
    backtest_confidence DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_strategies_name ON strategies(name);
CREATE INDEX IF NOT EXISTS idx_strategies_asset_class ON strategies(asset_class);
CREATE INDEX IF NOT EXISTS idx_strategies_regime ON strategies(market_regime);

-- ============================================
-- RELATIONSHIPS TABLE (Graph Edges)
-- ============================================
CREATE TABLE IF NOT EXISTS relationships (
    id SERIAL PRIMARY KEY,
    from_concept_id INTEGER REFERENCES concepts(id) ON DELETE CASCADE,
    to_concept_id INTEGER REFERENCES concepts(id) ON DELETE CASCADE,
    relationship_type VARCHAR(100) NOT NULL,
    relationship_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_relationships_from ON relationships(from_concept_id);
CREATE INDEX IF NOT EXISTS idx_relationships_to ON relationships(to_concept_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type);

-- ============================================
-- MARKET REGIMES TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS market_regimes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    detection_criteria JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- ASSET CLASSES TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS asset_classes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- BACKTEST RESULTS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(id) ON DELETE CASCADE,
    asset_class VARCHAR(50),
    market_regime VARCHAR(100),
    timeframe VARCHAR(20),
    start_date DATE,
    end_date DATE,
    total_trades INTEGER,
    win_rate DECIMAL(5,2),
    profit_factor DECIMAL(5,2),
    max_drawdown DECIMAL(5,2),
    sharpe_ratio DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON backtest_results(strategy_id);
CREATE INDEX IF NOT EXISTS idx_backtest_regime ON backtest_results(market_regime);

-- ============================================
-- KNOWLEDGE CHUNKS TABLE (for RAG retrieval)
-- ============================================
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id SERIAL PRIMARY KEY,
    concept_id INTEGER REFERENCES concepts(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_source VARCHAR(255),
    chunk_page INTEGER,
    embedding vector(1024),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunks_concept ON knowledge_chunks(concept_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops);

-- ============================================
-- SEED DATA
-- ============================================

INSERT INTO asset_classes (name, description) VALUES
('Stock', 'Equity securities'),
('Option', 'Derivative contracts'),
('Future', 'Futures contracts'),
('Forex', 'Foreign exchange'),
('Crypto', 'Cryptocurrencies'),
('ETF', 'Exchange-traded funds')
ON CONFLICT (name) DO NOTHING;

INSERT INTO market_regimes (name, description, detection_criteria) VALUES
('Trending', 'Sustained directional movement', '{"adx": ">25", "ma_alignment": "ordered"}'),
('Range-bound', 'Sideways consolidation', '{"adx": "<20", "bollinger_squeeze": true}'),
('High Volatility', 'Large price swings', '{"atr_percent": ">3", "vix": ">20"}'),
('Low Volatility', 'Compressed price action', '{"atr_percent": "<1", "vix": "<15"}')
ON CONFLICT (name) DO NOTHING;
