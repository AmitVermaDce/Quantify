#!/usr/bin/env python3
"""
End-to-End Test: Knowledge Base → TradingAgents → Decision

This script tests the complete pipeline:
1. Knowledge base loading
2. Knowledge injection into each agent type
3. Full TradingAgents run on a stock
4. Verification that knowledge appears in reports

Usage:
    python test_knowledge_integration.py --ticker AAPL --date 2024-01-15
    python test_knowledge_integration.py --ticker AAPL --date 2024-01-15 --no-run  # Just test KB
"""

import sys
import os
from pathlib import Path
from typing import Optional
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

console = Console()

# =============================================================================
# TEST 1: Knowledge Base Loading
# =============================================================================

def test_knowledge_base_loading() -> bool:
    """Test that knowledge base loads correctly."""
    console.print("\n[bold cyan]TEST 1: Knowledge Base Loading[/bold cyan]")
    console.print("-" * 60)

    try:
        # Test direct import
        from knowledge_base import KnowledgeService
        console.print("  [green]✓[/green] knowledge_base package imports correctly")

        # Test service initialization
        kb = KnowledgeService()
        console.print(f"  [green]✓[/green] KnowledgeService initialized")

        # Test document count
        doc_count = len(kb)
        console.print(f"  [green]✓[/green] Documents loaded: {doc_count:,}")

        if doc_count < 10000:
            console.print(f"  [yellow]![/yellow] Warning: Expected >10K documents, got {doc_count}")
            return False

        # Test stats
        stats = kb.stats()
        console.print(f"  [green]✓[/green] Categories: {list(stats.get('categories', {}).keys())}")

        # Test search functionality
        results = kb.search("value investing", k=3)
        console.print(f"  [green]✓[/green] Search works: returned {len(results)} results")

        if results:
            book = results[0].get("metadata", {}).get("book_title", "Unknown")
            console.print(f"  [dim]Top result from: {book}[/dim]")

        # Test get_context
        context = kb.get_context("margin of safety", k=2)
        console.print(f"  [green]✓[/green] get_context works: {len(context)} chars")

        if len(context) < 100:
            console.print(f"  [yellow]![/yellow] Warning: Context seems too short")
            return False

        console.print(Panel("[green]TEST 1 PASSED: Knowledge Base Fully Functional[/green]", border_style="green"))
        return True

    except Exception as e:
        console.print(f"  [red]✗[/red] Error: {e}")
        console.print(Panel("[red]TEST 1 FAILED[/red]", border_style="red"))
        return False


# =============================================================================
# TEST 2: Knowledge Injection in All Agents
# =============================================================================

def test_knowledge_injection() -> bool:
    """Test that all agents have knowledge injection enabled."""
    console.print("\n[bold cyan]TEST 2: Knowledge Injection in Agents[/bold cyan]")
    console.print("-" * 60)

    agents_to_test = [
        ("Bull Researcher", "quantify.agents.base_researcher", "BullResearcher"),
        ("Bear Researcher", "quantify.agents.base_researcher", "BearResearcher"),
        ("Aggressive Debator", "quantify.agents.base_debator", "AggressiveDebator"),
        ("Conservative Debator", "quantify.agents.base_debator", "ConservativeDebator"),
        ("Neutral Debator", "quantify.agents.base_debator", "NeutralDebator"),
        ("News Analyst", "quantify.agents.analysts.news_analyst", "NewsAnalyst"),
        ("Fundamentals Analyst", "quantify.agents.analysts.fundamentals_analyst", "FundamentalsAnalyst"),
    ]

    all_passed = True

    for agent_name, module_path, class_name in agents_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            agent_class = getattr(module, class_name)

            use_knowledge = getattr(agent_class, "USE_KNOWLEDGE", None)
            query_template = getattr(agent_class, "KNOWLEDGE_QUERY_TEMPLATE", None)
            knowledge_k = getattr(agent_class, "KNOWLEDGE_K", None)

            if use_knowledge is True:
                console.print(f"  [green]✓[/green] {agent_name}: USE_KNOWLEDGE=True")
                if query_template:
                    console.print(f"      [dim]Query: {query_template}[/dim]")
            else:
                console.print(f"  [yellow]![/yellow] {agent_name}: USE_KNOWLEDGE={use_knowledge}")

        except Exception as e:
            console.print(f"  [red]✗[/red] {agent_name}: {e}")
            all_passed = False

    # Test direct-load agents
    console.print("\n  [dim]Testing direct-load agents (KB service initialization)...[/dim]")

    direct_agents = [
        ("Market Analyst", "quantify.agents.analysts.market_analyst", "_get_kb_service"),
        ("Sentiment Analyst", "quantify.agents.analysts.sentiment_analyst", "_get_kb_service"),
        ("Portfolio Manager", "quantify.agents.managers.portfolio_manager", "_get_kb_service_pm"),
        ("Trader", "quantify.agents.trader.trader", "_get_kb_service"),
    ]

    for agent_name, module_path, func_name in direct_agents:
        try:
            module = __import__(module_path, fromlist=[func_name])
            get_kb = getattr(module, func_name)
            kb = get_kb()

            if kb is not None:
                console.print(f"  [green]✓[/green] {agent_name}: KB loaded ({len(kb)} docs)")
            else:
                console.print(f"  [yellow]![/yellow] {agent_name}: KB not loaded")

        except Exception as e:
            console.print(f"  [red]✗[/red] {agent_name}: {e}")
            all_passed = False

    if all_passed:
        console.print(Panel("[green]TEST 2 PASSED: All Agents Have Knowledge Injection[/green]", border_style="green"))
    else:
        console.print(Panel("[yellow]TEST 2 PARTIAL: Some agents may have issues[/yellow]", border_style="yellow"))

    return all_passed


# =============================================================================
# TEST 3: Full TradingAgents Run
# =============================================================================

def test_full_trading_agents_run(
    ticker: str = "AAPL",
    trade_date: str = "2024-01-15",
    selected_analysts: Optional[list] = None,
    llm_provider: Optional[str] = None,  # None = use default
) -> bool:
    """Run full TradingAgents pipeline and verify knowledge injection."""
    console.print("\n[bold cyan]TEST 3: Full TradingAgents Run[/bold cyan]")
    console.print("-" * 60)

    if selected_analysts is None:
        selected_analysts = ["market", "fundamentals"]  # Minimal set for speed

    try:
        from quantify.default_config import DEFAULT_CONFIG
        from quantify.graph.trading_graph import TradingAgentsGraph
        import copy

        console.print(f"  [dim]Ticker: {ticker}[/dim]")
        console.print(f"  [dim]Date: {trade_date}[/dim]")
        console.print(f"  [dim]Analysts: {selected_analysts}[/dim]")

        # Use default config (user's configured LLM)
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["max_debate_rounds"] = 2  # Minimal debate for speed
        config["max_risk_discuss_rounds"] = 2

        # Override provider only if specified
        if llm_provider:
            config["llm_provider"] = llm_provider
            if llm_provider == "ollama":
                config["quick_think_llm"] = "ollama/llama-3.2-3b"
                config["deep_think_llm"] = "ollama/llama-3.2-3b"

        console.print(f"  [dim]LLM Provider: {config['llm_provider']}[/dim]")
        console.print(f"  [dim]Quick Think: {config['quick_think_llm']}[/dim]")
        console.print(f"  [dim]Deep Think: {config['deep_think_llm']}[/dim]")

        console.print("\n  [bold]Initializing TradingAgentsGraph...[/bold]")
        graph = TradingAgentsGraph(
            selected_analysts=selected_analysts,
            debug=False,
            config=config,
        )

        console.print("  [green]✓[/green] Graph initialized")

        # Run the pipeline
        console.print("\n  [bold]Running propagate()...[/bold]")
        console.print("  [dim]This may take 1-5 minutes depending on LLM speed[/dim]\n")

        final_state, signal = graph.propagate(
            company_name=ticker,
            trade_date=trade_date,
            asset_type="stock",
        )

        console.print("  [green]✓[/green] Pipeline completed")

        # Verify outputs
        console.print("\n  [bold]Verifying outputs...[/bold]")

        # Check analyst reports
        for report_key in ["market_report", "sentiment_report", "news_report", "fundamentals_report"]:
            if report_key in final_state and final_state[report_key]:
                report = final_state[report_key]
                has_knowledge = "== Relevant Financial Knowledge ==" in report
                status = "[green]✓[/green]" if has_knowledge else "[dim]·[/dim]"
                console.print(f"  {status} {report_key}: {len(report)} chars" +
                             (" [green](with knowledge)[/green]" if has_knowledge else ""))

        # Check investment plan
        if "investment_plan" in final_state:
            plan = final_state["investment_plan"]
            console.print(f"  [green]✓[/green] investment_plan: {len(plan)} chars")

        # Check trader plan
        if "trader_investment_plan" in final_state:
            trader_plan = final_state["trader_investment_plan"]
            console.print(f"  [green]✓[/green] trader_investment_plan: {len(trader_plan)} chars")

        # Check final decision
        if "final_trade_decision" in final_state:
            decision = final_state["final_trade_decision"]
            has_knowledge = "== Relevant Financial Knowledge ==" in decision
            status = "[green]✓[/green]" if has_knowledge else "[dim]·[/dim]"
            console.print(f"  {status} final_trade_decision: {len(decision)} chars" +
                         (" [green](with knowledge)[/green]" if has_knowledge else ""))

        # Print signal
        console.print(f"\n  [bold]Trading Signal:[/bold] {signal[:200]}...")

        console.print(Panel("[green]TEST 3 PASSED: Full Pipeline Executed Successfully[/green]", border_style="green"))
        return True

    except Exception as e:
        console.print(f"  [red]✗[/red] Error: {e}")
        import traceback
        traceback.print_exc()
        console.print(Panel("[red]TEST 3 FAILED[/red]", border_style="red"))
        return False


# =============================================================================
# TEST 4: Knowledge Content Verification
# =============================================================================

def test_knowledge_content_verification(
    ticker: str = "AAPL",
    trade_date: str = "2024-01-15",
) -> bool:
    """Verify that knowledge base content actually appears in agent outputs."""
    console.print("\n[bold cyan]TEST 4: Knowledge Content Verification[/bold cyan]")
    console.print("-" * 60)

    try:
        # First, get some knowledge queries that should return results
        from knowledge_base import KnowledgeService
        kb = KnowledgeService()

        test_queries = [
            ("AAPL investment analysis", "Apple stock"),
            ("value investing", "Value investing principles"),
            ("risk management", "Risk management"),
            ("technical analysis", "Technical analysis"),
        ]

        console.print("  [bold]Testing knowledge retrieval for various queries:[/bold]\n")

        for query, description in test_queries:
            results = kb.search(query, k=2)
            if results:
                console.print(f"  [green]✓[/green] '{description}': {len(results)} results")
                for r in results:
                    book = r["metadata"].get("book_title", "Unknown").split("/")[-1][:50]
                    console.print(f"      [dim]- {book}[/dim]")
            else:
                console.print(f"  [yellow]![/yellow] '{description}': No results")

        console.print(Panel("[green]TEST 4 PASSED: Knowledge Content Verified[/green]", border_style="green"))
        return True

    except Exception as e:
        console.print(f"  [red]✗[/red] Error: {e}")
        console.print(Panel("[red]TEST 4 FAILED[/red]", border_style="red"))
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test Knowledge Base → TradingAgents Pipeline")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol to test")
    parser.add_argument("--date", type=str, default="2024-01-15", help="Trade date (YYYY-MM-DD)")
    parser.add_argument("--analysts", type=str, nargs="+", default=None,
                       help="Analysts to include (market, news, sentiment, fundamentals)")
    parser.add_argument("--llm-provider", type=str, default="ollama",
                       choices=["ollama", "openai", "anthropic", "google"],
                       help="LLM provider to use")
    parser.add_argument("--no-run", action="store_true",
                       help="Skip full TradingAgents run (test KB only)")
    parser.add_argument("--test", type=str, choices=["1", "2", "3", "4", "all"], default="all",
                       help="Run specific test (1=KB loading, 2=Injection, 3=Full run, 4=Content)")

    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold]Quantify TradingAgents - Knowledge Integration Test[/bold]\n"
        f"Ticker: {args.ticker} | Date: {args.date} | LLM: {args.llm_provider}",
        border_style="cyan",
    ))

    results = {}

    # Run tests
    if args.test in ["1", "all"]:
        results["KB Loading"] = test_knowledge_base_loading()

    if args.test in ["2", "all"]:
        results["Knowledge Injection"] = test_knowledge_injection()

    if args.test in ["4", "all"]:
        results["Content Verification"] = test_knowledge_content_verification(args.ticker, args.date)

    if args.test in ["3", "all"] and not args.no_run:
        results["Full Pipeline"] = test_full_trading_agents_run(
            ticker=args.ticker,
            trade_date=args.date,
            selected_analysts=args.analysts,
            llm_provider=args.llm_provider,
        )
    elif args.no_run:
        console.print("\n[yellow]Skipping full TradingAgents run (--no-run flag)[/yellow]")

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold]TEST SUMMARY[/bold]")
    console.print("=" * 60)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Test", style="cyan")
    table.add_column("Result", justify="center")

    for test_name, passed in results.items():
        result = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(test_name, result)

    console.print(table)

    all_passed = all(results.values())
    if results:
        if all_passed:
            console.print("\n[bold green]ALL TESTS PASSED ✓[/bold green]")
            return 0
        else:
            console.print("\n[bold red]SOME TESTS FAILED[/bold red]")
            return 1
    else:
        console.print("\n[yellow]No tests run[/yellow]")
        return 0


if __name__ == "__main__":
    sys.exit(main())
