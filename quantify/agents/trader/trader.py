"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal.

Knowledge Base Integration:
    Automatically injects relevant knowledge from financial books into prompts.
"""

from __future__ import annotations
import functools
from pathlib import Path

from langchain_core.messages import AIMessage

from quantify.agents.schemas import TraderProposal, render_trader_proposal
from quantify.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from quantify.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)

# Lazy-load knowledge service
_kb_service = None

def _get_kb_service():
    """Lazy-load knowledge service for auto-injection."""
    global _kb_service
    if _kb_service is not None:
        return _kb_service

    try:
        current_file = Path(__file__).resolve()
        kb_path = current_file.parent.parent.parent.parent / "knowledge_base"
        if kb_path.exists() and (kb_path / "service.py").exists():
            import sys
            if str(kb_path) not in sys.path:
                sys.path.insert(0, str(kb_path))
            from service import KnowledgeService
            _kb_service = KnowledgeService(
                index_path=str(kb_path / "data" / "knowledge_base"),
                auto_load=True,
            )
    except Exception:
        _kb_service = None

    return _kb_service


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        asset_type = state.get("asset_type", "stock")
        instrument_context = build_instrument_context(company_name, asset_type)
        investment_plan = state["investment_plan"]

        # Build knowledge section
        knowledge_section = ""
        kb = _get_kb_service()
        if kb:
            try:
                context = kb.get_context(f"{company_name} trading analysis", k=2)
                if context and len(context.strip()) > 0:
                    knowledge_section = f"\n\n== Relevant Financial Knowledge ==\nUse the following knowledge from financial books and research to inform your trading decision:\n\n{context}\n\n== End Knowledge =="
            except Exception:
                pass  # Silently fail - knowledge is optional

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    "Anchor your reasoning in the analysts' reports and the research plan."
                    + get_language_instruction()
                    + knowledge_section
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on a comprehensive analysis by a team of analysts, here is an investment "
                    f"plan tailored for {company_name}. {instrument_context} This plan incorporates "
                    f"insights from current technical market trends, macroeconomic indicators, and "
                    f"social media sentiment. Use this plan as a foundation for evaluating your next "
                    f"trading decision.\n\nProposed Investment Plan: {investment_plan}\n\n"
                    f"Leverage these insights to make an informed and strategic decision."
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
