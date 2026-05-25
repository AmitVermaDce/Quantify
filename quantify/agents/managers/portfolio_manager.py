"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.

Knowledge Base Integration:
    Automatically injects relevant knowledge from financial books into prompts.
"""

from __future__ import annotations

from quantify.agents.schemas import PortfolioDecision, render_pm_decision
from quantify.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from quantify.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)

# Lazy-load knowledge service
_kb_service_pm = None

def _get_kb_service_pm():
    """Lazy-load knowledge service for auto-injection."""
    global _kb_service_pm
    if _kb_service_pm is not None:
        return _kb_service_pm

    try:
        from pathlib import Path
        current_file = Path(__file__).resolve()
        kb_path = current_file.parent.parent.parent.parent / "knowledge_base"
        if kb_path.exists() and (kb_path / "service.py").exists():
            import sys
            if str(kb_path) not in sys.path:
                sys.path.insert(0, str(kb_path))
            from service import KnowledgeService
            _kb_service_pm = KnowledgeService(
                index_path=str(kb_path / "data" / "knowledge_base"),
                auto_load=True,
            )
    except Exception:
        _kb_service_pm = None

    return _kb_service_pm


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}"""

        # Auto-inject knowledge from knowledge base
        kb = _get_kb_service_pm()
        if kb:
            ticker = state.get("ticker", state.get("company_of_interest", "market"))
            try:
                context = kb.get_context(f"{ticker} investment analysis", k=3)
                if context and len(context.strip()) > 0:
                    prompt += f"\n\n== Relevant Financial Knowledge ==\nUse the following knowledge from financial books and research to inform your decision:\n\n{context}\n\n== End Knowledge ==\n"
            except Exception:
                pass  # Silently fail - knowledge is optional

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
