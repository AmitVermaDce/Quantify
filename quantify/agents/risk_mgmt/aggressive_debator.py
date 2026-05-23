"""Aggressive debator using base class."""
from quantify.agents.base_debator import AggressiveDebator


def create_aggressive_debator(llm):
    """Create aggressive debator node."""
    return AggressiveDebator(llm)
