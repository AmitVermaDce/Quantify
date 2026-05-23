"""Neutral debator using base class."""
from quantify.agents.base_debator import NeutralDebator


def create_neutral_debator(llm):
    """Create neutral debator node."""
    return NeutralDebator(llm)
