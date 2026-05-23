"""Bear researcher using base class."""
from quantify.agents.base_researcher import BearResearcher


def create_bear_researcher(llm):
    """Create bear researcher node."""
    return BearResearcher(llm)
