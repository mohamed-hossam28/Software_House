"""The frontend agent's system prompt carries the frontend serving convention."""
from core.state import ArchData
from agents.frontend import FRONTEND_PROFILE


def test_system_prompt_requires_relative_asset_links():
    prompt = FRONTEND_PROFILE.system_prompt(ArchData(tech_stack={"frontend": "HTML + Vanilla JS"}))

    assert "relative" in prompt
    assert 'href="style.css"' in prompt
    assert "/static/" in prompt  # named as the forbidden form
