import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from core.config import settings
from providers.llm.llmFactory import LLMFactory
from providers.llm.openai_compatible import PROVIDER_SPECS, OpenAICompatibleProvider


@pytest.fixture
def capture_openai(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: captured.update(kwargs))
    return captured


class TestFactoryRegistry:
    def test_all_names_registered(self):
        assert set(LLMFactory.available()) == {
            "anthropic", "gemini",
            "open_router", "orca_router", "token_router", "nara", "apinex",
            "agent_router",
        }

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMFactory.get("does_not_exist")


class TestProviderConstruction:
    def test_per_provider_base_url(self, capture_openai, monkeypatch):
        expected = {
            "open_router": "https://openrouter.ai/api/v1",
            "orca_router": "https://api.orcarouter.ai/v1",
            "token_router": "https://api.tokenrouter.com/v1",
            "nara": "https://router.bynara.id/v1",
            "apinex": "https://api.apinex.bond/v1",
            "agent_router": "https://agentrouter.org/v1",
        }
        monkeypatch.delenv("APINEX_API_KEY", raising=False)
        monkeypatch.delenv("AGENT_ROUTER_API_KEY", raising=False)
        for name, base_url in expected.items():
            provider = LLMFactory.get(name, api_key="explicit-key")
            assert provider.provider_name == name
            assert capture_openai["base_url"] == base_url
            assert capture_openai["api_key"] == "explicit-key"
            assert capture_openai["timeout"] == 180.0

    def test_settings_api_key_used_when_default_provider_matches(self, capture_openai, monkeypatch):
        monkeypatch.delenv("APINEX_API_KEY", raising=False)
        monkeypatch.setattr(settings, "default_provider", "apinex", raising=False)
        monkeypatch.setattr(settings, "apinex_api", "settings-key", raising=False)
        LLMFactory.get("apinex")
        assert capture_openai["api_key"] == "settings-key"

    def test_env_key_fallback_chain(self, capture_openai, monkeypatch):
        monkeypatch.setattr(settings, "default_provider", "open_router", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("apinex_api", raising=False)
        monkeypatch.setattr(settings, "apinex_api", None, raising=False)
        monkeypatch.setenv("APINEX_API_KEY", "env-key")
        LLMFactory.get("apinex")
        assert capture_openai["api_key"] == "env-key"

    def test_settings_key_beats_env_key(self, capture_openai, monkeypatch):
        monkeypatch.delenv("APINEX_API_KEY", raising=False)
        monkeypatch.setattr(settings, "default_provider", "apinex", raising=False)
        monkeypatch.setattr(settings, "apinex_api", "settings-key", raising=False)
        monkeypatch.setenv("apinex_api", "env-key")
        LLMFactory.get("apinex")
        assert capture_openai["api_key"] == "settings-key"

    def test_openrouter_alias_matches_settings(self, capture_openai, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("open_router_api", raising=False)
        monkeypatch.setattr(settings, "default_provider", "openrouter", raising=False)
        monkeypatch.setattr(settings, "open_router_api", "alias-key", raising=False)
        LLMFactory.get("open_router")
        assert capture_openai["api_key"] == "alias-key"


class TestConfigIntegration:
    def test_model_name_fallback_for_apinex(self, monkeypatch):
        monkeypatch.setattr(settings, "default_model", None, raising=False)
        monkeypatch.setattr(settings, "default_provider", "apinex", raising=False)
        assert settings.model_name == "free/glm-5.3-flash"

    def test_api_key_property_for_apinex(self, monkeypatch):
        monkeypatch.setattr(settings, "default_provider", "apinex", raising=False)
        monkeypatch.setattr(settings, "apinex_api", "prop-key", raising=False)
        assert settings.api_key == "prop-key"

    def test_no_nvidia_endpoint_registered(self):
        assert "glm" not in PROVIDER_SPECS
        assert "glm" not in LLMFactory.available()
