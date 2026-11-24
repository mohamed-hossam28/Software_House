"""The agents' LLM call always streams; a provider that cannot is a bug, not a fallback."""
import pytest

from agents.Requirements import RequirementsAgent
from providers.llm.base import BaseLLMProvider, LLMResponse


class NoStreamingProvider(BaseLLMProvider):
    """A provider whose complete() does not take the streaming arguments."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "no-streaming"

    def complete(self, system_prompt, user_message, model, max_tokens=8192, temperature=0.2):
        self.calls += 1
        return LLMResponse(text="{}")


def test_the_llm_call_does_not_retry_without_streaming():
    provider = NoStreamingProvider()
    agent = RequirementsAgent(provider=provider, retries=1, backoff=0.0)

    with pytest.raises(TypeError):
        agent.llm_call("Build a todo app.")
    assert provider.calls == 0
