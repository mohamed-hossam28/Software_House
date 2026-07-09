import anthropic
import os 
from providers.llm.base import BaseLLMProvider, LLMResponse
 
 
class AnthropicProvider(BaseLLMProvider):
 
    def __init__(self, api_key: str | None = None):
        from core.config import settings
        # api_key=None → SDK reads ANTHROPIC_API_KEY from environment automatically
        key = (
            api_key 
            or (settings.api_key if settings.default_provider.lower().strip() == "anthropic" else None)
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.client = anthropic.Anthropic(api_key=key)
 
    @property
    def provider_name(self) -> str:
        return "anthropic"
 
    def complete(
        self,
        system_prompt: str,
        user_message:  str,
        model:         str   = "claude-sonnet-4-6",
        max_tokens:    int   = 8192,
        temperature:   float = 0.2,
    ) -> LLMResponse:
 
        response = self.client.messages.create(
            model       = model,
            max_tokens  = max_tokens,
            temperature = temperature,
            system      = system_prompt,
            messages    = [{"role": "user", "content": user_message}],
        )
 
        # Map Anthropic's response format → our shared LLMResponse
        return LLMResponse(
            text          = response.content[0].text,
            input_tokens  = response.usage.input_tokens,
            output_tokens = response.usage.output_tokens,
            model         = response.model,
            provider      = self.provider_name,
        )
