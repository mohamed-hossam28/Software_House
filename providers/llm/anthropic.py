import anthropic
import os 
from providers.llm.base import BaseLLMProvider, LLMResponse
from typing import Callable, Optional
 
 
class AnthropicProvider(BaseLLMProvider):
 
    def __init__(self, api_key: str | None = None):
        from core.config import settings
        # api_key=None → SDK reads ANTHROPIC_API_KEY from environment automatically
        key = (
            api_key 
            or (settings.api_key if settings.default_provider.lower().strip() == "anthropic" else None)
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.client = anthropic.Anthropic(
            api_key=key,
            timeout=180.0,
        )
 
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
        stream:        bool  = False,
        chunk_callback: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
 
        if stream:
            chunks: list[str] = []
            input_tokens = 0
            output_tokens = 0
            with self.client.messages.stream(
                model       = model,
                max_tokens  = max_tokens,
                temperature = temperature,
                system      = system_prompt,
                messages    = [{"role": "user", "content": user_message}],
            ) as stream_resp:
                for text in stream_resp.text_stream:
                    chunks.append(text)
                    if chunk_callback:
                        chunk_callback(text)
                # Get final message for usage stats
                final = stream_resp.get_final_message()
                input_tokens = final.usage.input_tokens
                output_tokens = final.usage.output_tokens

            return LLMResponse(
                text          = "".join(chunks),
                input_tokens  = input_tokens,
                output_tokens = output_tokens,
                model         = model,
                provider      = self.provider_name,
            )

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
