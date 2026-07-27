from .base import BaseLLMProvider,LLMResponse
import os
from typing import Callable, Optional

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str | None = None):
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )
        from core.config import settings
        key = (
            api_key 
            or (settings.api_key if settings.default_provider.lower().strip() == "openai" else None)
            or os.environ.get("OPENAI_API_KEY") 
            or os.environ.get("open_router_api")
        )
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            timeout=180.0,
        )

    
    @property
    def provider_name(self) -> str:
        return "openai"
 
    def complete(
        self,
        system_prompt: str,
        user_message:  str,
        model:         str   = "openai/gpt-oss-120b:free",
        max_tokens:    int   = 8192,
        temperature:   float = 0.2,
        stream:        bool  = False,
        chunk_callback: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
 
        if stream:
            # Streaming: collect chunks incrementally
            chunks: list[str] = []
            stream_resp = self.client.chat.completions.create(
                model       = model,
                max_tokens  = max_tokens,
                temperature = temperature,
                stream      = True,
                messages    = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
            )
            for chunk in stream_resp:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    chunks.append(delta.content)
                    if chunk_callback:
                        chunk_callback(delta.content)

            full_text = "".join(chunks)
            return LLMResponse(
                text          = full_text,
                input_tokens  = 0,
                output_tokens = 0,
                model         = model,
                provider      = self.provider_name,
            )

        response = self.client.chat.completions.create(
            model       = model,
            max_tokens  = max_tokens,
            temperature = temperature,
            messages    = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )
 
        return LLMResponse(
            text          = response.choices[0].message.content,
            input_tokens  = response.usage.prompt_tokens,
            output_tokens = response.usage.completion_tokens,
            model         = response.model,
            provider      = self.provider_name,
        )