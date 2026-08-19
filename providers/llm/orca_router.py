from .base import BaseLLMProvider, LLMResponse
import os
from typing import Callable, Optional

class OrcaRouterProvider(BaseLLMProvider):
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
            or (settings.api_key if settings.default_provider.lower().strip() == "orca_router" else None)
            or os.environ.get("orca_router_api")
        )
        self.client = OpenAI(
            base_url="https://api.orcarouter.ai/v1",
            api_key=key,
            timeout=180.0,
        )

    
    @property
    def provider_name(self) -> str:
        return "orca_router"
 
    def complete(
        self,
        system_prompt: str,
        user_message:  str,
        model:         str  ,
        max_tokens:    int   = 8192,
        temperature:   float = 0.2,
        stream:        bool  = False,
        chunk_callback: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
 
        if stream:
            chunks: list[str] = []
            input_tokens = 0
            output_tokens = 0
            try:
                stream_resp = self.client.chat.completions.create(
                    model       = model,
                    max_tokens  = max_tokens,
                    temperature = temperature,
                    stream      = True,
                    stream_options = {"include_usage": True},
                    messages    = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_message},
                    ],
                )
            except Exception:
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
                if hasattr(chunk, "usage") and chunk.usage:
                    input_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                    output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0
                if chunk.choices:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        chunks.append(delta.content)
                        if chunk_callback:
                            chunk_callback(delta.content)

            full_text = "".join(chunks)
            if input_tokens == 0:
                input_tokens = max(1, (len(system_prompt) + len(user_message)) // 4)
            if output_tokens == 0:
                output_tokens = max(1, len(full_text) // 4)

            return LLMResponse(
                text          = full_text,
                input_tokens  = input_tokens,
                output_tokens = output_tokens,
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