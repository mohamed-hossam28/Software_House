import os
from providers.llm.base import BaseLLMProvider, LLMResponse
from google import genai 
# pyrefly: ignore [missing-import]
from google.genai import types
from typing import Callable, Optional

class GeminiProvider(BaseLLMProvider):
 
    def __init__(self, api_key: str | None = None):
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package not installed. Run: pip install google-genai"
            )
        from core.config import settings
        
        key = (
            api_key 
            or (settings.api_key if settings.default_provider.lower().strip() == "gemini" else None)
            or settings.gemini_api
            or os.environ.get("GEMINI_API_KEY") 
            or os.environ.get("gemini_api")
        )
        self.client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=180_000),  # 180s in ms
        )
 
    @property
    def provider_name(self) -> str:
        return "gemini"
 
    def complete(
        self,
        system_prompt: str,
        user_message:  str,
        model:         str   = "gemini-2.0-flash",
        max_tokens:    int   = 8192,
        temperature:   float = 0.2,
        stream:        bool  = False,
        chunk_callback: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
 
        actual_model = model
        if actual_model.startswith("gemini/"):
            actual_model = actual_model[len("gemini/"):]

        config = types.GenerateContentConfig(
            system_instruction = system_prompt,
            max_output_tokens  = max_tokens,
            temperature        = temperature,
        )

        if stream:
            chunks: list[str] = []
            input_tokens = 0
            output_tokens = 0
            for chunk in self.client.models.generate_content_stream(
                model    = actual_model,
                contents = user_message,
                config   = config,
            ):
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    usage = chunk.usage_metadata
                    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
                if chunk.text:
                    chunks.append(chunk.text)
                    if chunk_callback:
                        chunk_callback(chunk.text)

            full_text = "".join(chunks)
            if input_tokens == 0:
                input_tokens = max(1, (len(system_prompt) + len(user_message)) // 4)
            if output_tokens == 0:
                output_tokens = max(1, len(full_text) // 4)

            return LLMResponse(
                text          = full_text,
                input_tokens  = input_tokens,
                output_tokens = output_tokens,
                model         = actual_model,
                provider      = self.provider_name,
            )

        response = self.client.models.generate_content(
            model    = actual_model,
            contents = user_message,
            config   = config,
        )
 
        usage = response.usage_metadata
        return LLMResponse(
            text          = response.text,
            input_tokens  = getattr(usage, "prompt_token_count",     0),
            output_tokens = getattr(usage, "candidates_token_count", 0),
            model         = actual_model,
            provider      = self.provider_name,
        )