import os
from providers.llm.base import BaseLLMProvider, LLMResponse
from google import genai 
# pyrefly: ignore [missing-import]
from google.genai import types

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
            or os.environ.get("GEMINI_API_KEY") 
            or os.environ.get("gemini_api")
        )
        self.client = genai.Client(api_key=key)
 
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
    ) -> LLMResponse:
 
        actual_model = model
        if actual_model.startswith("gemini/"):
            actual_model = actual_model[len("gemini/"):]

        response = self.client.models.generate_content(
            model    = actual_model,
            contents = user_message,
            config   = types.GenerateContentConfig(
                system_instruction = system_prompt,
                max_output_tokens  = max_tokens,
                temperature        = temperature,
            ),
        )
 
        usage = response.usage_metadata
        return LLMResponse(
            text          = response.text,
            input_tokens  = getattr(usage, "prompt_token_count",     0),
            output_tokens = getattr(usage, "candidates_token_count", 0),
            model         = actual_model,
            provider      = self.provider_name,
        )