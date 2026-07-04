from .base import BaseLLMProvider,LLMResponse

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str | None = None):
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
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
    ) -> LLMResponse:
 
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