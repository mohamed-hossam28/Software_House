"""
OpenAI-compatible LLM providers.

All providers that speak the OpenAI chat-completions protocol are served by a
single OpenAICompatibleProvider class configured from PROVIDER_SPECS: one
spec per provider name holds its base_url, env-key fallback chain, default
model, and the names under which config.default_provider selects it.

Adding a provider = adding one spec entry; no new file.
"""
import os
from typing import Any, Callable, Optional

from providers.llm.base import BaseLLMProvider, LLMResponse

PROVIDER_SPECS: dict[str, dict[str, Any]] = {
    "open_router": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_keys": ["OPENAI_API_KEY", "open_router_api"],
        "default_model": "openai/gpt-oss-120b:free",
        "aliases": {"open_router", "openrouter"},
    },
    "orca_router": {
        "base_url": "https://api.orcarouter.ai/v1",
        "env_keys": ["orca_router_api"],
        "default_model": "qwen/qwen3.8-27b-free",
        "aliases": {"orca_router"},
    },
    "token_router": {
        "base_url": "https://api.tokenrouter.com/v1",
        "env_keys": ["token_router_api"],
        "default_model": "deepseek/deepseek-v4-pro-0813-free",
        "aliases": {"token_router"},
    },
    "nara": {
        "base_url": "https://router.bynara.id/v1",
        "env_keys": ["NARA_API_KEY", "nara_api"],
        "default_model": "deepseek-3.2",
        "aliases": {"nara"},
    },
    "apinex": {
        "base_url": "https://api.apinex.bond/v1",
        "env_keys": ["APINEX_API_KEY", "apinex_api"],
        "default_model": "free/glm-5.3-flash",
        "aliases": {"apinex"},
    },
}


class OpenAICompatibleProvider(BaseLLMProvider):
    """One provider class for every OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "",
        env_keys: tuple[str, ...] | list[str] = (),
        default_model: str = "",
        provider_name: str = "",
        aliases: set[str] | frozenset[str] = frozenset(),
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )
        self.base_url = base_url
        self.default_model = default_model
        self._env_keys = list(env_keys)
        self._aliases = set(aliases) or {provider_name}
        self._provider_name = provider_name

        from core.config import settings
        default_provider = settings.default_provider.lower().strip()
        key = (
            api_key
            or (settings.api_key if default_provider in self._aliases else None)
            or next((os.environ[k] for k in self._env_keys if os.environ.get(k)), None)
        )
        self.client = OpenAI(
            base_url=base_url,
            api_key=key,
            timeout=180.0,
        )

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        model: str = "",
        max_tokens: int = 8192,
        temperature: float = 0.2,
        stream: bool = False,
        chunk_callback: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        model = model or self.default_model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        if stream:
            try:
                stream_resp = self.client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                    stream_options={"include_usage": True},
                    messages=messages,
                )
            except Exception:
                stream_resp = self.client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                    messages=messages,
                )

            chunks: list[str] = []
            input_tokens = 0
            output_tokens = 0
            for chunk in stream_resp:
                if getattr(chunk, "usage", None):
                    input_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                    output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0
                if chunk.choices:
                    delta = chunk.choices[0].delta
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
                text=full_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                provider=self.provider_name,
            )

        response = self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=response.model,
            provider=self.provider_name,
        )


def build_provider(name: str, api_key: str | None = None) -> OpenAICompatibleProvider:
    """Instantiate the shared provider class for a registered spec name."""
    name = name.lower().strip()
    if name not in PROVIDER_SPECS:
        raise ValueError(f"Unknown OpenAI-compatible provider: '{name}'")
    spec = PROVIDER_SPECS[name]
    return OpenAICompatibleProvider(api_key=api_key, provider_name=name, **spec)
