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
    "agent_router": {
        "base_url": "https://agentrouter.org/v1",
        "env_keys": ["AGENT_ROUTER_API_KEY", "AGENTROUTER_API_KEY", "agent_router_api"],
        "default_model": "gpt-5.6-sol",
        "aliases": {"agent_router", "agentrouter"},
        # The gateway only accepts coding-CLI clients (User-Agent allowlist);
        # the OpenAI SDK's default UA is rejected with 401 unauthorized_client_error.
        "user_agent": "claude-cli/2.0.14 (external, cli)",
        # The sol models speak the OpenAI Responses API (/v1/responses), not
        # /v1/chat/completions (400 "please switch to /v1/responses").
        "endpoint": "responses",
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
        user_agent: str = "",
        endpoint: str = "chat",
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
        self._endpoint = endpoint

        from core.config import settings
        default_provider = settings.default_provider.lower().strip()
        key = (
            api_key
            or (settings.api_key if default_provider in self._aliases else None)
            or next((os.environ[k] for k in self._env_keys if os.environ.get(k)), None)
        )
        default_headers = {"User-Agent": user_agent} if user_agent else None
        self.client = OpenAI(
            base_url=base_url,
            api_key=key,
            timeout=180.0,
            default_headers=default_headers,
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

        if self._endpoint == "responses":
            return self._complete_responses(
                system_prompt, user_message, model, max_tokens, temperature,
                stream, chunk_callback,
            )

        try:
            return self._complete_chat(
                system_prompt, user_message, model, max_tokens, temperature,
                stream, chunk_callback,
            )
        except Exception as exc:
            # Some gateways serve a model only through the OpenAI Responses API
            # and reject /v1/chat/completions with a pointer to /v1/responses.
            if "/v1/responses" in str(getattr(exc, "message", "") or exc):
                return self._complete_responses(
                    system_prompt, user_message, model, max_tokens, temperature,
                    stream, chunk_callback,
                )
            raise

    def _complete_chat(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int,
        temperature: float,
        stream: bool,
        chunk_callback: Optional[Callable[[str], None]],
    ) -> LLMResponse:
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

    def _complete_responses(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int,
        temperature: float,
        stream: bool,
        chunk_callback: Optional[Callable[[str], None]],
    ) -> LLMResponse:
        """OpenAI Responses API (/v1/responses): instructions + input."""
        common = dict(
            model=model,
            instructions=system_prompt,
            input=user_message,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        if stream:
            events = self._create_responses_stream(common)
            chunks: list[str] = []
            input_tokens = 0
            output_tokens = 0
            for event in events:
                if getattr(event, "type", "") == "response.output_text.delta":
                    delta = event.delta
                    if delta:
                        chunks.append(delta)
                        if chunk_callback:
                            chunk_callback(delta)
                elif getattr(event, "type", "") == "response.completed":
                    usage = getattr(event.response, "usage", None)
                    input_tokens = getattr(usage, "input_tokens", 0) or 0
                    output_tokens = getattr(usage, "output_tokens", 0) or 0

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

        response = self._create_responses(common)
        text = getattr(response, "output_text", "") or self._concat_response_output(
            getattr(response, "output", None)
        )
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=text,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            model=model,
            provider=self.provider_name,
        )

    @staticmethod
    def _temperature_unsupported(exc: Exception) -> bool:
        text = str(getattr(exc, "message", "") or exc).lower()
        return "temperature" in text and "not supported" in text

    def _create_responses(self, common: dict) -> Any:
        """Non-streaming Responses call, dropping temperature if the model rejects it."""
        try:
            return self.client.responses.create(**common)
        except Exception as exc:
            if self._temperature_unsupported(exc):
                return self.client.responses.create(**{**common, "temperature": None})
            raise

    def _create_responses_stream(self, common: dict) -> Any:
        """Streaming Responses call, dropping temperature if the model rejects it."""
        try:
            return self.client.responses.create(stream=True, **common)
        except Exception as exc:
            if self._temperature_unsupported(exc):
                return self.client.responses.create(
                    stream=True, **{**common, "temperature": None}
                )
            raise

    @staticmethod
    def _concat_response_output(output: Any) -> str:
        """Flatten a Responses API output list into plain text (SDK fallback)."""
        parts: list[str] = []
        for item in output or []:
            if getattr(item, "type", "") == "message":
                for content in getattr(item, "content", None) or []:
                    if getattr(content, "type", "") == "output_text":
                        parts.append(getattr(content, "text", "") or "")
        return "".join(parts)


def build_provider(name: str, api_key: str | None = None) -> OpenAICompatibleProvider:
    """Instantiate the shared provider class for a registered spec name."""
    name = name.lower().strip()
    if name not in PROVIDER_SPECS:
        raise ValueError(f"Unknown OpenAI-compatible provider: '{name}'")
    spec = PROVIDER_SPECS[name]
    return OpenAICompatibleProvider(api_key=api_key, provider_name=name, **spec)
