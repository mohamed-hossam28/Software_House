from providers.llm.base import BaseLLMProvider

from providers.llm.openai_compatible import PROVIDER_SPECS, build_provider

# Dotted paths for providers with their own protocol (dynamic import keeps
# their SDKs unloaded unless used). OpenAI-compatible names are served by
# the single OpenAICompatibleProvider class — see providers/llm/openai_compatible.py.
_SDK_PROVIDERS: dict[str, str] = {
    "anthropic": "providers.llm.anthropic.AnthropicProvider",
    "gemini":    "providers.llm.gemini.GeminiProvider",
}


class LLMFactory:

    @staticmethod
    def get(provider_name: str, api_key: str | None = None) -> BaseLLMProvider:
        """
        Returns an instantiated LLM provider.

        Args:
            provider_name:  one of "anthropic", "gemini", or any OpenAI-compatible
                            name in providers.llm.openai_compatible.PROVIDER_SPECS
            api_key:        optional — if None, the provider reads from settings/env

        Raises:
            ValueError:  if the provider name is not registered
        """
        name = provider_name.lower().strip()

        if name in PROVIDER_SPECS:
            return build_provider(name, api_key=api_key)

        if name not in _SDK_PROVIDERS:
            available = ", ".join(_SDK_PROVIDERS.keys() | PROVIDER_SPECS.keys())
            raise ValueError(
                f"Unknown LLM provider: '{provider_name}'. "
                f"Available providers: {available}"
            )

        import importlib
        module_path, class_name = _SDK_PROVIDERS[name].rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        return cls(api_key=api_key)

    @staticmethod
    def available() -> list[str]:
        """Returns the list of registered provider names."""
        return list(_SDK_PROVIDERS.keys() | PROVIDER_SPECS.keys())
