from providers.llm.base import BaseLLMProvider
 
 
# Registry maps provider name → class
# Add a new provider here when you add a new file
_REGISTRY: dict[str, str] = {
    "anthropic": "providers.llm.anthropic.AnthropicProvider",
    "openai":    "providers.llm.openai.OpenAIProvider",
    "gemini":    "providers.llm.gemini.GeminiProvider",
}
 
 
class LLMFactory:
 
    @staticmethod
    def get(provider_name: str, api_key: str | None = None) -> BaseLLMProvider:
        """
        Returns an instantiated LLM provider.
 
        Args:
            provider_name:  one of "anthropic", "openai", "gemini"
            api_key:        optional — if None, each provider reads from env
 
        Raises:
            ValueError:  if the provider name is not in the registry
        """
        name = provider_name.lower().strip()
 
        if name not in _REGISTRY:
            available = ", ".join(_REGISTRY.keys())
            raise ValueError(
                f"Unknown LLM provider: '{provider_name}'. "
                f"Available providers: {available}"
            )
 
        # Import the class dynamically so unused providers are never imported
        # e.g. if you only use Anthropic, openai and gemini packages are never touched
        module_path, class_name = _REGISTRY[name].rsplit(".", 1)
 
        import importlib
        module   = importlib.import_module(module_path)
        cls      = getattr(module, class_name)
 
        return cls(api_key=api_key)
 
    @staticmethod
    def available() -> list[str]:
        """Returns the list of registered provider names."""
        return list(_REGISTRY.keys())