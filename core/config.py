from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices
from typing import Optional
import os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys with fallbacks/aliases
    openai_api: Optional[str] = Field(None, validation_alias=AliasChoices("openai_api_key", "OPENAI_API_KEY", "open_router_api"))
    
    gemini_api: Optional[str] = Field(None, validation_alias=AliasChoices("gemini_api", "gemini_api_key", "GEMINI_API_KEY"))
    
    qrok_api: Optional[str] = Field(None, validation_alias=AliasChoices("qrok_api", "QROK_API"))
    groq_api: Optional[str] = Field(None, validation_alias=AliasChoices("groq_api_key", "GROQ_API_KEY", "qrok_api"))
    
    anthropic_api: Optional[str] = Field(None, validation_alias=AliasChoices("anthropic_api_key", "ANTHROPIC_API_KEY"))
    nvidia_api: Optional[str] = Field(None, validation_alias=AliasChoices("nvidia_api_key", "NVIDIA_API_KEY", "nvidia_api", "NVIDIA_API"))
    nara_api: Optional[str] = Field(None, validation_alias=AliasChoices("nara_api", "nara_api_key", "NARA_API_KEY"))

    # Provider and model settings
    # Support both DEFAULY_PROVIDER (as requested) and standard DEFAULT_PROVIDER
    default_provider: str = Field("openai", validation_alias=AliasChoices("DEFAULY_PROVIDER", "default_provider", "DEFAULT_PROVIDER"))
    default_model: Optional[str] = Field(None, validation_alias=AliasChoices("default_model", "DEFAULT_MODEL"))

    default_max_tokens: Optional[int] = Field(None, validation_alias=AliasChoices("DEFAULT_MAX_TOKENS", "default_max_tokens"))
    default_retries: Optional[int]    = Field(None, validation_alias=AliasChoices("DEFAULT_RETRIES", "default_retries"))
    default_backoff: Optional[float]  = Field(None, validation_alias=AliasChoices("DEFAULT_BACKOFF", "default_backoff"))

    @property
    def model_name(self) -> str:
        if self.default_model:
            return self.default_model
        # Fallback defaults based on provider
        provider = self.default_provider.lower().strip()
        if provider == "gemini":
            return "gemini-2.0-flash"
        elif provider == "anthropic":
            return "claude-sonnet-4-6"
        elif provider == "glm":
            return "z-ai/glm-5.2"
        elif provider=="openai":
            return "openai/gpt-oss-120b:free"
        elif provider == "nara":
            return "mistral-large"
        else:
            return "openai/gpt-oss-120b:free"

    @property
    def api_key(self) -> Optional[str]:
        provider = self.default_provider.lower().strip()
        if provider == "gemini":
            return self.gemini_api
        elif provider == "anthropic":
            return self.anthropic_api
        elif provider == "openai":
            return self.openai_api
        elif provider in ("groq", "qrok"):
            return self.groq_api
        elif provider == "glm":
            return self.nvidia_api
        elif provider == "nara":
            return self.nara_api
        return None
    

settings = Settings()
