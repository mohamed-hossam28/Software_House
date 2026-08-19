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
    open_router_api: Optional[str] = Field(None, validation_alias=AliasChoices("open_router_api", "OPEN_ROUTER_API_KEY", "OPENAI_API_KEY"))
    
    orca_router_api: Optional[str] = Field(None, validation_alias=AliasChoices("orca_router_api", "orca_router_api_key", "ORCA_ROUTER_API"))
    
    token_router_api: Optional[str] = Field(None, validation_alias=AliasChoices("token_router_api", "token_router_api_key", "TOKEN_ROUTER_API"))
    
    gemini_api: Optional[str] = Field(None, validation_alias=AliasChoices("gemini_api", "gemini_api_key", "GEMINI_API_KEY"))
    
    qrok_api: Optional[str] = Field(None, validation_alias=AliasChoices("qrok_api", "QROK_API"))
    groq_api: Optional[str] = Field(None, validation_alias=AliasChoices("groq_api_key", "GROQ_API_KEY", "qrok_api"))
    
    anthropic_api: Optional[str] = Field(None, validation_alias=AliasChoices("anthropic_api_key", "ANTHROPIC_API_KEY"))
    nvidia_api: Optional[str] = Field(None, validation_alias=AliasChoices("nvidia_api_key", "NVIDIA_API_KEY", "nvidia_api", "NVIDIA_API"))
    nara_api: Optional[str] = Field(None, validation_alias=AliasChoices("nara_api", "nara_api_key", "NARA_API_KEY"))

    # Provider and model settings
    # Support both DEFAULY_PROVIDER (as requested) and standard DEFAULT_PROVIDER
    default_provider: str = Field("open_router", validation_alias=AliasChoices("DEFAULY_PROVIDER", "default_provider", "DEFAULT_PROVIDER"))
    default_model: Optional[str] = Field(None, validation_alias=AliasChoices("default_model", "DEFAULT_MODEL"))

    # Coding agent (backend / frontend / database) overrides — fall back to default when unset
    coding_provider: Optional[str] = Field(None, validation_alias=AliasChoices("CODING_PROVIDER", "coding_provider"))
    coding_model: Optional[str] = Field(None, validation_alias=AliasChoices("CODING_MODEL", "coding_model"))

    default_max_tokens: Optional[int] = Field(None, validation_alias=AliasChoices("DEFAULT_MAX_TOKENS", "default_max_tokens"))
    default_retries: Optional[int]    = Field(None, validation_alias=AliasChoices("DEFAULT_RETRIES", "default_retries"))
    default_backoff: Optional[float]  = Field(None, validation_alias=AliasChoices("DEFAULT_BACKOFF", "default_backoff"))

    # Integration REWORK loop cap (integration -> coding agent -> integration)
    rework_trial: Optional[int] = Field(None, validation_alias=AliasChoices("REWORK_TRIAL", "rework_trial"))

    # Per-file REWORK give-up threshold: a file that keeps failing past this
    # many integration rounds stops routing back (kept in the summary as
    # unresolved) so it cannot starve other files of global rounds.
    rework_file_max: Optional[int] = Field(None, validation_alias=AliasChoices("REWORK_FILE_MAX", "rework_file_max"))

    # Per-agent max token settings matching .env keys
    default_req_max_tokens: Optional[int]         = Field(None, validation_alias=AliasChoices("DEFAULT_Req_MAX_TOKENS", "DEFAULT_REQ_MAX_TOKENS", "default_req_max_tokens"))
    default_arch_max_tokens: Optional[int]        = Field(None, validation_alias=AliasChoices("DEFAULT_Arch_MAX_TOKENS", "DEFAULT_ARCH_MAX_TOKENS", "default_arch_max_tokens"))
    default_planner_max_tokens: Optional[int]     = Field(None, validation_alias=AliasChoices("DEFAULT_Planner_MAX_TOKENS", "DEFAULT_PLANNER_MAX_TOKENS", "default_planner_max_tokens"))
    default_db_max_tokens: Optional[int]          = Field(None, validation_alias=AliasChoices("DEFAULT_DB_MAX_TOKENS", "DEFAULT_DB_MAX_TOKENS", "default_db_max_tokens"))
    default_be_max_tokens: Optional[int]          = Field(None, validation_alias=AliasChoices("DEFAULT_BE_MAX_TOKENS", "DEFAULT_BE_MAX_TOKENS", "default_be_max_tokens"))
    default_fe_max_tokens: Optional[int]          = Field(None, validation_alias=AliasChoices("DEFAULT_FE_MAX_TOKENS", "DEFAULT_FE_MAX_TOKENS", "default_fe_max_tokens"))
    default_integration_max_tokens: Optional[int] = Field(None, validation_alias=AliasChoices("DEFAULT_Integration_MAX_TOKENS", "DEFAULT_INTEGRATION_MAX_TOKENS", "default_integration_max_tokens"))

    @property
    def req_max_tokens(self) -> int:
        return self.default_req_max_tokens or self.default_max_tokens or 2000

    @property
    def arch_max_tokens(self) -> int:
        return self.default_arch_max_tokens or self.default_max_tokens or 3000

    @property
    def planner_max_tokens(self) -> int:
        return self.default_planner_max_tokens or self.default_max_tokens or 5000

    @property
    def db_max_tokens(self) -> int:
        return self.default_db_max_tokens or self.default_max_tokens or 2000

    @property
    def be_max_tokens(self) -> int:
        return self.default_be_max_tokens or self.default_max_tokens or 3000

    @property
    def fe_max_tokens(self) -> int:
        return self.default_fe_max_tokens or self.default_max_tokens or 3000

    @property
    def integration_max_tokens(self) -> int:
        return self.default_integration_max_tokens or self.default_max_tokens or 1500


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
        elif provider in ("open_router", "openrouter"):
            return "openai/gpt-oss-120b:free"
        elif provider == "orca_router":
            return "qwen/qwen3.8-27b-free"
        elif provider == "token_router":
            return "deepseek/deepseek-v4-pro-0813-free"
        elif provider == "nara":
            return "mistral-large"
        else:
            return "openai/gpt-oss-120b:free"

    @property
    def coding_model_name(self) -> str:
        return self.coding_model or self.model_name

    @property
    def coding_provider_name(self) -> str:
        return self.coding_provider or self.default_provider

    @property
    def api_key(self) -> Optional[str]:
        provider = self.default_provider.lower().strip()
        if provider == "gemini":
            return self.gemini_api
        elif provider == "anthropic":
            return self.anthropic_api
        elif provider in ("open_router", "openrouter"):
            return self.open_router_api
        elif provider == "orca_router":
            return self.orca_router_api
        elif provider == "token_router":
            return self.token_router_api
        elif provider in ("groq", "qrok"):
            return self.groq_api
        elif provider == "glm":
            return self.nvidia_api
        elif provider == "nara":
            return self.nara_api
        return None
    

settings = Settings()
