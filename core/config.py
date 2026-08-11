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
