"""
Configuration management for git-summarize.

Handles environment variables, config files, and default settings.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import set_key


class ProviderSettings(BaseSettings):
    """Settings for a specific AI provider."""

    host: Optional[str] = None
    model: Optional[str] = None


class OllamaSettings(ProviderSettings):
    """Ollama-specific settings."""

    host: str = "http://localhost:11434"
    model: str = "llama2"


class ClaudeSettings(ProviderSettings):
    """Claude-specific settings."""

    model: str = "claude-3-sonnet-20240229"


class OpenAISettings(ProviderSettings):
    """OpenAI-specific settings."""

    model: str = "gpt-4-turbo-preview"


class GeminiSettings(ProviderSettings):
    """Gemini-specific settings."""

    model: str = "gemini-1.5-flash"


class Config(BaseSettings):
    """
    Main configuration for git-summarize.

    Loads from environment variables and config file.
    Priority: Environment variables > Config file > Defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="GCM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # AI Provider settings
    provider: str = Field(
        default="claude",
        description="Default AI provider (claude, openai, ollama, gemini)",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model name (overrides provider default)",
    )
    num_suggestions: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of commit message suggestions to generate",
    )

    # API Keys
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key for Claude",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key",
    )
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key",
    )

    # Ollama settings
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama server host",
    )
    ollama_model: str = Field(
        default="llama2",
        description="Ollama model name",
    )

    # Behavior settings
    auto: bool = Field(
        default=False,
        description="Auto-select first suggestion without interaction",
    )
    preview: bool = Field(
        default=False,
        description="Preview suggestions without committing",
    )
    apply: bool = Field(
        default=False,
        description="Apply first suggestion directly",
    )
    push: bool = Field(
        default=False,
        description="Push after commit (with branch selection)",
    )
    no_add: bool = Field(
        default=False,
        description="Skip git add (only stage specific files)",
    )

    # Performance and token usage settings
    diff_context_lines: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Number of context lines to include in Git diffs (0 for minimal)",
    )
    diff_max_length: int = Field(
        default=10000,
        ge=500,
        le=100000,
        description="Maximum characters for the combined diff text",
    )

    # Provider-specific configurations
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    claude: ClaudeSettings = Field(default_factory=ClaudeSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for specified provider."""
        if provider == "claude":
            return (
                self.anthropic_api_key
                or os.getenv("GCM_ANTHROPIC_API_KEY")
                or os.getenv("ANTHROPIC_API_KEY")
            )
        elif provider == "openai":
            return (
                self.openai_api_key
                or os.getenv("GCM_OPENAI_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )
        elif provider == "gemini":
            return (
                self.gemini_api_key
                or os.getenv("GCM_GEMINI_API_KEY")
                or os.getenv("GEMINI_API_KEY")
            )
        return None  # Ollama doesn't need a key

    def get_model(self, provider: str) -> str:
        """Get model name for specified provider."""
        if self.model:
            return self.model

        if provider == "claude":
            return self.claude.model
        elif provider == "openai":
            return self.openai.model
        elif provider == "ollama":
            return self.ollama_model
        elif provider == "gemini":
            return self.gemini.model

        raise ValueError(f"Unknown provider: {provider}")

    def get_ollama_host(self) -> str:
        """Get Ollama host URL."""
        return self.ollama_host or self.ollama.host or "http://localhost:11434"

    def is_configured(self) -> bool:
        """Check if the current provider is properly configured with an API key."""
        if self.provider == "ollama":
            return True
        return bool(self.get_api_key(self.provider))

    def save_to_env(self, provider: str, api_key: str, model: Optional[str] = None) -> None:
        """Save configuration to the .env file in the current directory."""
        env_path = self.get_env_path()

        # Update or create the .env file
        set_key(str(env_path), "GCM_PROVIDER", provider)
        if provider == "claude":
            set_key(str(env_path), "GCM_ANTHROPIC_API_KEY", api_key)
        elif provider == "openai":
            set_key(str(env_path), "GCM_OPENAI_API_KEY", api_key)
        elif provider == "gemini":
            set_key(str(env_path), "GCM_GEMINI_API_KEY", api_key)

        if model:
            set_key(str(env_path), "GCM_MODEL", model)

    @classmethod
    def get_env_path(cls) -> Path:
        """Get the path to the .env file."""
        env_file = Path(".env")
        if not env_file.exists():
            # Try package directory
            package_dir = Path(__file__).parent.parent.parent
            env_file = package_dir / ".env"
        
        # If still doesn't exist, use current directory
        if not env_file.exists():
            env_file = Path(".env")
            
        return env_file

    @classmethod
    def load(cls) -> "Config":
        """
        Load configuration from environment and config file.

        Config file location: ~/.git-summarize/config.toml
        """
        env_file = cls.get_env_path()

        if env_file.exists():
            print(f"Loading config from {env_file}")
            return cls(_env_file=str(env_file), _env_file_encoding="utf-8")

        return cls()


def get_config() -> Config:
    """Get the global configuration instance."""
    return Config.load()
