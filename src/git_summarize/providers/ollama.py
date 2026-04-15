"""
Ollama provider for git-summarize.

Supports local AI models via Ollama server.
"""

import asyncio
from typing import Optional

import httpx

from git_summarize.providers.base import (
    AIProvider,
    GenerationRequest,
    GenerationResponse,
    ProviderError,
    ProviderRegistry,
)


@ProviderRegistry.register("ollama")
class OllamaProvider(AIProvider):
    """
    Ollama local AI provider.

    Supports any model available in local Ollama installation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        host: str = "http://localhost:11434",
    ):
        """
        Initialize Ollama provider.

        Args:
            api_key: Not used (Ollama doesn't require authentication)
            model: Model name to use
            host: Ollama server host URL
        """
        super().__init__(api_key, model)
        self.host = host.rstrip("/")

    @property
    def name(self) -> str:
        return "Ollama"

    @property
    def default_model(self) -> str:
        return "llama3.2"

    @property
    def requires_api_key(self) -> bool:
        return False

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using Ollama."""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "model": self.model,
                    "prompt": request.prompt,
                    "stream": False,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens,
                    },
                }

                if request.system_prompt:
                    payload["system"] = request.system_prompt

                response = await client.post(
                    f"{self.host}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                return GenerationResponse(
                    text=data.get("response", ""),
                    model=self.model,
                    usage={
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                    },
                    raw_response=data,
                )

        except httpx.ConnectError as e:
            raise ProviderError(
                f"Cannot connect to Ollama server at {self.host}. "
                "Make sure Ollama is running: ollama serve",
                provider="ollama",
                original_error=e,
            )
        except Exception as e:
            raise ProviderError(
                f"Failed to generate with Ollama: {str(e)}",
                provider="ollama",
                original_error=e,
            )

    async def check_availability(self) -> bool:
        """Check if Ollama server is available and reachable."""
        try:
            # We try both the configured host and 127.0.0.1 as a fallback
            hosts_to_try = [self.host]
            if "localhost" in self.host:
                hosts_to_try.append(self.host.replace("localhost", "127.0.0.1"))
            elif "127.0.0.1" in self.host:
                hosts_to_try.append(self.host.replace("127.0.0.1", "localhost"))

            for host in hosts_to_try:
                try:
                    async with httpx.AsyncClient(timeout=3.0) as client:
                        response = await client.get(f"{host}/api/tags")
                        if response.status_code == 200:
                            # Update self.host if we found a better one
                            self.host = host
                            return True
                except (httpx.ConnectError, httpx.TimeoutException):
                    continue
            
            return False

        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """List available models on Ollama server."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.host}/api/tags")
                response.raise_for_status()
                data = response.json()
                return [m.get("name", "unknown") for m in data.get("models", [])]
        except Exception:
            return []
