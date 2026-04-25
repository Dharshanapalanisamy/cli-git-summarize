"""
Google Gemini provider for git-summarize.

Supports Gemini Pro models via Google AI API.
"""

import asyncio
from typing import Optional

from git_summarize.providers.base import (
    AIProvider,
    GenerationRequest,
    GenerationResponse,
    ProviderError,
    ProviderRegistry,
)


@ProviderRegistry.register("gemini")
class GeminiProvider(AIProvider):
    """
    Google Gemini AI provider.

    Supports Gemini 1.5 Flash and Pro models.
    Free tier available: https://ai.google.dev/pricing
    """

    @property
    def name(self) -> str:
        return "Gemini"

    @property
    def default_model(self) -> str:
        return "gemini-2.5-flash"

    @property
    def requires_api_key(self) -> bool:
        return True

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using Gemini."""
        try:
            client = self._get_client()

            # Build the full prompt with optional system prompt
            full_prompt = request.prompt
            if request.system_prompt:
                full_prompt = f"{request.system_prompt}\n\n{request.prompt}"

            # Use asyncio.to_thread to run synchronous API in async context
            response = await asyncio.to_thread(
                self._generate_with_gemini,
                client,
                full_prompt,
                request.max_tokens,
                request.temperature,
            )

            # Extract usage info if available
            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                    "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
                }

            return GenerationResponse(
                text=response.text,
                model=self.model,
                usage=usage if usage else None,
                raw_response={"text": response.text},
            )

        except Exception as e:
            raise ProviderError(
                f"Failed to generate with Gemini: {str(e)}",
                provider="gemini",
                original_error=e,
            )

    def _generate_with_gemini(self, client, prompt: str, max_tokens: int, temperature: float):
        """Generate text using Gemini SDK (synchronous)."""
        from google import genai
        
        # Configure generation parameters
        config = genai.types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        return response

    async def check_availability(self) -> bool:
        """Check if Gemini API is available."""
        if not self.api_key:
            return False

        try:
            client = self._get_client()
            from google import genai

            # Try a minimal request to check connectivity
            await asyncio.to_thread(
                client.models.generate_content,
                model=self.model,
                contents=".",
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=1,
                    temperature=0,
                ),
            )
            return True
        except Exception as e:
            print(f"\n[DEBUG] Gemini connection failed: {str(e)}")
            return False

    def _get_client(self):
        """Get Google Gen AI client instance."""
        from google import genai

        if not self.api_key:
            raise ProviderError(
                "Gemini API key not provided. Set GEMINI_API_KEY or use --api-key flag.",
                provider="gemini",
            )

        return genai.Client(api_key=self.api_key)
