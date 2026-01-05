from collections.abc import AsyncGenerator
from typing import Any

from ..integrations.openrouter import OpenRouterClient


class OpenRouterProcessor:
    """
    Manifold Processor back-ended by OpenRouter.
    Adapts parameters (temperature, models) based on manifold type.
    """

    def __init__(
        self,
        client: OpenRouterClient,
        model: str,
        temperature: float,
        system_prompt: str,
        name: str = "processor"
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.name = name

    async def process_chat(
        self,
        history: list[dict[str, str]],
        user_input: str,
        tools: list[dict[str, Any]] | None = None,
        api_key: str | None = None,
        include_reasoning: bool = False
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Process chat through this manifold configuration.
        Returns a stream generator.
        """
        messages = [{"role": "system", "content": self.system_prompt}] + history + [{"role": "user", "content": user_input}]

        async for chunk in self.client.stream_chat(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            tools=tools,
            api_key=api_key,
            include_reasoning=include_reasoning
        ):
            yield chunk
