import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any, cast

import aiohttp

logger = logging.getLogger(__name__)

class OpenRouterClient:
    """
    Client for interacting with the OpenRouter API.
    Supports streaming chat, tool calls, reasoning extraction, and embeddings.
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.info("OPENROUTER_API_KEY not set in environment. Expecting keys via API requests.")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://nexus-ai.local",  # Required by OpenRouter
            "X-Title": "NEXUS Cognitive Architecture",
            "Content-Type": "application/json"
        }

    async def fetch_models(self, api_key: str | None = None) -> list[dict[str, Any]]:
        """
        Fetch available models from OpenRouter.
        Filters for models that explicitly support tools via 'supported_parameters'.
        """
        request_key = api_key or self.api_key
        # We don't error immediately if no key, just try the request (OpenRouter might have public endpoint? No, usually needs Auth)
        # But let's check:
        headers = self.headers.copy()
        if request_key:
             headers["Authorization"] = f"Bearer {request_key}"

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.BASE_URL}/models", headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch models: {await response.text()}")
                    return []

                data = await response.json()
                models = data.get("data", [])

                tool_models = []
                for model in models:
                    # Strict check based on OpenRouter docs:
                    # check if 'tools' is in the 'supported_parameters' list
                    # Safely handle potential None value for supported_parameters
                    params = model.get("supported_parameters") or []
                    supports_tools = "tools" in params

                    if supports_tools:
                        tool_models.append({
                            "id": model["id"],
                            "name": model["name"],
                            "context_length": model["context_length"],
                            "pricing": model.get("pricing", {"prompt": "0", "completion": "0"}),
                            "description": model.get("description", ""),
                            "architecture": model.get("architecture", {}),
                            "supported_parameters": params,
                            "supports_tools": True
                        })

                logger.info(f"Fetched {len(models)} models, found {len(tool_models)} with tools support.")

                return tool_models

    async def get_embeddings(self, text: str, model: str = "openai/text-embedding-3-small", api_key: str | None = None) -> list[float]:
        """
        Generate embeddings for the given text.
        """

        request_key = api_key or self.api_key
        if not request_key:
             raise Exception("API Key missing")

        headers = self.headers.copy()
        headers["Authorization"] = f"Bearer {request_key}"

        payload = {
            "model": model,
            "input": text
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.BASE_URL}/embeddings",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Embedding failed: {error_text}")
                    raise Exception(f"Embedding failed: {error_text}")

                data = await response.json()
                return cast(list[float], data["data"][0]["embedding"])

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = "auto",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        api_key: str | None = None,
        include_reasoning: bool = False
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream chat completion with support for reasoning and tools.
        """

        # Use provided key or fall back to instance key
        request_key = api_key or self.api_key
        if not request_key:
             yield {"type": "error", "content": "API Key missing. Please configure in Settings."}
             return

        headers = self.headers.copy()
        headers["Authorization"] = f"Bearer {request_key}"

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream_options": {"include_usage": True}
        }

        # Use the new `reasoning` object parameter instead of legacy `include_reasoning`
        # For Gemini models, this uses max_tokens which maps to thinking_budget
        if include_reasoning:
            payload["reasoning"] = {
                "max_tokens": 2000,  # Reasonable default for reasoning budget
                "exclude": False
            }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        logger.debug(f"OpenRouter request payload: model={model}, reasoning={include_reasoning}")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield {"type": "error", "content": f"API Error: {error_text}"}
                    return

                async for line in response.content:
                    line = line.strip()
                    if not line or line == b'data: [DONE]':
                        continue

                    if line.startswith(b'data: '):
                        try:
                            json_str = line.decode('utf-8')[6:] # Remove 'data: '
                            chunk = json.loads(json_str)

                            # Handle Usage Chunk (usually the last one)
                            if "usage" in chunk and chunk["usage"]:
                                usage = chunk["usage"]
                                # Extract reasoning tokens if available
                                reasoning_tokens = 0
                                if "completion_tokens_details" in usage:
                                     reasoning_tokens = usage["completion_tokens_details"].get("reasoning_tokens", 0)

                                yield {
                                    "type": "usage",
                                    "prompt_tokens": usage.get("prompt_tokens", 0),
                                    "completion_tokens": usage.get("completion_tokens", 0),
                                    "total_tokens": usage.get("total_tokens", 0),
                                    "reasoning_tokens": reasoning_tokens
                                }
                                continue # Usage chunk might not have choices

                            if not chunk.get("choices"):
                                continue

                            delta = chunk["choices"][0].get("delta", {})

                            # Debug: Log all delta keys to see what's being returned
                            if delta:
                                delta_keys = list(delta.keys())
                                if delta_keys and delta_keys != ['content']:  # Only log non-content-only deltas
                                    logger.debug(f"Delta keys: {delta_keys}, values preview: {str(delta)[:200]}")

                            # 1. Handle Reasoning (from reasoning_details array)
                            # OpenRouter returns reasoning in `reasoning_details` array with objects:
                            # - type: "reasoning.text" with "text" field
                            # - type: "reasoning.summary" with "summary" field
                            # - type: "reasoning.encrypted" with "data" field (encrypted/redacted)
                            if "reasoning_details" in delta and delta["reasoning_details"]:
                                details = delta["reasoning_details"]
                                if isinstance(details, list):
                                    for detail in details:
                                        detail_type = detail.get("type", "")
                                        if detail_type == "reasoning.text":
                                            text = detail.get("text", "")
                                            if text:
                                                yield {"type": "thinking", "content": text}
                                        elif detail_type == "reasoning.summary":
                                            summary = detail.get("summary", "")
                                            if summary:
                                                yield {"type": "thinking", "content": f"[Summary] {summary}"}
                                        elif detail_type == "reasoning.encrypted":
                                            # Encrypted reasoning - log but don't display raw
                                            logger.debug(f"Received encrypted reasoning: {detail.get('format', 'unknown')}")
                                            # Optionally yield a placeholder
                                            yield {"type": "thinking", "content": "[Thinking...]"}
                                        else:
                                            # Unknown type, log and pass through
                                            logger.warning(f"Unknown reasoning_details type: {detail_type}")
                                else:
                                    # Handle non-array reasoning_details (fallback)
                                    if isinstance(details, str):
                                        yield {"type": "thinking", "content": details}
                                    else:
                                        yield {"type": "thinking", "content": str(details)}

                            # Legacy: Check for 'reasoning' field (some providers use this)
                            elif "reasoning" in delta and delta["reasoning"]:
                                reasoning_content = delta["reasoning"]
                                if not isinstance(reasoning_content, str):
                                    reasoning_content = str(reasoning_content)
                                yield {"type": "thinking", "content": reasoning_content}

                            # 2. Handle Tool Calls
                            if "tool_calls" in delta:
                                yield {"type": "tool_call_chunk", "tool_calls": delta["tool_calls"]}

                            # 3. Handle Standard Content (Token)
                            elif "content" in delta and delta["content"]:
                                yield {"type": "token", "content": delta["content"]}

                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode chunk: {line.decode('utf-8', errors='replace')}")
                        except Exception as e:
                            logger.error(f"Stream error: {e}")
