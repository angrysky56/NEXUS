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
        self.models_metadata: dict[str, dict[str, Any]] = {}  # Cache model metadata
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
                    # Cache the full metadata for lookup during chat
                    # Get provider specific limits if available
                    top_provider = model.get("top_provider", {})
                    self.models_metadata[model["id"]] = {
                        "context_length": model.get("context_length", 0),
                        "max_completion_tokens": top_provider.get("max_completion_tokens"),
                        "supported_parameters": model.get("supported_parameters", []),
                        "default_parameters": model.get("default_parameters", {})
                    }

                    # Strict check based on OpenRouter docs:
                    # check if 'tools' is in the 'supported_parameters' list
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

    async def ensure_metadata(self, api_key: str | None = None) -> None:
        """
        Ensure model metadata is fetched.
        """
        if not self.models_metadata:
            await self.fetch_models(api_key)

    async def get_model_metadata(self, model_id: str, api_key: str | None = None) -> dict[str, Any]:
        """
        Retrieves cached metadata for a model.
        """
        await self.ensure_metadata(api_key)
        return self.models_metadata.get(model_id, {})

    async def supports_parameter(self, model_id: str, parameter: str, api_key: str | None = None) -> bool:
        """
        Checks if a model supports a specific parameter (e.g., 'temperature').
        """
        metadata = await self.get_model_metadata(model_id, api_key)
        supported = metadata.get("supported_parameters") or []
        return parameter in supported

    async def get_model_default_temp(self, model_id: str) -> float | None:
        """
        Retrieves the default temperature for a model from metadata if NEXUS doesn't override it.
        """
        metadata = await self.get_model_metadata(model_id)
        default_params = metadata.get("default_parameters") or {}
        temp = default_params.get("temperature")
        return float(temp) if temp is not None else None

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

        # Use dynamic max_tokens from metadata if not explicitly provided or default
        await self.ensure_metadata(request_key)
        metadata = self.models_metadata.get(model, {})
        model_max = metadata.get("max_completion_tokens") or 16384 # Modern default
        context_limit = metadata.get("context_length") or 128000 # Default if unknown

        # 0. Truncate history if it exceeds context volume
        # Simple heuristic: 1 token ~= 4 characters
        token_limit = int(context_limit * 0.8) # Conservative buffer
        trimmed_messages = self._truncate_messages(messages, token_limit)

        # If user provided 1000 (default) and model supports more, use model's max
        actual_max_tokens = max_tokens
        if max_tokens == 1000 and model_max > 1000:
             actual_max_tokens = model_max

        payload = {
            "model": model,
            "messages": trimmed_messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": actual_max_tokens,
            "stream_options": {"include_usage": True}
        }

        # Use the official `reasoning` object configuration
        if include_reasoning:
            # Using effort: high is the standardized way for OpenRouter thinking models
            payload["reasoning"] = {
                "effort": "high",
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

                            # 1. Handle Reasoning (from reasoning_details array or direct field)
                            # We use a flag to avoid yielding duplicates if multiple fields are present
                            reasoning_yielded = False

                            # Prioritize reasoning_details (OpenRouter standard)
                            if "reasoning_details" in delta and delta["reasoning_details"]:
                                details = delta["reasoning_details"]
                                if isinstance(details, list):
                                    for detail in details:
                                        detail_type = detail.get("type", "")
                                        if detail_type == "reasoning.text":
                                            text = detail.get("text", "")
                                            if text:
                                                yield {"type": "thinking", "content": text}
                                                reasoning_yielded = True
                                        elif detail_type == "reasoning.summary":
                                            summary = detail.get("summary", "")
                                            if summary:
                                                yield {"type": "thinking", "content": f"[Summary] {summary}"}
                                                reasoning_yielded = True
                                        elif detail_type == "reasoning.encrypted":
                                            # Redacted signatures - skip yielding placeholders to avoid UI clutter
                                            logger.debug(f"Skipping encrypted reasoning signature: {detail.get('format', 'unknown')}")

                            # Fallback 1: Direct reasoning field (common in some providers)
                            if not reasoning_yielded and "reasoning" in delta and delta["reasoning"]:
                                reasoning_content = delta["reasoning"]
                                if reasoning_content and isinstance(reasoning_content, str):
                                    yield {"type": "thinking", "content": reasoning_content}
                                    reasoning_yielded = True

                            # Fallback 2: Check for interleaved reasoning in content (rare on OpenRouter but possible)
                            # We don't do this here to avoid mixing thoughts with final output unless strictly necessary.

                            # 2. Handle Tool Calls
                            # 2. Handle Tool Calls
                            if "tool_calls" in delta and delta["tool_calls"]:
                                yield {"type": "tool_call_chunk", "tool_calls": delta["tool_calls"]}

                            # 3. Handle Standard Content (Token)
                            elif "content" in delta and delta["content"]:
                                # Only yield as token if not already yielded as reasoning in this chunk
                                # and doesn't look like reasoning that bled into content
                                yield {"type": "token", "content": delta["content"]}

                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode chunk: {line.decode('utf-8', errors='replace')}")
                        except Exception as e:
                            logger.error(f"Stream error: {e}")

    def _truncate_messages(self, messages: list[dict[str, Any]], token_limit: int) -> list[dict[str, Any]]:
        """
        Rough truncation of messages to fit within token limit.
        Heuristic: 3.5 chars per token (conservative).
        Always keeps the system message.
        """
        if not messages:
            return []

        system_msg = None
        other_msgs = []

        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg
            else:
                other_msgs.append(msg)

        # Estimate tokens
        def est_tokens(m_list: list[dict[str, Any]]) -> int:
            total_chars = 0
            for m in m_list:
                if not m:
                    continue
                content = m.get("content") or ""
                if isinstance(content, str):
                    total_chars += len(content)
                elif isinstance(content, list): # handle multi-modal/tool result types
                    total_chars += len(str(content))
                # Add overhead for role/name
                total_chars += 20
            return int(total_chars // 3.5)

        if est_tokens(messages) <= token_limit:
            return messages

        # Truncate from the oldest (start of other_msgs)
        while other_msgs and est_tokens([system_msg] + other_msgs if system_msg else other_msgs) > token_limit:
            if len(other_msgs) > 1:
                other_msgs.pop(0)
            else:
                # If only one message left and still too big, we might need to truncate its content
                msg = other_msgs[0]
                content = msg.get("content") or ""
                if isinstance(content, str) and len(content) > token_limit * 3:
                    msg["content"] = content[-(token_limit * 3):]
                break

        return [system_msg] + other_msgs if system_msg else other_msgs

