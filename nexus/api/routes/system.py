from typing import Any

from fastapi import APIRouter, Header
from pydantic import BaseModel

from ...integrations.openrouter import OpenRouterClient

router = APIRouter()
client = OpenRouterClient()

# Simple In-Memory Config Store for MVP
CONFIG_STORE = {
    "allowed_paths": ["./"],
    "max_tool_iterations": 20,
    "system_prompt": "You are NEXUS, a cognitive architecture with Bicameral bicameralism."
}

class ConfigUpdate(BaseModel):
    allowed_paths: list[str] | None = None
    max_tool_iterations: int | None = None
    system_prompt: str | None = None

@router.get("/config")
async def get_config() -> dict[str, Any]:
    return CONFIG_STORE

@router.post("/config")
async def update_config(cfg: ConfigUpdate) -> dict[str, Any]:
    if cfg.allowed_paths is not None:
        CONFIG_STORE["allowed_paths"] = cfg.allowed_paths
    if cfg.max_tool_iterations is not None:
        CONFIG_STORE["max_tool_iterations"] = cfg.max_tool_iterations
    if cfg.system_prompt is not None:
        CONFIG_STORE["system_prompt"] = cfg.system_prompt
    return CONFIG_STORE

@router.get("/models")
async def get_models(x_openrouter_key: str | None = Header(None, alias="X-OpenRouter-Key")) -> list[dict[str, Any]]:
    """Proxy OpenRouter models, filtered for tools."""
    return await client.fetch_models(api_key=x_openrouter_key)
