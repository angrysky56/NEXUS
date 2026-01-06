from typing import Any

from fastapi import APIRouter, Header
from pydantic import BaseModel

from ...integrations.openrouter import OpenRouterClient

router = APIRouter()
client = OpenRouterClient()

# trunk-ignore(ruff/E402)
from ...core.config import global_config


class ConfigUpdate(BaseModel):
    allowed_paths: list[str] | None = None
    workspace_dir: str | None = None
    max_tool_iterations: int | None = None
    system_prompt: str | None = None


@router.get("/config")
async def get_config() -> dict[str, Any]:
    return global_config.to_dict()


@router.post("/config")
async def update_config(cfg: ConfigUpdate) -> dict[str, Any]:
    update_data = {}
    if cfg.allowed_paths is not None:
        update_data["allowed_paths"] = cfg.allowed_paths
    if cfg.workspace_dir is not None:
        update_data["workspace_dir"] = cfg.workspace_dir
    if cfg.max_tool_iterations is not None:
        update_data["max_tool_iterations"] = cfg.max_tool_iterations
    if cfg.system_prompt is not None:
        update_data["system_prompt"] = cfg.system_prompt

    global_config.update(update_data)
    return global_config.to_dict()


@router.get("/models")
async def get_models(
    x_openrouter_key: str | None = Header(None, alias="X-OpenRouter-Key")
) -> list[dict[str, Any]]:
    """Proxy OpenRouter models, filtered for tools."""
    return await client.fetch_models(api_key=x_openrouter_key)
