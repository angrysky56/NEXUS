import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any, TypedDict

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ...engine.hybrid_engine import HybridEngine
from ...integrations.openrouter import OpenRouterClient
from ...memory.chroma_store import ChromaMemory
from ...tools.manager import ToolManager
from .cognitive import COGNITIVE_STATE

logger = logging.getLogger(__name__)


class MessageDict(TypedDict):
    role: str
    content: str | None
    tool_calls: list[dict[str, Any]] | None
    tool_call_id: str | None


router: APIRouter = APIRouter()

# Init Core Components
client: OpenRouterClient = OpenRouterClient()
memory: ChromaMemory = ChromaMemory()
tool_manager: ToolManager = ToolManager()


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    model: str | None = None  # Primary model or fallback
    logic_model: str | None = None
    creative_model: str | None = None
    include_reasoning: bool = False
    max_tool_iterations: int | None = None


@router.get("/sessions")
async def list_sessions() -> list[str]:
    return memory.list_sessions()


@router.get("/history/{session_id}")
async def get_history(session_id: str) -> list[dict[str, Any]]:
    history = memory.get_history(session_id)
    # Filter to match the Message interface expected by frontend if needed
    # but the frontend will parse the raw documents.
    return history


@router.post("/completions")
async def chat_completions(
    req: ChatRequest,
    x_openrouter_key: str | None = Header(None, alias="X-OpenRouter-Key"),
) -> StreamingResponse:
    logger.info(
        f"[CHAT] New request: session={req.session_id}, model={req.model}, "
        f"include_reasoning={req.include_reasoning}, iterations={req.max_tool_iterations}"
    )
    logger.debug(
        f"[CHAT] User message: {req.message[:100]}..."
        if len(req.message) > 100
        else f"[CHAT] User message: {req.message}"
    )

    # Retrieve History
    history = memory.get_history(req.session_id)
    history_dicts: list[MessageDict] = [
        {
            "role": h["role"],
            "content": h["content"],
            "tool_calls": None,
            "tool_call_id": None,
        }
        for h in history
    ]
    logger.info(f"[CHAT] Retrieved {len(history_dicts)} history messages")

    # Save User Input
    memory.add_interaction(req.session_id, "user", req.message)

    effective_key = x_openrouter_key or os.getenv("OPENROUTER_API_KEY")
    if not effective_key:
        logger.error("[CHAT] No API key provided")
        raise HTTPException(
            status_code=401,
            detail="API Key required. Set in Settings or OPENROUTER_API_KEY env var.",
        )

    request_client = OpenRouterClient(api_key=effective_key)

    # Use models from request. Default to the same model for both manifolds
    # unless logic_model or creative_model are explicitly provided.
    logic_model = req.logic_model or req.model
    creative_model = req.creative_model or req.model

    if not logic_model:
        raise HTTPException(
            status_code=400, detail="Model is required ('model' or 'logic_model')."
        )

    request_engine = HybridEngine(
        client=request_client,
        logic_model=logic_model,
        creative_model=creative_model or logic_model,
    )
    logger.info(
        f"[CHAT] Engine initialized: logic={logic_model}, creative={creative_model or logic_model}"
    )

    return StreamingResponse(
        event_generator(
            req=req,
            history_dicts=history_dicts,
            request_engine=request_engine,
            request_client=request_client,
            x_openrouter_key=x_openrouter_key,
            logic_model=logic_model,
        ),
        media_type="text/event-stream",
    )


async def event_generator(
    req: ChatRequest,
    history_dicts: list[MessageDict],
    request_engine: HybridEngine,
    request_client: OpenRouterClient,
    x_openrouter_key: str | None,
    logic_model: str,
) -> AsyncGenerator[str, None]:
    """Generates a stream of events for the chat completion."""
    full_response = ""
    full_thinking = ""
    all_executed_tool_calls: list[dict[str, Any]] = []
    packet_counts: dict[str, int] = {}
    current_history = history_dicts.copy()
    from ...core.config import global_config

    max_tool_iterations = req.max_tool_iterations or global_config.max_tool_iterations
    iteration = 0

    system_prompt_for_continuation = ""
    selected_model = logic_model

    logger.info("[STREAM] Starting stream processing")

    while iteration < max_tool_iterations:
        iteration += 1
        tool_calls_buffer: dict[int, dict[str, Any]] = {}
        is_first_iteration = iteration == 1

        if is_first_iteration:
            # First iteration: Use full HybridEngine cognitive pipeline
            logger.info(
                f"[STREAM] Iteration {iteration}: Using HybridEngine cognitive pipeline"
            )
            async for packet in request_engine.process_stream(
                user_input=req.message,
                history=current_history,  # type: ignore[arg-type]
                tools=tool_manager.tools_schema,
                api_key=x_openrouter_key,
                include_reasoning=req.include_reasoning,
            ):
                packet_type = packet.get("type", "unknown")
                packet_counts[packet_type] = packet_counts.get(packet_type, 0) + 1

                if packet_type == "cognitive_update":
                    logger.info(
                        f"[STREAM] Cognitive update: manifold="
                        f"{packet['data'].get('primary_manifold')}, "
                        f"ID={packet['data'].get('intrinsic_dimension', 'N/A'):.2f}"
                    )
                    COGNITIVE_STATE.update(packet["data"])
                    tool_manager.update_cognitive_state(packet["data"])
                    system_prompt_for_continuation = (
                        f"You are NEXUS, a cognitive AI assistant. "
                        f"Current state: manifold={packet['data'].get('primary_manifold')}"
                    )
                    selected_model = packet["data"].get("model", selected_model)
                    yield f"event: cognitive\ndata: {json.dumps(packet['data'])}\n\n"

                elif packet_type == "thinking":
                    content = packet.get("content", "")
                    full_thinking += content
                    yield f"data: {json.dumps({'type': 'thinking', 'content': content})}\n\n"

                elif packet_type == "token":
                    full_response += packet["content"]
                    yield f"data: {json.dumps({'content': packet['content']})}\n\n"

                elif packet_type == "tool_call_chunk":
                    tool_calls = packet.get("tool_calls", [])
                    for tc in tool_calls:
                        idx = tc.get("index", 0)
                        tc_id = tc.get("id", "")
                        func = tc.get("function", {})
                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {
                                "id": tc_id,
                                "name": func.get("name", ""),
                                "arguments": func.get("arguments", ""),
                            }
                        else:
                            if func.get("arguments"):
                                tool_calls_buffer[idx]["arguments"] += func["arguments"]
                            if tc_id and not tool_calls_buffer[idx]["id"]:
                                tool_calls_buffer[idx]["id"] = tc_id
                            if func.get("name") and not tool_calls_buffer[idx]["name"]:
                                tool_calls_buffer[idx]["name"] = func["name"]
                    yield (
                        f"data: {json.dumps({'type': 'tool_call_chunk', 'tool_calls': tool_calls})}"
                        f"\n\n"
                    )

                elif packet_type == "usage":
                    yield f"data: {json.dumps({'type': 'usage', **packet})}\n\n"

                elif packet_type == "error":
                    logger.error(f"[STREAM] Received error packet: {packet['content']}")
                    yield f"data: {json.dumps({'type': 'error', 'content': packet['content']})}\n\n"

                else:
                    debug_info = {
                        "type": "debug",
                        "packet_type": packet_type,
                        "content": str(packet)[:500],
                    }
                    yield f"data: {json.dumps(debug_info)}\n\n"
        else:
            # Subsequent iterations: Use direct LLM call with tool results
            logger.info(
                f"[STREAM] Iteration {iteration}: Direct LLM call for tool result continuation"
            )

            messages = [
                {"role": "system", "content": system_prompt_for_continuation}
            ] + current_history

            async for packet in request_client.stream_chat(
                messages=messages,  # type: ignore[arg-type]
                model=selected_model,
                tools=tool_manager.tools_schema,
                temperature=0.5,
                api_key=x_openrouter_key,
                include_reasoning=req.include_reasoning,
            ):
                packet_type = packet.get("type", "unknown")
                packet_counts[packet_type] = packet_counts.get(packet_type, 0) + 1

                if packet_type == "thinking":
                    content = packet.get("content", "")
                    full_thinking += content
                    yield f"data: {json.dumps({'type': 'thinking', 'content': content})}\n\n"

                elif packet_type == "token":
                    full_response += packet["content"]
                    yield f"data: {json.dumps({'content': packet['content']})}\n\n"

                elif packet_type == "tool_call_chunk":
                    tool_calls = packet.get("tool_calls", [])
                    for tc in tool_calls:
                        idx = tc.get("index", 0)
                        tc_id = tc.get("id", "")
                        func = tc.get("function", {})
                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {
                                "id": tc_id,
                                "name": func.get("name", ""),
                                "arguments": func.get("arguments", ""),
                            }
                        else:
                            if func.get("arguments"):
                                tool_calls_buffer[idx]["arguments"] += func["arguments"]
                            if tc_id and not tool_calls_buffer[idx]["id"]:
                                tool_calls_buffer[idx]["id"] = tc_id
                            if func.get("name") and not tool_calls_buffer[idx]["name"]:
                                tool_calls_buffer[idx]["name"] = func["name"]
                    yield (
                        f"data: {json.dumps({'type': 'tool_call_chunk', 'tool_calls': tool_calls})}"
                        f"\n\n"
                    )

                elif packet_type == "usage":
                    yield f"data: {json.dumps({'type': 'usage', **packet})}\n\n"

                elif packet_type == "error":
                    logger.error(
                        f"[STREAM] Received error packet in direct call: {packet['content']}"
                    )
                    yield f"data: {json.dumps({'type': 'error', 'content': packet['content']})}\n\n"

        # After stream completes, check if we have tool calls to execute
        if not tool_calls_buffer:
            logger.info(f"[STREAM] No tool calls, ending loop at iteration {iteration}")
            break

        # Execute tool calls
        logger.info(
            f"[STREAM] Executing {len(tool_calls_buffer)} tool calls (iteration {iteration})"
        )

        assistant_tool_calls = []
        for idx in sorted(tool_calls_buffer.keys()):
            tc = tool_calls_buffer[idx]
            assistant_tool_calls.append(
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
            )

        # Add assistant message with tool_calls to history
        current_history.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": assistant_tool_calls,
                "tool_call_id": None,
            }
        )

        # Execute each tool and collect results
        for idx in sorted(tool_calls_buffer.keys()):
            tc = tool_calls_buffer[idx]
            tool_name = tc["name"]
            try:
                tool_args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError as e:
                logger.error(f"[STREAM] Failed to parse tool arguments: {e}")
                tool_args = {}

            logger.info(f"[STREAM] Executing tool: {tool_name}({tool_args})")
            yield (
                f"data: {json.dumps({'type': 'tool_executing', 'name': tool_name, 'id': tc['id']})}"
                f"\n\n"
            )

            try:
                result = await tool_manager.execute(tool_name, tool_args)
            except Exception as e:
                result = f"Error executing tool: {str(e)}"
                logger.error(f"[STREAM] Tool execution error: {e}")

            tool_res = {
                "type": "tool_result",
                "name": tool_name,
                "id": tc["id"],
                "result": result,
            }
            yield f"data: {json.dumps(tool_res)}\n\n"

            # Add tool result to history
            current_history.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                    "tool_calls": None,
                }
            )

            # Also collect for metadata
            all_executed_tool_calls.append(
                {
                    "id": tc["id"],
                    "name": tool_name,
                    "arguments": tc["arguments"],
                    "result": result,
                }
            )

        logger.info(
            f"[STREAM] Tool execution complete, continuing to iteration {iteration + 1}"
        )

    if iteration >= max_tool_iterations:
        logger.warning(f"[STREAM] Reached max tool iterations ({max_tool_iterations})")

    logger.info(
        f"[STREAM] Complete. Packet counts: {packet_counts}, "
        f"response_len={len(full_response)}, iterations={iteration}"
    )

    if full_response or full_thinking or all_executed_tool_calls:
        memory.add_interaction(
            req.session_id,
            "assistant",
            full_response,
            metadata={
                "thinking": full_thinking,
                "tool_calls": all_executed_tool_calls,
                "cognitive_state": COGNITIVE_STATE.copy(),
            },
        )

    yield "data: [DONE]\n\n"
