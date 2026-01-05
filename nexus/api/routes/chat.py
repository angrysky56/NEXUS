import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ...engine.hybrid_engine import HybridEngine
from ...integrations.openrouter import OpenRouterClient
from ...memory.chroma_store import ChromaMemory
from ...tools.manager import ToolManager
from .cognitive import COGNITIVE_STATE

logger = logging.getLogger(__name__)

router = APIRouter()

# Init Core Components
# In a real app, use verify_api_key dependency and get models from Config
client = OpenRouterClient()
memory = ChromaMemory()
tool_manager = ToolManager()

# Default Engine
# We'll default to decent models. Frontend can override via config in future.
engine = HybridEngine(
    client=client,
    logic_model="anthropic/claude-3.5-sonnet", # High capability
    creative_model="google/gemini-flash-1.5" # High speed/context
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    model: str | None = None
    include_reasoning: bool = False

@router.post("/completions")
async def chat_completions(req: ChatRequest, x_openrouter_key: str | None = Header(None, alias="X-OpenRouter-Key")) -> StreamingResponse:
    logger.info(f"[CHAT] New request: session={req.session_id}, model={req.model}, include_reasoning={req.include_reasoning}")
    logger.debug(f"[CHAT] User message: {req.message[:100]}..." if len(req.message) > 100 else f"[CHAT] User message: {req.message}")

    # Retrieve History
    history = memory.get_history(req.session_id)
    history_dicts = [{"role": h['role'], "content": h['content']} for h in history]
    logger.info(f"[CHAT] Retrieved {len(history_dicts)} history messages")

    # Save User Input
    memory.add_interaction(req.session_id, "user", req.message)

    # Use model from request - frontend is responsible for providing a valid model
    if not req.model:
        raise HTTPException(status_code=400, detail="Model is required")
    selected_model = req.model

    # Create client with API key - read from header or environment at runtime
    import os
    effective_key = x_openrouter_key or os.getenv("OPENROUTER_API_KEY")
    if not effective_key:
        logger.error("[CHAT] No API key provided")
        raise HTTPException(status_code=401, detail="API Key required. Set in Settings or OPENROUTER_API_KEY env var.")

    request_client = OpenRouterClient(api_key=effective_key)

    # Create engine with selected model for both manifolds
    # In a full implementation, you might have separate logic/creative model selection
    request_engine = HybridEngine(
        client=request_client,
        logic_model=selected_model,
        creative_model=selected_model
    )
    logger.info(f"[CHAT] Engine initialized with model: {selected_model}")

    async def event_generator() -> AsyncGenerator[str, None]:
        full_response = ""
        packet_counts: dict[str, int] = {}
        current_history = history_dicts.copy()
        max_tool_iterations = 5
        iteration = 0
        system_prompt_for_continuation = ""  # Captured from first iteration's cognitive context

        logger.info(f"[STREAM] Starting stream processing")

        while iteration < max_tool_iterations:
            iteration += 1
            tool_calls_buffer: dict[int, dict] = {}
            is_first_iteration = iteration == 1

            if is_first_iteration:
                # First iteration: Use full HybridEngine cognitive pipeline
                logger.info(f"[STREAM] Iteration {iteration}: Using HybridEngine cognitive pipeline")
                async for packet in request_engine.process_stream(
                    user_input=req.message,
                    history=current_history,
                    tools=tool_manager.tools_schema,
                    api_key=x_openrouter_key,
                    include_reasoning=req.include_reasoning
                ):
                    packet_type = packet.get("type", "unknown")
                    packet_counts[packet_type] = packet_counts.get(packet_type, 0) + 1

                    if packet_type == "cognitive_update":
                        logger.info(f"[STREAM] Cognitive update: manifold={packet['data'].get('primary_manifold')}, ID={packet['data'].get('intrinsic_dimension', 'N/A'):.2f}")
                        COGNITIVE_STATE.update(packet["data"])
                        tool_manager.update_cognitive_state(packet["data"])
                        # Capture a basic system prompt for continuation
                        system_prompt_for_continuation = f"You are NEXUS, a cognitive AI assistant. Current state: manifold={packet['data'].get('primary_manifold')}"
                        yield f"event: cognitive\ndata: {json.dumps(packet['data'])}\n\n"

                    elif packet_type == "thinking":
                        content = packet.get("content", "")
                        preview = content[:80] + "..." if len(content) > 80 else content
                        logger.debug(f"[STREAM] Thinking chunk: {preview}")
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
                                tool_calls_buffer[idx] = {"id": tc_id, "name": func.get("name", ""), "arguments": func.get("arguments", "")}
                                if tc_id:
                                    logger.info(f"[STREAM] Tool call started: id={tc_id}, name={func.get('name', 'unknown')}")
                            else:
                                if func.get("arguments"):
                                    tool_calls_buffer[idx]["arguments"] += func["arguments"]
                                if tc_id and not tool_calls_buffer[idx]["id"]:
                                    tool_calls_buffer[idx]["id"] = tc_id
                                if func.get("name") and not tool_calls_buffer[idx]["name"]:
                                    tool_calls_buffer[idx]["name"] = func["name"]
                        yield f"data: {json.dumps({'type': 'tool_call_chunk', 'tool_calls': tool_calls})}\n\n"

                    elif packet_type == "usage":
                        logger.info(f"[STREAM] Usage: prompt={packet.get('prompt_tokens', 0)}, completion={packet.get('completion_tokens', 0)}, reasoning={packet.get('reasoning_tokens', 0)}")
                        yield f"data: {json.dumps({'type': 'usage', **packet})}\n\n"

                    elif packet_type == "error":
                        logger.error(f"[STREAM] Error from engine: {packet.get('content', 'unknown error')}")
                        yield f"data: {json.dumps({'type': 'error', 'content': packet['content']})}\n\n"

                    else:
                        logger.warning(f"[STREAM] Unhandled packet type: {packet_type}, content: {str(packet)[:200]}")
                        yield f"data: {json.dumps({'type': 'debug', 'packet_type': packet_type, 'content': str(packet)[:500]})}\n\n"
            else:
                # Subsequent iterations: Use direct LLM call with tool results
                logger.info(f"[STREAM] Iteration {iteration}: Direct LLM call for tool result continuation")

                # Build messages for direct call
                messages = [{"role": "system", "content": system_prompt_for_continuation}] + current_history

                async for packet in request_client.stream_chat(
                    messages=messages,
                    model=selected_model,
                    tools=tool_manager.tools_schema,
                    temperature=0.5,  # Moderate temperature for follow-up
                    api_key=x_openrouter_key,
                    include_reasoning=req.include_reasoning
                ):
                    packet_type = packet.get("type", "unknown")
                    packet_counts[packet_type] = packet_counts.get(packet_type, 0) + 1

                    if packet_type == "thinking":
                        content = packet.get("content", "")
                        preview = content[:80] + "..." if len(content) > 80 else content
                        logger.debug(f"[STREAM] Thinking chunk: {preview}")
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
                                tool_calls_buffer[idx] = {"id": tc_id, "name": func.get("name", ""), "arguments": func.get("arguments", "")}
                                if tc_id:
                                    logger.info(f"[STREAM] Tool call started: id={tc_id}, name={func.get('name', 'unknown')}")
                            else:
                                if func.get("arguments"):
                                    tool_calls_buffer[idx]["arguments"] += func["arguments"]
                                if tc_id and not tool_calls_buffer[idx]["id"]:
                                    tool_calls_buffer[idx]["id"] = tc_id
                                if func.get("name") and not tool_calls_buffer[idx]["name"]:
                                    tool_calls_buffer[idx]["name"] = func["name"]
                        yield f"data: {json.dumps({'type': 'tool_call_chunk', 'tool_calls': tool_calls})}\n\n"

                    elif packet_type == "usage":
                        logger.info(f"[STREAM] Usage: prompt={packet.get('prompt_tokens', 0)}, completion={packet.get('completion_tokens', 0)}, reasoning={packet.get('reasoning_tokens', 0)}")
                        yield f"data: {json.dumps({'type': 'usage', **packet})}\n\n"

                    elif packet_type == "error":
                        logger.error(f"[STREAM] Error from engine: {packet.get('content', 'unknown error')}")
                        yield f"data: {json.dumps({'type': 'error', 'content': packet['content']})}\n\n"

                    else:
                        logger.warning(f"[STREAM] Unhandled packet type: {packet_type}, content: {str(packet)[:200]}")

            # After stream completes, check if we have tool calls to execute
            if not tool_calls_buffer:
                logger.info(f"[STREAM] No tool calls, ending loop at iteration {iteration}")
                break

            # Execute tool calls
            logger.info(f"[STREAM] Executing {len(tool_calls_buffer)} tool calls (iteration {iteration})")

            # Build assistant message with tool calls for history
            assistant_tool_calls = []
            for idx in sorted(tool_calls_buffer.keys()):
                tc = tool_calls_buffer[idx]
                logger.info(f"[STREAM] Complete tool call [{idx}]: id={tc['id']}, name={tc['name']}, args={tc['arguments']}")
                assistant_tool_calls.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]}
                })

            # Add assistant message with tool_calls to history
            current_history.append({
                "role": "assistant",
                "content": None,
                "tool_calls": assistant_tool_calls
            })

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
                yield f"data: {json.dumps({'type': 'tool_executing', 'name': tool_name, 'id': tc['id']})}\n\n"

                try:
                    result = await tool_manager.execute(tool_name, tool_args)
                    logger.info(f"[STREAM] Tool result for {tool_name}: {result[:100]}..." if len(result) > 100 else f"[STREAM] Tool result for {tool_name}: {result}")
                except Exception as e:
                    result = f"Error executing tool: {str(e)}"
                    logger.error(f"[STREAM] Tool execution error: {e}")

                yield f"data: {json.dumps({'type': 'tool_result', 'name': tool_name, 'id': tc['id'], 'result': result})}\n\n"

                # Add tool result to history
                current_history.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result
                })

            logger.info(f"[STREAM] Tool execution complete, continuing to iteration {iteration + 1}")

        if iteration >= max_tool_iterations:
            logger.warning(f"[STREAM] Reached max tool iterations ({max_tool_iterations})")

        logger.info(f"[STREAM] Complete. Packet counts: {packet_counts}, response_len={len(full_response)}, iterations={iteration}")

        if full_response:
            memory.add_interaction(req.session_id, "assistant", full_response)

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
