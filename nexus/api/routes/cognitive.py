import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter()

# Global state store (simple in-memory for MVP, could use Redis/SharedMemory)
# This will be updated by the Engine during processing
COGNITIVE_STATE = {
    "valence": 0.0,
    "arousal": 0.0,
    "intrinsic_dimension": 0.0,
    "gate_value": 0.5,
    "primary_manifold": "neutral"
}

@router.get("/status")
async def stream_status() -> StreamingResponse:
    """Server-Sent Events for real-time cognitive dashboard."""
    async def event_generator() -> AsyncGenerator[str, None]:
        while True:
            # yield current state every 100ms
            yield f"data: {json.dumps(COGNITIVE_STATE)}\n\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
