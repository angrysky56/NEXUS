import logging
import os
import sys

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import chat, cognitive, system

# Configure logging for NEXUS modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
# Set NEXUS modules to INFO level
logging.getLogger("nexus").setLevel(logging.INFO)
# Set openrouter to DEBUG to see raw API responses
logging.getLogger("nexus.integrations.openrouter").setLevel(logging.DEBUG)
# Reduce noise from other libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

load_dotenv()
logger.info("NEXUS Cognitive Engine starting...")

app = FastAPI(title="NEXUS Cognitive Engine", version="0.2.0")

# Configure CORS for React UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(cognitive.router, prefix="/api/v1/cognitive", tags=["cognitive"])
app.include_router(system.router, prefix="/api/v1/system", tags=["system"])

@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "operational", "system": "NEXUS Hybrid Engine"}
