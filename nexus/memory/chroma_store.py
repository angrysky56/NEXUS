import datetime
import logging
from typing import Any

import chromadb

logger = logging.getLogger(__name__)

class ChromaMemory:
    """
    Persistent memory using ChromaDB.
    Stores chat history with metadata (Emotions, ID, etc).
    """

    def __init__(self, persist_dir: str = "./nexus_data"):
        self.client = chromadb.PersistentClient(path=persist_dir)

    def get_session_collection(self, session_id: str) -> Any:
        return self.client.get_or_create_collection(
            name=f"session_{session_id}",
            metadata={"created_at": str(datetime.datetime.now())}
        )

    def add_interaction(
        self,
        session_id: str,
        role: str,
        content: str,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] = {}
    ) -> None:
        collection = self.get_session_collection(session_id)

        # ID is timestamp + role
        msg_id = f"{datetime.datetime.now().isoformat()}_{role}"

        collection.add(
            documents=[content],
            embeddings=[embedding] if embedding else None,
            metadatas=[{**metadata, "role": role, "timestamp": str(datetime.datetime.now())}],
            ids=[msg_id]
        )

    def get_history(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        try:
            collection = self.get_session_collection(session_id)
            # Chroma doesn't have a simple "get last N" without querying
            # We fetch all (assuming reasonable context) or query.
            # For now, just getting the raw data.
            results = collection.get()

            # Sort by ID (timestamp)

            docs = results['documents']
            metas = results['metadatas']

            history = []
            for i, doc in enumerate(docs):
                history.append({
                    "role": metas[i]["role"],
                    "content": doc,
                    "metadata": metas[i]
                })

            history.sort(key=lambda x: x["metadata"]["timestamp"])
            return history[-limit:]
        except Exception as e:
            logger.error(f"Error reading history: {e}")
            return []

    def list_sessions(self) -> list[str]:
        # Chroma API change: list_collections returns objects
        colls = self.client.list_collections()
        return [c.name.replace("session_", "") for c in colls if c.name.startswith("session_")]
