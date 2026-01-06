import asyncio
import json
import os

from dotenv import load_dotenv

from nexus.integrations.openrouter import OpenRouterClient

load_dotenv()

async def main():
    print("Starting debug stream...")
    client = OpenRouterClient()

    # Simulate the request that failed
    messages = [
        {"role": "user", "content": "Write me a logic of poems and a poem of logic."}
    ]

    try:
        async for chunk in client.stream_chat(
            messages=messages,
            model="prime-intellect/intellect-3",
            include_reasoning=True,
            api_key=os.getenv("OPENROUTER_API_KEY") # Ensure this is picked up
        ):
            print(f"Chunk: {chunk}")
            if chunk.get("type") == "error":
                print(f"ERROR PACKET: {chunk['content']}")

    except Exception as e:
        print(f"EXCEPTION: {e}")

if __name__ == "__main__":
    asyncio.run(main())
