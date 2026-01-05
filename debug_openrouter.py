
import aiohttp
import asyncio
import json

async def check():
    url = "https://openrouter.ai/api/v1/models"
    print(f"Fetching {url}...")

    # Try 1: No Auth
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            print(f"Status (No Auth): {response.status}")
            if response.status == 200:
                data = await response.json()
                models = data.get('data', [])
                print(f"Found {len(models)} models.")

                # Check for 'tools' support in a known model
                sonnet = next((m for m in models if 'claude-3.5-sonnet' in m['id']), None)
                if sonnet:
                    print(f"Sonnet ID: {sonnet['id']}")
                    print(f"Sonnet Supported Params: {sonnet.get('supported_parameters', [])}")
                    print(f"Tools supported? {'tools' in sonnet.get('supported_parameters', [])}")
                else:
                    print("Sonnet not found in list!")
            else:
                print(f"Error: {await response.text()}")

    # Try 2: With Bad Auth (Simulating "Bearer None" or bad key)
    headers = {"Authorization": "Bearer None"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            print(f"Status (Bad Auth): {response.status}")
            # If 401, that explains why the backend fails if the key is not set or set wrong

if __name__ == "__main__":
    asyncio.run(check())
