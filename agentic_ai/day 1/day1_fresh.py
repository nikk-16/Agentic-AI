import os
import json
import httpx
import asyncio
import datetime
from dotenv import load_dotenv

# load_dotenv(dotenv_path="../.env")

# API_KEY = os.getenv("OPENWEATHER_API_KEY", "no-key-yet")
# print(f"Key loaded: {API_KEY[:6]}...")


# # Simulating a raw API response (always comes as a string)
# raw_response = '{"model": "gpt-4", "usage": {"prompt_tokens": 50, "completion_tokens": 120}, "choices": [{"message": {"content": "Hello!"}}]}'

# #1 Parse 
# data = json.loads(raw_response)

# #2 Access nested values
# print(data["choices"][0]["message"]["content"])

# #3 add a field and convert back to string
# data["total_tokens"] = data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
# print(json.dumps(data, indent = 5))



# async def fetch_price(coin: str) -> dict:
#     async with httpx.AsyncClient() as client:
#         response = await client.get(
#             f"https://api.coinbase.com/v2/prices/{coin}-USD/spot"
#         )
#         data = response.json()
#         price = data["data"]["amount"]
#         print(f"{coin}: ${price}")
#         return {"coin": coin, "price": price}

# async def main():
#     # Run all 3 fetches concurrently
#     results = await asyncio.gather(
#         fetch_price("BTC"),
#         fetch_price("ETH"),
#         fetch_price("SOL"),
#         return_exceptions=True   # ← failed calls return the Exception object instead of crashing
#     )
#     for result in results:
#         if isinstance(result, Exception):
#             print(f"Failed: {result}")
#         else:
#             print(result)

# asyncio.run(main())

# ========================================================================================================

# Async programming 

print(f"Select the currency you want to get info about: BTC/ETH/SOL")
coin = input("Currency: ").upper()

async def fetch_price(coin: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.coinbase.com/v2/prices/{coin}-USD/spot"
        )
        data = response.json()
        price = data["data"]["amount"]
        print(f"{coin}: ${price}")
        return {"coin": coin, "price": price}

async def main():
    results = await asyncio.gather(
        fetch_price(coin),
        return_exceptions=True   # ← failed calls return the Exception object instead of crashing
    )
    for result in results:
        if isinstance(result, Exception):
            print(f"Failed: {result}")
        else:
            print(result)
    
    output = {"timestamp": datetime.datetime.now().isoformat(), "results": results}
    with open("prices.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved to prices.json")


asyncio.run(main())