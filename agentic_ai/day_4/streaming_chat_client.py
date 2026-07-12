
# A terminal chat client where:

# Responses stream token by token (not dump all at once)
# Full conversation history maintained across turns
# Token count shown after every reply
# /clear command resets history
# /quit exits
# Rate limit backoff built in
# System prompt makes it a Stack Overflow expert

import os
import time
from dotenv import load_dotenv
import anthropic

load_dotenv(dotenv_path='../../.env')

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM = """You are a Stack Overflow expert. Rules:
- Answer only programming questions
- Always include a short code example
- If you don't know, say so — never guess
- Be concise, max 5 sentences + code"""


def getResponse(history, max_retries = 4) :
    for attempt in range(max_retries):
        try:
            # history.append({"role": "user", "content": message})

            print("Response: ", end="", flush=True)

            full_reply = ""

            with client.messages.stream(
                model="claude-haiku-4-5",
                max_tokens=500,
                system=SYSTEM,
                messages=history
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    full_reply += text

                final = stream.get_final_message()
            
            print() # new line 
            return full_reply, final.usage
        
        except anthropic.RateLimitError:
            wait = 2 ** attempt
            print(f"\n[Rate Limited. Waiting {wait}s...]")
            time.sleep(wait) 

        except anthropic.APIStatusError as e:
            print(f"\n[API error {e.status_code}: {e.message}]")
            raise

    raise RuntimeError("Max Retries Exceeded")

def chat():
    history = []
    total_input = 0
    total_output = 0

    print("=== Stack Overflow Chat (streaming) ===")
    print("Commands: /clear — reset history | /quit — exit")
    print("Model: claude-haiku-4-5\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print(f"\nSession total — Input: {total_input} tokens | Output: {total_output} tokens")
            break

        if user_input == "/clear":
            history = []
            print("[History cleared]\n")
            continue

        history.append({"role": "user", "content": user_input})

        reply, usage = getResponse(history)

        history.append({"role": "assistant", "content": reply})

        total_input  += usage.input_tokens
        total_output += usage.output_tokens

        print(f"[Tokens — in: {usage.input_tokens} | out: {usage.output_tokens} | history: {len(history)} msgs]\n")

        # Trim if history gets long (keep last 10 messages)
        if len(history) > 10:
            history = history[-10:]
            print("[History trimmed to last 10 messages]\n")


if __name__ == "__main__":
    chat()
