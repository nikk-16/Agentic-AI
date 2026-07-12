import os
import anthropic
from dotenv import load_dotenv, find_dotenv

load_dotenv(dotenv_path="../../.env")

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def basic_call():
    print("=== Part 1: Basic API Call ===")

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        system="You are a concise Python expert. Answer in 2-3 sentences max.",
        messages=[
            {"role": "user", "content": "What is a Python list comprehension?"}
        ]
    )

    print("Response:", response.content[0].text)
    print("\nToken usage:")
    print(f"  Input tokens:  {response.usage.input_tokens}")
    print(f"  Output tokens: {response.usage.output_tokens}")
    print(f"  Stop reason:   {response.stop_reason}")

basic_call()


# =================================================================================

# System Prompt

def system_prompt_demo():
    print("=== Part 2: System Prompts ===")

    # Same question, two different system prompts
    question = "What is recursion?"

    for label, system in [
        ("Teacher",    "You are a Python teacher for beginners. Use a simple real-world analogy. Max 3 sentences."),
        ("Interviewer","You are a senior engineer. Give a technical definition with time complexity implications. Max 2 sentences."),
    ]:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=200,
            system=system,
            messages=[{"role": "user", "content": question}]
        )
        print(f"\n[{label}]")
        print(response.content[0].text)

system_prompt_demo()

# =================================================================================

# Streaming

def streaming_call():
    print("=== Part 3: Streaming ===")
    print("Response: ", end="", flush=True)

    with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=300,
        system="You are a Python teacher.",
        messages=[
            {"role": "user", "content": "Explain Python generators in simple terms."}
        ]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    print("\n\nDone.")

streaming_call()

# =================================================================================

# Context window

def token_counting_demo():
    print("=== Part 4: Token Counting ===")

    system = "You are a Python expert."
    messages = [{"role": "user", "content": "Explain Python generators in detail."}]

    # Count before sending
    count_response = client.beta.messages.count_tokens(
        model="claude-haiku-4-5",
        system=system,
        messages=messages
    )
    print(f"Tokens this call will cost: {count_response.input_tokens}")

    # Now actually send
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        system=system,
        messages=messages
    )
    print(f"Actual input tokens:  {response.usage.input_tokens}")
    print(f"Actual output tokens: {response.usage.output_tokens}")
    print(f"Total tokens used:    {response.usage.input_tokens + response.usage.output_tokens}")

    # Show trim function
    print("\n--- History Trim Demo ---")
    fake_history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
                    for i in range(20)]
    print(f"Before trim: {len(fake_history)} messages")
    trimmed = fake_history[-10:]
    print(f"After trim:  {len(trimmed)} messages")

token_counting_demo()


# =================================================================================

# Rate Limits and Exponential Backoff 

import time

def call_with_backoff(prompt, max_retries=4):
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        except anthropic.RateLimitError:
            wait = 2 ** attempt          # 1, 2, 4, 8 seconds
            print(f"  Rate limited (attempt {attempt + 1}). Waiting {wait}s...")
            time.sleep(wait)

        except anthropic.APIStatusError as e:
            print(f"  API error {e.status_code}: {e.message}")
            raise    # don't retry on non-rate-limit errors

    raise RuntimeError("Max retries exceeded — giving up")

def rate_limit_demo():
    print("=== Part 5: Rate Limit Backoff ===")
    result = call_with_backoff("What is Python in one sentence?")
    print(f"Result: {result}")

rate_limit_demo()

# =================================================================================

# Rate Limits + Exponential Backoff

import time

def call_with_backoff(prompt, max_retries=4):
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        except anthropic.RateLimitError:
            wait = 2 ** attempt          # 1, 2, 4, 8 seconds
            print(f"  Rate limited (attempt {attempt + 1}). Waiting {wait}s...")
            time.sleep(wait)

        except anthropic.APIStatusError as e:
            print(f"  API error {e.status_code}: {e.message}")
            raise    # don't retry on non-rate-limit errors

    raise RuntimeError("Max retries exceeded — giving up")

def rate_limit_demo():
    print("=== Part 5: Rate Limit Backoff ===")
    result = call_with_backoff("What is Python in one sentence?")
    print(f"Result: {result}")

rate_limit_demo()

# =================================================================================

# Model Selection

def model_comparison():
    print("=== Part 6: Model Selection ===")

    prompt = "Is Python pass-by-reference or pass-by-value? One sentence."

    for model_id, label in [
        ("claude-haiku-4-5", "Haiku "),
        ("claude-opus-4-8",  "Opus"),
    ]:
        response = client.messages.create(
            model=model_id,
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}]
        )
        total = response.usage.input_tokens + response.usage.output_tokens
        print(f"{label} ({total} tokens): {response.content[0].text.strip()}")

model_comparison()

# =================================================================================

# Multi turn conversations - store history

def multi_turn_demo():
    print("=== Part 7: Multi-turn Conversation ===")

    history = []
    system = "You are a Python tutor. Keep answers under 3 sentences."

    turns = [
        "What is a Python generator?",
        "How is it different from a list?",
        "Show me a one-line example.",
    ]

    for user_msg in turns:
        history.append({"role": "user", "content": user_msg})

        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=150,
            system=system,
            messages=history
        )

        reply = response.content[0].text
        history.append({"role": "assistant", "content": reply})

        print(f"\nUser:   {user_msg}")
        print(f"Claude: {reply}")

    print(f"\nFinal history length: {len(history)} messages")
    print(f"Turn 3 sent {len(history)} messages to the API vs Turn 1 sent 1")

multi_turn_demo()
