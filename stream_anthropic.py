import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

print("Anthropic (streaming) response:")
print("-" * 60)

with client.messages.stream(
    model="claude-haiku-4-5-20251001",
    max_tokens=200,
    system="You are a concise, witty assistant.",
    messages=[
        {
            "role": "user",
            "content": "Tell me a 4-line poem about returning to tech after parental leave.",
        }
    ],
    temperature=1.0,
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

print("\n" + "-" * 60)
