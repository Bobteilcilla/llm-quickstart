import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=200,
    system="You are a concise, witty assistant.",
    messages=[
        {"role": "user", "content": "In one sentence: why should I learn LLM engineering in 2026?"},
    ],
)

print("=" * 60)
print("Claude says:")
print("=" * 60)
print(response.content[0].text)
print()
print(f"Input tokens: {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
