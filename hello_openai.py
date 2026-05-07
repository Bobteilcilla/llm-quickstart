import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a concise, witty assistant."},
        {"role": "user", "content": "In one sentence: why should I learn LLM engineering in 2026?"},
    ],
    temperature=0.7,
)

print("=" * 60)
print("OpenAI says:")
print(response.choices[0].message.content)
print()
print(f"Token usage: {response.usage.total_tokens} tokens")
print(f"Model used: {response.model}")
print("=" * 60)
