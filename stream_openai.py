import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a concise, witty assistant."},
        {
            "role": "user",
            "content": "Tell me a 4-line poem about returning to tech after parental leave.",
        },
    ],
    temperature=1.5,
    stream=True,
)

print("OpenAI (streaming) response:")
print("-" * 60)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print("\n" + "-" * 60)
