import os

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

SYSTEM_PROMPT = """You are a senior software engineer mentoring someone
transitioning into AI engineering.

Your style:
- Concise. Never use 5 sentences when 2 will do.
- Concrete. Use real examples, not abstractions.
- Honest. If something is overrated, say so.

Format your answer in this XML structure:
<answer>
  <core_idea>One-sentence answer.</core_idea>
  <example>One concrete example.</example>
  <gotcha>One thing most people get wrong.</gotcha>
</answer>"""

QUESTION = "What is RAG, and when should I NOT use it?"


def ask_openai(q: str) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ],
    )
    return r.choices[0].message.content


def ask_anthropic(q: str) -> str:
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": q}],
    )
    return r.content[0].text


if __name__ == "__main__":
    console.print(Panel(QUESTION, title="Question", style="bold cyan"))
    console.print(Panel(ask_openai(QUESTION), title="OpenAI (gpt-4o-mini)", style="green"))
    console.print(Panel(ask_anthropic(QUESTION), title="Claude Haiku 4.5", style="magenta"))
