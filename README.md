# llm-quickstart

My first repo on the journey from Software Engineer → AI/LLM Engineer.
Returning to tech after parental leave with a focused 16-week intensive plan.
This is Day 2 of 112.

## What's here

- `hello_openai.py` — first call to OpenAI's `gpt-4o-mini`
- `hello_anthropic.py` — first call to Anthropic's Claude Haiku 4.5
- `compare.py` — same question, both models, side-by-side output
- `compare_with_systemprompt.py` — same question, both models, with system prompts and XML structure
- `stream_openai.py` / `stream_anthropic.py` — streaming responses (tokens appear live)
- `extract_openai.py` / `extract_anthropic.py` — **structured outputs** with pydantic
- `tests/` — smoke tests

## Setup

\`\`\`bash
# 1. Install uv (https://docs.astral.sh/uv/)
# 2. Clone the repo
# 3. Install deps
uv sync

# 4. Add your API keys to .env
cp .env.example .env  # then edit with real keys

# 5. Run something
uv run python compare.py
\`\`\`

## Stack

`uv` · `openai` · `anthropic` · `pydantic` · `typer` · `rich` · `pytest` · `ruff` · `pre-commit`

## Following along?

I'm posting weekly progress on LinkedIn — connect with me here:
[Erica Gonzalez Gutierrez](https://www.linkedin.com/in/erica-gonzalez-gutierrez-88876955/)
