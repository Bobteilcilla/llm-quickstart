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
- `llm_quickstart\summarize.py` - A command-line tool to summarize a text in a json output or streaming.

## CLI: `summarize`

Summarise any text/markdown file with streaming or structured JSON output.

### Usage

\`\`\`bash
# Streaming summary (default: medium length, Anthropic/Claude)
uv run summarize README.md

# Structured JSON output
uv run summarize README.md --json

# Control summary length
uv run summarize README.md --length short      # 1-2 sentences
uv run summarize README.md --length medium     # 3-5 sentences (default)
uv run summarize README.md --length long       # 1-2 paragraphs

# Choose LLM provider
uv run summarize README.md --model anthropic   # Claude Haiku (default)
uv run summarize README.md --model openai      # GPT-4o (JSON) or GPT-4o-mini (streaming)

# Combine options
uv run summarize README.md --model openai --length short --json
\`\`\`

### Features

- **Dual LLM support**: Anthropic (Claude Haiku) or OpenAI (GPT-4o/mini)
- **Two output modes**: Streaming (live tokens) or structured JSON with title, one-liner, key points, and audience
- **Smart retries**: Invalid JSON responses trigger automatic self-correction
- **Cost tracking**: Token usage and estimated costs logged to stderr
- **Fallback handling**: If primary provider fails, automatically tries the other

## Production patterns demonstrated

- **Multi-provider abstraction** — Anthropic + OpenAI swappable via `--model`
- **Streaming + structured outputs** — both response modes in one tool
- **Fallback on API errors** — automatic retry on the other provider
- **Self-correcting JSON** — retries with the validation error on invalid output
- **Per-call cost logging** — token counts + $ cost on stderr (pipe-safe)

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
