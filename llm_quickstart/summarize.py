"""Summarize a text file - streaming or structured JSON output."""

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console

load_dotenv()
app = typer.Typer(help="Summarise a text file - streaming or structured JSON output.")
console = Console()
err_console = Console(stderr=True)

# Pricing per million tokens (May 2026, rough)
_PRICING_PER_MILLION = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def _log_cost(model: str, input_tokens: int, output_tokens: int) -> None:
    """Print token usage + estimated $ cost to stderr."""
    prices = _PRICING_PER_MILLION.get(model, {"input": 0, "output": 0})
    cost = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000
    err_console.print(
        f"[dim][cost] {model}  in={input_tokens}  out={output_tokens}  ~${cost:.6f}[/dim]"
    )


class Summary(BaseModel):
    """Structured summary of a document"""

    title: str = Field(description="A 4-8 word title summarizing the document.")
    one_liner: str = Field(description="A one-liner summary of the document.")
    key_points: list[str] = Field(description="3-5 main points from the document.")
    audience: str = Field(description="The target audience for the summary.")


def _read_file(file_path: Path) -> str:
    """Read the contents of a text file."""
    if not file_path.exists():
        console.print(f"[red]Error: File not found - {file_path}[/red]")
        raise typer.Exit(code=1)
    return file_path.read_text(encoding="utf-8")


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from the text."""
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        return text.strip()
    return text.strip()


def _anthropic_chat_json_completion(client: Anthropic, text: str, schema: str) -> Summary:
    """Get structured JSON summary from Anthropic. Retries once on invalid JSON."""
    system_prompt = (
        "You summarise documents into structured JSON only. "
        "Respond ONLY with valid JSON matching this schema. "
        "No prose, no markdown fences, no backticks.\n\n"
        f"Schema:\n{schema}"
    )
    messages = [{"role": "user", "content": text}]

    # First attempt
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        system=system_prompt,
        messages=messages,
    )
    raw_json = _strip_markdown_fences(response.content[0].text)
    _log_cost(
        "claude-haiku-4-5-20251001", response.usage.input_tokens, response.usage.output_tokens
    )
    try:
        return Summary.model_validate_json(raw_json)
    except ValidationError as e:
        console.print("[yellow]Invalid JSON — retrying with self-correction...[/yellow]")
        # Continue the conversation: show the model its mistake and ask for a fix
        messages.append({"role": "assistant", "content": raw_json})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"That response was not valid JSON. Error:\n{e}\n\n"
                    "Return ONLY valid JSON matching the schema. No prose, no markdown fences."
                ),
            }
        )
        retry_response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            system=system_prompt,
            messages=messages,
        )
        retry_json = _strip_markdown_fences(retry_response.content[0].text)
    _log_cost(
        "claude-haiku-4-5-20251001",
        retry_response.usage.input_tokens,
        retry_response.usage.output_tokens,
    )
    return Summary.model_validate_json(retry_json)  # if this fails too, error propagates


def _openai_chat_json_completion(client: OpenAI, text: str, schema: str) -> Summary:
    """Get structured JSON summary from OpenAI. Retries once on invalid JSON."""
    system_prompt = (
        "You summarise documents into structured JSON only. "
        "Respond ONLY with valid JSON matching this schema. "
        "No prose, no markdown fences, no backticks.\n\n"
        f"Schema:\n{schema}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    # First attempt
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1000,
        messages=messages,
    )
    raw_json = _strip_markdown_fences(response.choices[0].message.content)
    _log_cost("gpt-4o", response.usage.input_tokens, response.usage.output_tokens)
    try:
        return Summary.model_validate_json(raw_json)
    except ValidationError as e:
        console.print("[yellow]Invalid JSON — retrying with self-correction...[/yellow]")
        messages.append({"role": "assistant", "content": raw_json})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"That response was not valid JSON. Error:\n{e}\n\n"
                    "Return ONLY valid JSON matching the schema. No prose, no markdown fences."
                ),
            }
        )
        retry_response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1000,
            messages=messages,
        )
        retry_json = _strip_markdown_fences(retry_response.choices[0].message.content)
    _log_cost("gpt-4o", retry_response.usage.input_tokens, retry_response.usage.output_tokens)
    return Summary.model_validate_json(retry_json)


def _anthropic_chat_streaming_completion(client: Anthropic, text: str, length_hint: str) -> None:
    """Stream a summary from Anthropic."""
    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=(
            "You write warm, concise, high-signal summaries. "
            f"Target length: {length_hint}. "
            "Lead with the single most important takeaway."
        ),
        messages=[{"role": "user", "content": text}],
    ) as stream:
        for delta in stream.text_stream:
            print(delta, end="", flush=True)
    print()
    _log_cost(
        "claude-haiku-4-5-20251001",
        stream.get_final_message().usage.input_tokens,
        stream.get_final_message().usage.output_tokens,
    )


def _openai_chat_streaming_completion(client: OpenAI, text: str, length_hint: str) -> None:
    """Stream a summary from OpenAI."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=600,
        stream=True,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write warm, concise, high-signal summaries. "
                    f"Target length: {length_hint}. "
                    "Lead with the single most important takeaway."
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    final_usage = None
    for chunk in response:
        # The very last chunk has chunk.usage set and chunk.choices empty.
        if chunk.usage is not None:
            final_usage = chunk.usage
            continue
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()
    if final_usage:
        _log_cost("gpt-4o-mini", final_usage.prompt_tokens, final_usage.completion_tokens)


@app.command()
def main(
    file: Annotated[Path, typer.Argument(help="Path to the text file to summarize.")],
    llm: Annotated[str, typer.Option("--model", help="openai | anthropic")] = "anthropic",
    length: Annotated[str, typer.Option(help="short | medium | long")] = "medium",
    json_output: Annotated[
        bool, typer.Option("--json", help="Return structured JSON output.")
    ] = False,
) -> None:
    """Summarise a text file with optional structured JSON output."""
    text = _read_file(file)
    if llm == "anthropic":
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    elif llm == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        raise typer.BadParameter("Invalid LLM. Please choose 'anthropic' or 'openai'.")

    if json_output:
        # Structured JSON path
        schema = json.dumps(Summary.model_json_schema(), indent=2)
        try:
            # First attempt - the user's chosen provider
            if llm == "anthropic":
                result = _anthropic_chat_json_completion(client, text, schema)
            elif llm == "openai":
                result = _openai_chat_json_completion(client, text, schema)
        except Exception as e:
            console.print(f"[red]Primary provider error: {e}[/red]")
            console.print("[yellow]Falling back to the other provider...[/yellow]")  # ADD THIS LINE
            if llm == "anthropic":
                fallback_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                result = _openai_chat_json_completion(fallback_client, text, schema)
            elif llm == "openai":
                fallback_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                result = _anthropic_chat_json_completion(fallback_client, text, schema)

        console.print_json(data=result.model_dump())
        return

    # Streaming path
    length_hint = {
        "short": "a brief summary in 1-2 sentences",
        "medium": "a concise summary in 3-5 sentences",
        "long": "1-2 paragraphs summarizing the main points",
    }.get(length, "a concise summary in 3-5 sentences")

    console.print(f"[bold cyan]Generating a {length} summary...[/bold cyan]")
    try:
        if llm == "anthropic":
            _anthropic_chat_streaming_completion(client, text, length_hint)
        elif llm == "openai":
            _openai_chat_streaming_completion(client, text, length_hint)
    except Exception as e:
        console.print(f"[red]Primary provider error: {e}[/red]")
        console.print("[yellow]Falling back to the other provider...[/yellow]")
        if llm == "anthropic":
            fallback_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            _openai_chat_streaming_completion(fallback_client, text, length_hint)
        elif llm == "openai":
            fallback_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            _anthropic_chat_streaming_completion(fallback_client, text, length_hint)


if __name__ == "__main__":
    app()
