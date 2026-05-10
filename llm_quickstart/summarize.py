"""Summarize a text file - streaming or structured JSON output."""

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from rich.console import Console

load_dotenv()
app = typer.Typer(help="Summarise a text file - streaming or structured JSON output.")
console = Console()


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

        if llm == "anthropic":
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1000,
                system=(
                    "You summarise documents into structured JSON only. "
                    "Respond ONLY with valid JSON matching this schema. "
                    "No prose, no markdown fences, no backticks.\n\n"
                    f"Schema:\n{schema}"
                ),
                messages=[{"role": "user", "content": text}],
            )
            json_text = response.content[0].text.strip()
        elif llm == "openai":
            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=1000,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You summarise documents into structured JSON only. "
                            "Respond ONLY with valid JSON matching this schema. "
                            "No prose, no markdown fences, no backticks.\n\n"
                            f"Schema:\n{schema}"
                        ),
                    },
                    {"role": "user", "content": text},
                ],
            )
            json_text = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if json_text.startswith("```"):
            json_text = json_text.split("```")[1]
            if json_text.startswith("json"):
                json_text = json_text[4:]
            json_text = json_text.strip()
        result = Summary.model_validate_json(json_text)
        console.print_json(data=result.model_dump())
        return

    # Streaming path
    length_hint = {
        "short": "a brief summary in 1-2 sentences",
        "medium": "a concise summary in 3-5 sentences",
        "long": "1-2 paragraphs summarizing the main points",
    }.get(length, "a concise summary in 3-5 sentences")

    console.print(f"[bold cyan]Generating a {length} summary...[/bold cyan]")

    if llm == "anthropic":
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
    elif llm == "openai":
        response = client.chat.completions.create(
            model="gpt-4o",
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
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
        print()


if __name__ == "__main__":
    app()
