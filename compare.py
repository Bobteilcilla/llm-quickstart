import os

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

QUESTION = "Explain RAG in 2 sentences, like I'm a software engineer new to AI."


def ask_openai(q: str) -> tuple[str, int, str]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a concise, witty assistant."},
            {"role": "user", "content": q},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content, response.usage.total_tokens, response.model


def ask_anthropic(q: str) -> tuple[str, int, int]:
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system="You are a concise, witty assistant.",
        messages=[
            {"role": "user", "content": q},
        ],
    )
    return response.content[0].text, response.usage.input_tokens, response.usage.output_tokens


if __name__ == "__main__":
    # openai_response, openai_tokens, openai_model = ask_openai(QUESTION)
    # anthropic_response, anthropic_input_tokens, anthropic_output_tokens =
    # ask_anthropic(QUESTION)

    # console.print(Panel(f"[bold green]OpenAI ({openai_model})[/bold green]
    # \n\n{openai_response}\n\n[italic]Tokens used: {openai_tokens}[/italic]",
    # title="OpenAI Response", border_style="green"))
    # äconsole.print(Panel(f"[bold blue]Anthropic (claude-haiku-4-5-20251001)[/bold blue]
    # \n\n{anthropic_response}\n\n[italic]Input Tokens: {anthropic_input_tokens} |
    # Output Tokens: {anthropic_output_tokens}[/italic]", title="Anthropic Response",
    # border_style="blue"))

    openai_text, openai_tokens, openai_model = ask_openai(QUESTION)
    anthropic_text, anthropic_input, anthropic_output = ask_anthropic(QUESTION)

    console.print(Panel(QUESTION, title="Question", style="bold cyan"))
    console.print(
        Panel(
            f"{openai_text}\n\n[italic]Tokens: {openai_tokens} | Model: {openai_model}[/italic]",
            title="OpenAI (gpt-4o-mini)",
            style="green",
        )
    )
    console.print(
        Panel(
            f"{anthropic_text}\n\n"
            f"[italic]Input: {anthropic_input} | "
            f"Output: {anthropic_output}[/italic]",
            title="Anthropic (Claude Haiku 4.5)",
            style="magenta",
        )
    )
