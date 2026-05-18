from typing import Annotated

import tiktoken
import typer
from rich.console import Console
from rich.text import Text

app = typer.Typer(help="A simple tool to visualize tokenization of text using tiktoken.")
console = Console()

# Colours the cycle as we walk through the tokens
PALETTE = ["red", "green", "blue", "yellow", "magenta", "cyan"]


def _show_tokens(text: str, encoding_name: str) -> None:
    encoding = tiktoken.get_encoding(encoding_name)
    tokens_ids = encoding.encode(text)
    tokens_strs = [encoding.decode([tid]) for tid in tokens_ids]

    console.print(f"Encoding: [bold]{encoding_name}[/bold]")
    console.print(f"Text: [bold]{text}[/bold]" + f" | Characters: [dim]{len(text)}[/dim]\n")

    # Render each token in alternate colours
    line = Text()
    for i, t in enumerate(tokens_strs):
        line.append(t, style=PALETTE[i % len(PALETTE)])
    console.print(line)

    # Also show the token ids
    console.print(f"Token IDs: [bold]{tokens_ids}[/bold]\n\n")


@app.command()
def main(
    text: Annotated[str, typer.Argument(help="The text to tokenize and visualize.")],
    encoding: Annotated[
        str,
        typer.Option(
            "--encoding", help="cl100k_base (GPT-4) | o200k_base (GPT-4o) | p50k_base (older)"
        ),
    ] = "o200k_base",
    compare: Annotated[
        bool, typer.Option("--compare", "-c", help="Show all three encodings")
    ] = False,
) -> None:
    """Show how TEXT is split into tokens by an OpenAi tokeniser"""
    encodings = ["cl100k_base", "o200k_base", "p50k_base"] if compare else [encoding]
    for enc_name in encodings:
        _show_tokens(text, enc_name)


if __name__ == "__main__":
    app()
