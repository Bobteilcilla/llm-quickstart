"""Find the most semantically similar sentence
from a list of candidates"""

from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer

app = typer.Typer(help="Find the most semantically similar sentence from a list of candidates.")
console = Console()

CANDIDATES = [
    "Take this medication twice a day with food.",
    "The pitcher threw a perfect strike to end the inning.",
    "Apply the cream to the affected area three times daily.",
    "Both teams went into extra innings after the tied score.",
    "Drink plenty of water and rest until the fever breaks.",
    "The shortstop made an unbelievable diving catch.",
    "Schedule a follow-up appointment in two weeks.",
    "The crowd erupted when the home run cleared the fence.",
]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@app.command()
def main(
    query: Annotated[str, typer.Argument(help="What you are searching for")],
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            "-k",
            help="""How many similar
            sentences to show""",
        ),
    ] = 3,
) -> None:
    """Find the most semantically similar sentence from a list of candidates."""
    console.print(f"Query: [bold]{query}[/bold]\n")

    # 1.Load the embedding model (same one as used in Day 6)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # 2. Encode the query and all candidates
    query_emb = encoder.encode(query)
    candidate_embs = encoder.encode(CANDIDATES)

    # 3. Calculate similarity scores between the query and each candidate
    scores = [(c, _cosine(query_emb, e)) for c, e in zip(CANDIDATES, candidate_embs, strict=False)]

    # 4. Sort candidates by similarity and show the top K
    scores.sort(key=lambda x: x[1], reverse=True)

    # 5. Display results in a table
    table = Table(title=f"Top {top_k} Similar Sentences")
    table.add_column("Score", style="cyan")
    table.add_column("Candidate", style="magenta")
    for sentence, score in scores[:top_k]:
        table.add_row(f"{score:.4f}", sentence)
    console.print(table)


if __name__ == "__main__":
    app()
