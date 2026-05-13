"""Embeddings classifier - sentence-transformers + Logistic Regression. Day 6"""

from rich.console import Console
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

console = Console()


def build_model():
    # --- 1. Same data as Day 5 ---
    CATEGORIES = ["rec.sport.baseball", "sci.med"]  # baseball = ham, sci.med = spam
    DATA = fetch_20newsgroups(
        subset="all", categories=CATEGORIES, remove=("headers", "footers", "quotes")
    )
    X = DATA.data
    y = DATA.target
    target_names = DATA.target_names

    # --- 2. Train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. Get sentence embeddings ---
    encoder = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )  # 384-dimensional embeddings, smll, fast, good

    X_train_emb = encoder.encode(X_train, show_progress_bar=True)
    X_test_emb = encoder.encode(X_test, show_progress_bar=True)

    # --- 4. Train a Logistic Regression classifier on the embeddings ---
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_emb, y_train)
    return clf, encoder, X_train_emb, X_test_emb, y_train, y_test, target_names


if __name__ == "__main__":
    model, encoder, X_train_emb, X_test_emb, y_train, y_test, target_names = build_model()

    # ---5. Evaluate ---
    preds = model.predict(X_test_emb)
    accuracy = accuracy_score(y_test, preds)
    console.print(f"[green]Embeddings Classifier Accuracy:[/green] {accuracy:.4f}\n")
    console.print("[yellow]Classification Report:[/yellow]")
    console.print(classification_report(y_test, preds, target_names=target_names))

    # ---6. Try it on a custom string ---
    samples = [
        "The pitcher struck out three batters in the ninth inning.",
        "The patient was prescribed antibiotics for the infection.",
        "Both teams played extra innings.",
    ]
    sample_emb = encoder.encode(samples)
    for s, p, proba in zip(
        samples, model.predict(sample_emb), model.predict_proba(sample_emb), strict=False
    ):
        console.print(f"\n[cyan]Text:[/cyan] {s}")
        console.print(
            f"[yellow]Predicted:[/yellow] {target_names[p]}  (confidence: {proba.max():.2%})"
        )
