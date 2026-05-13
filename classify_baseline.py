"""Baseline text classifier -TF + IDF (Term Frequency-Inverse Document Frequency)+
Logistic Regression"""

from rich.console import Console

# --- 1. Get some data ---
# We'll use a tiny built-in dataset: spam vs not-spam from sklearn's fetch_20newsgroups
# (treating one newsgroup as 'spam', another as 'ham' for a quick baseline)
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

console = Console()


def build_model():
    CATEGORIES = ["rec.sport.baseball", "sci.med"]  # baseball = ham, sci.med = spam
    DATA = fetch_20newsgroups(
        subset="all", categories=CATEGORIES, remove=("headers", "footers", "quotes")
    )

    X = DATA.data
    y = DATA.target
    target_names = DATA.target_names

    # --- 2. Train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. Pipeline: TF-IDF vectoriser + Logistic Regression ---
    tfidf = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), stop_words="english"),
            ),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    tfidf.fit(X_train, y_train)
    return tfidf, X_train, X_test, y_train, y_test, target_names


if __name__ == "__main__":
    model, X_train, X_test, y_train, y_test, target_names = build_model()
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    console.print(f"[green]Baseline Accuracy:[/green] {accuracy:.4f}\n")
    console.print("[yellow]Classification Report:[/yellow]")
    console.print(classification_report(y_test, preds, target_names=target_names))

    # --- 5. Try it on a custom string ---
    samples = [
        "The pitcher struck out three batters in the ninth inning.",
        "The patient was prescribed antibiotics for the infection.",
        "Both teams played extra innings.",
    ]
    for s in samples:
        pred = model.predict([s])[0]
        proba = model.predict_proba([s])[0]
        console.print(f"\n[cyan]Text:[/cyan] {s}")
        console.print(
            f"[yellow]Predicted:[/yellow] {target_names[pred]}  (confidence: {proba.max():.2%})"
        )
