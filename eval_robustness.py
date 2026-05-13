"""Adversarial comparison: TF-IDF vs Embeddings on paraphrased inputs."""

from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

console = Console()

# --- 1. Same training data + split as Days 5 and 6 ---
CATEGORIES = ["rec.sport.baseball", "sci.med"]
DATA = fetch_20newsgroups(
    subset="all", categories=CATEGORIES, remove=("headers", "footers", "quotes")
)
X = DATA.data
y = DATA.target
target_names = DATA.target_names

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Train both models ---

console.print("[cyan]Training TF-IDF model...[/cyan]")

tfidf_model = Pipeline(
    [
        ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000)),
    ]
)
tfidf_model.fit(X_train, y_train)

console.print("[cyan]Training Embeddings model...[/cyan]")

encoder = SentenceTransformer("all-MiniLM-L6-v2")
X_train_emb = encoder.encode(X_train, show_progress_bar=False, batch_size=32)
emb_clf = LogisticRegression(max_iter=1000)
emb_clf.fit(X_train_emb, y_train)

# --- 3. Adversarial test set ---

# Each pair: (original-style sentence, paraphrased sentence using synonyms / different words)
# The label is the same for both — only the wording changes.
ADVERSARIAL = [
    # --- BASEBALL (no obvious baseball words) ---
    {
        "label": "rec.sport.baseball",
        "original": "The pitcher struck out three batters in the ninth inning.",
        "paraphrase": "He threw it past the guy three times in a row to end the game.",
    },
    {
        "label": "rec.sport.baseball",
        "original": "Both teams played extra innings.",
        "paraphrase": "Nobody could break the tie, so the game just kept going.",
    },
    {
        "label": "rec.sport.baseball",
        "original": "The crowd cheered when the runner slid into home.",
        "paraphrase": "The fans went wild when he dove across the line just in time.",
    },
    # --- MEDICINE (no obvious medical words) ---
    {
        "label": "sci.med",
        "original": "The patient was prescribed antibiotics for the infection.",
        "paraphrase": "She got something to take twice a day for the next ten days.",
    },
    {
        "label": "sci.med",
        "original": "Researchers found a new treatment for diabetes.",
        "paraphrase": """They figured out a new way to help people whose bodies don't process
        sugar well.""",
    },
    {
        "label": "sci.med",
        "original": "The study showed reduced side effects in the trial group.",
        "paraphrase": "The people in the experiment had fewer problems than the others.",
    },
]

label_to_id = {name: i for i, name in enumerate(target_names)}

# --- 4. Score both models ---
table = Table(title="Adversarial test: TF-IDF vs Embeddings", show_lines=True)
table.add_column("Sentence", style="cyan", max_width=60)
table.add_column("True label", style="yellow")
table.add_column("TF-IDF guess", style="magenta")
table.add_column("Embeddings guess", style="green")

tfidf_correct = 0
emb_correct = 0
total = 0

for row in ADVERSARIAL:
    true_id = label_to_id[row["label"]]
    for which in ("original", "paraphrase"):
        text = row[which]
        # TF-IDF
        tfidf_pred = tfidf_model.predict([text])[0]
        # Embeddings
        emb = encoder.encode([text])
        emb_pred = emb_clf.predict(emb)[0]

        tag_tfidf = f"{target_names[tfidf_pred]} {'✓' if tfidf_pred == true_id else '✗'}"
        tag_emb = f"{target_names[emb_pred]} {'✓' if emb_pred == true_id else '✗'}"

        table.add_row(
            f"[{which}] {text}",
            target_names[true_id],
            tag_tfidf,
            tag_emb,
        )

        if tfidf_pred == true_id:
            tfidf_correct += 1
        if emb_pred == true_id:
            emb_correct += 1
        total += 1

console.print(table)
console.print(f"\n[bold magenta]TF-IDF correct:[/bold magenta] {tfidf_correct}/{total}")
console.print(f"[bold green]Embeddings correct:[/bold green] {emb_correct}/{total}")
