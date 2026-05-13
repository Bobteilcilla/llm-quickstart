# Week 2 — Embeddings vs TF-IDF

A working journal of my Week 2 ML refresher. Two classifiers, same data, three increasingly hard tests — and what those tests actually revealed.

## Setup

- **Dataset:** `fetch_20newsgroups`, two classes: `rec.sport.baseball` vs `sci.med`
- **Split:** 80 / 20 train/test, `random_state=42` (same split for both models)
- **Models:**
  - **TF-IDF + Logistic Regression** (`classify_baseline.py`)
  - **`all-MiniLM-L6-v2` embeddings + Logistic Regression** (`classify_embeddings.py`)

## Day 6 — Results

| Model       | Clean test set | Adversarial v1 (leaky paraphrases) | Adversarial v2 (no domain vocab) |
|-------------|----------------|------------------------------------|----------------------------------|
| TF-IDF      | 96.98%         | 8 / 8                              | 11 / 12                          |
| Embeddings  | 96.98%         | 8 / 8                              | 12 / 12                          |

## What I actually learned (in order)

### 1. On the clean test set, both models tied

Both hit 96.98%. By that one number alone, the two approaches look interchangeable — and most tutorials would stop here.

The instinctive next move would be to ship TF-IDF: it's faster, cheaper, simpler, and has identical accuracy. That instinct is wrong, but you cannot see why from this number.

### 2. My first adversarial test was too easy

I wrote 8 "paraphrased" sentences expecting TF-IDF to break on them. Both models got all 8 right.

Looking back at the sentences:

> *"The thrower retired three **hitters** in the final period of the game."*
> *"The **doctor** recommended a course of **medication** to fight the **illness**."*
> *"Scientists discovered a novel **therapy** for **blood sugar** problems."*

The "paraphrases" still leaked domain vocabulary. TF-IDF only needs **one distinctive word** to win — and I gave it several.

**Lesson:** if your eval set shares vocabulary with the training data, you can't tell whether the model has learned meaning or just memorised words.

### 3. The real adversarial test

I rewrote the paraphrases to remove **all** domain-specific words. Examples:

> *"She got something to take twice a day for the next ten days."*  → medicine
> *"Nobody could break the tie, so the game just kept going."* → baseball
> *"The people in the experiment had fewer problems than the others."* → medicine

Now the only signal is **meaning**, not vocabulary.

- **TF-IDF: 11 / 12** — failed on the "twice a day" sentence.
- **Embeddings: 12 / 12** — handled all of them.

### The one TF-IDF failure

> *"She got something to take twice a day for the next ten days."*

- No medical-distinctive words at all
- Words like `twice`, `day`, `ten` happen to lean baseball in this training corpus
- The embedding model recognised `"take twice a day"` as a medical pattern from its pre-training, even though no individual word said so

This is the entire point of using embeddings for retrieval.

## The real takeaway

> **The eval set determines what you can see.**

If I had only looked at clean-set accuracy → I would have shipped the wrong model and never known.
If I had only looked at my first adversarial test → I would have concluded both models were equivalent.

Constructing inputs that exercise the model's actual weak point — not the inputs that happen to come with the dataset — is the skill that separates "I ran a classifier" from "I evaluated a system."

## Why this matters for RAG (Phase 2 preview)

Keyword search always has a failure mode where the query and the document use different words for the same thing. Embeddings smooth over that gap because they encode meaning, not word identity.

That's why every modern RAG system uses embeddings for retrieval — and why "is your retrieval recall actually good?" is a harder question than the surface number suggests.

## Open questions for later

- Would the gap widen if I used a harder dataset (e.g. two CS subfields with overlapping vocabulary)?
- Would `all-mpnet-base-v2` (a larger encoder) close the failure on the "twice a day" sentence's *near misses* I didn't catch?
- How does this play out when I move from classification to retrieval (which is the Phase 2 capstone)?
