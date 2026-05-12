# Week 2 Kickoff — ML Refresher

## Reflection on the TF-IDF baseline

### 1. Why does TF-IDF + Logistic Regression work so well on this problem?

The two topics — baseball and medical science — use fundamentally different vocabulary. Baseball articles are dominated by terms like "pitcher," "inning," "strike," "batter," "game," while medical articles use "disease," "patient," "treatment," "symptoms," "antibiotic." These word distributions are so distinct that a simple linear classifier can draw a clean boundary between them in TF-IDF space. The model doesn't need to understand context or meaning—pure statistical word frequency is signal enough. That's why we hit 96.98% accuracy with zero effort.

### 2. What kind of input would break this baseline?

**Synonyms and paraphrasing:** "The ball carrier advanced down the field" uses different vocabulary than "The player ran with the ball," but they mean the same thing. TF-IDF would treat them as different.

**Sarcasm:** "This medical procedure is amazing! 😒" contains positive words but negative intent. TF-IDF sees only the words, not the tone.

**Multilingual or informal text:** Slang, typos, abbreviations, or code-switching would create sparse vectors that don't match the training distribution.

**Cross-domain transfer:** This model trained on baseball vs. medicine would fail on "sports injury" (mixes both topics) or any new domain entirely.

---

## Why embeddings fix this

Embeddings learn **semantic relationships** instead of just word counts. A model trained on embeddings would understand that "player" and "athlete" are similar concepts, that punctuation + positive words can signal sarcasm, and that "injury" bridges both domains. That's the unlock for Week 2.

Tomorrow: beat this baseline with `sentence-transformers`.
