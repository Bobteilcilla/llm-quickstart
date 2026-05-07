# llm-quickstart — Strategy 1-Pager

> A learning project. The "user" is me, and the "product" is my own muscle memory.
> I'm filling this out so the format is automatic by Week 6.

## 1. The user
- **Persona**: Erica — Software Engineer returning to tech, transitioning into LLM/GenAI Engineering.
- **Job to be done**: Internalise the basic shape of OpenAI and Anthropic SDKs so I can reach for them without thinking.

## 2. The pain
- **Today**: I've never made a real LLM API call. Everything I "know" is from tutorials.
- **Cost**: Without this, I'd start Week 4 (prompting) with no muscle memory and waste a day on setup that should be reflexive.

## 3. Why AI
- **Why an LLM specifically?** This is literally a project about LLMs. (Meta, but real.)
- **Failure mode if AI is wrong**: Zero — this is a practice project. Wrong answers are part of learning.

## 4. What "good" looks like
- **In 1 sentence**: I can write a script that calls OpenAI or Anthropic from memory in <5 minutes by end of Week 3.
- **By when**: End of Week 3 (May 28, 2026).

## 5. KPIs
- **North-star**: A reusable mental model for "how do I call any LLM SDK?"
- **Product KPI**: I can rebuild `compare.py` from a blank file in <10 minutes.
- **Model KPI**: Both calls return non-empty responses with token counts logged.

## 6. Guardrails (v1 scope)
- ✅ Input validation: pydantic on inputs (Week 2 task)
- ✅ Logging: print model name and token counts on every call
- ⏭️ Output validation: deferred to Week 4 (structured outputs)
- ⏭️ Rate limiting: not relevant for a personal learning script
- ⏭️ Retrieval grounding: no retrieval in this project
- ⏭️ Human-in-the-loop: not relevant

## 7. MVP scope (MoSCoW)
| Feature | Bucket |
|---|---|
| `hello_openai.py` | M (done) |
| `hello_anthropic.py` | M (done) |
| `compare.py` side-by-side | M (done) |
| Streaming responses | S (Day 2) |
| Structured outputs with pydantic | S (Day 2-3) |
| Streamlit UI | C (Week 3) |
| Cost tracking | C (Week 3) |
| Async batching | W (not this project) |
| Multi-turn memory | W (not this project) |
