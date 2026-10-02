# rag-chat-assistant

A grounded question-answering service for customer support. It answers only from a set of
help-centre articles, cites the article it used, and declines when they do not cover the
question.

The point of the project is the guardrails, not the retrieval. A support bot that
confidently invents a refund window is worse than one that says it does not know, so the
pipeline is built so a wrong citation cannot reach the customer.

```bash
cp .env.example .env        # then put your OpenAI key in it
docker compose up --build   # serves on localhost:5001
BASE_URL=http://localhost:5001 python smoke_test.py
```

Or run it directly, which serves on `localhost:5000`:

```bash
pip install -r requirements.txt && python app.py
python smoke_test.py
```

Either way `.env` is picked up, and real environment variables override it.

## The contract

```json
{
  "answer": "Annual plans are discounted 20% against monthly billing.",
  "sources": ["billing.md"],
  "debug": {
    "refused_by": null, "top_score": 0.4148, "original_answer": null,
    "retrieved": [{"id": "c1", "source": "billing.md", "score": 0.4148}],
    "grounding": {
      "mode": "shadow", "supported": true, "claims_checked": 1,
      "unsupported_claims": [], "figures_not_in_source": [], "error": null
    }
  }
}
```

`debug` carries the retrieval trace, plus the model's own wording when it declines.

## Four guardrails

The knowledge base is chunked and embedded once at start-up. A question comes in, only the
question is embedded, cosine similarity is computed against every chunk, and they are sorted
best first. Then:

1. **Retrieval gate, in code.** If the best chunk does not clear a similarity threshold, the
   question is refused with no chat call at all. Cheapest possible refusal.
2. **Strict JSON schema, from the model.** The model declares in a field whether the excerpts
   settle the question, rather than the service parsing intent out of prose.
3. **Citation verification, in code.** The model only ever sees opaque chunk ids. Cited ids
   are resolved against what was actually retrieved, so an invented id maps to nothing and is
   dropped. An answer left with no resolvable citation is discarded rather than returned
   uncited.

4. **Grounding check.** The first three prove the answer cites an article it was really
   shown. None of them prove the text is true to it, so a reply can cite `cancellation.md`
   correctly and still say 45 days where the article says 30. Every number in the answer is
   checked against the cited excerpts in code, then a cheaper model breaks the answer into
   claims and marks each one.

Steps 1 and 3 are pure code, which is what makes them guarantees rather than instructions a
model may ignore. Step 4 is a self-report, so it is treated as one: it fails closed, it is
validated against deliberately broken answers, and it ships in shadow.

### Grounding modes

`JUDGE_MODE` is `off`, `shadow` or `enforce`, defaulting to shadow. Shadow runs the check,
records what it found, and still returns the answer, so it can be measured on real traffic
before it is allowed to withhold anything. It costs one extra call per answered question.

```bash
python evaluate.py --judge
```

Six known-bad examples, each pairing a correct answer with a mutation of it: a changed
figure, a dropped condition, a widened scope, an invented detail, a flipped rule, a
plausible addition. It reports false alarms on the correct answers and misses on the
mutations, because a check that flags everything and a check that flags nothing both score
perfectly on one of those numbers alone.

## Web interface

```bash
.venv/bin/python app.py      # API on :5000
cd web && npm install && npm run dev   # UI on :3041
```

The UI is deliberately not a chat window. The backend is single-turn by design, and a
transcript would imply a memory it does not have. It asks one question and shows what the
pipeline did with it: the answer, the article it was drawn from, and every retrieved chunk
plotted against the gate threshold on a fixed cosine axis.

That chart is the point. You can watch a question clear the gate and get answered, clear
the gate and still get declined by the model, or fall short of the gate and never reach the
model at all. Those are the three guardrails, visible rather than described.

`vite.config.ts` proxies `/ask` and `/health` to the API, so the browser stays on one
origin and the service needs no CORS handling. Point it elsewhere with `API_URL`.

## The knowledge base

`data/` holds six articles for Tessera, a fictional team-collaboration SaaS. The company name
lives in one constant in `rag/prompts.py`, so pointing the service at a real corpus is that
line plus new markdown files in `data/`.

## Testing

```bash
pip install -r requirements-dev.txt && pytest
```

82 tests, no network and no API key. The embedder and generator are injected, so the real
pipeline runs against fakes. `tests/test_answer.py` pins guardrails 1 to 3, `tests/test_judge.py` pins the grounding check by mutating correct answers and asserting it notices.

## Evaluation

```bash
python evaluate.py --sweep       # how the gate threshold was chosen
python evaluate.py --retrieval   # recall@k, embeddings only
python evaluate.py               # full pipeline, tuning split
python evaluate.py --holdout     # full pipeline, held-out split
python evaluate.py --judge       # grounding check against known-bad answers
```

The labelled questions in `eval_questions.json` are split in two. `tune` (20 covered, 10
uncovered) is what `top_k` and `min_score` were chosen against. `holdout` (8 covered, 5
uncovered) was written first and left untouched while tuning.

That split is the point. Reporting a score on the same questions you tuned the thresholds
against measures fit to those questions, not how the system behaves on anything new, so
`--sweep` refuses to run on the holdout split.

Covered questions are worded the way a customer would ask rather than the way the article is
written, so retrieval has to do semantic work instead of matching keywords. Uncovered
questions sit deliberately close to a real article, because an absurd question is not what
breaks a refusal guardrail: "is there a free trial" next to `billing.md` is.

## Configuration

Read once at start-up and validated, so a bad value fails immediately.

| Variable | Default |
|---|---|
| `LLM_API_KEY` | required |
| `LLM_API_BASE` | `https://api.openai.com/v1` |
| `LLM_CHAT_MODEL` | `gpt-4o`, needs strict JSON schema support |
| `LLM_EMBED_MODEL` | `text-embedding-3-small` |
| `RAG_TOP_K` | `4` |
| `RAG_MIN_SCORE` | `0.25` |
| `LLM_JUDGE_MODEL` | `gpt-4o-mini` |
| `JUDGE_MODE` | `shadow`, one of `off` / `shadow` / `enforce` |

## Layout

`rag/` is the pipeline: `answer.py` (the guardrails in order), `judge.py` (the grounding
check), `index.py` (chunking and the vector index), plus prompts, schema and config. `app.py` is Flask only, `llm.py` is the
provider client, `evaluate.py` is the labelled eval.

`rag/` imports no HTTP and no provider, so swapping provider means rewriting the two methods
in `llm.py`.

Reasoning behind the design choices is in [DECISIONS.md](DECISIONS.md).
