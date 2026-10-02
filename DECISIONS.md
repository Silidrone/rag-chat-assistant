# Design notes

## Assumptions

- **Answers go to a customer, not to a support agent.** So they are short, plain, and never
  mention excerpts, chunk ids or retrieval.

- **"Grounded answer with a citation" is two separate guarantees.** One is that the answer
  says only what the excerpts say, which the prompt asks for. The other is that `sources`
  only ever names files the model actually saw, which is enforced in code.

  The model never sees a filename. It sees opaque ids (`c0`, `c1`), cites those, and
  `rag/answer.py` maps them back, dropping any id that was not retrieved for that request.
  It can still invent an id, but an invented id maps to nothing and gets dropped, so the
  failure is loud rather than a believable wrong filename.

- **Ambiguity fails closed.** A confident wrong answer about a refund window costs more
  than an unnecessary "contact support".

- **A negative answer still counts as covered.** "Can a normal team member add someone
  new?" is answerable from `seats_and_roles.md`, which says owners and admins can invite
  people and members cannot. Only questions the knowledge base is silent on get refused.
  That relies on the articles being written as closed rules, which these are ("only",
  explicit exclusion lists).

- **The knowledge base is static.** 2.1 KB, six articles, indexed once at start-up and
  held in memory.

## Pipeline

`embed -> search -> gate -> generate -> verify`, in `rag/answer.py`:

1. **Retrieval gate.** Below a cosine threshold, refuse with no chat call at all. Nothing
   generated, nothing billed.

2. **The model declares coverage in a field.** Strict JSON schema returning
   `{answered, answer, chunk_ids}`, rather than matching the answer text against a fixed
   refusal phrase. The latter breaks the first time the model rewords itself.

3. **Citation verification.** Cited ids are resolved against the retrieved set and unknown
   ids dropped. An answer left with no resolvable citation is discarded rather than
   returned uncited.

4. **Grounding check.** Guardrails 1 to 3 prove the answer cites an article it was really
   shown. They prove nothing about whether the text is true to it, so a reply can cite
   `cancellation.md` correctly and still say 45 days where the article says 30. See below.

Every refusal returns the same fixed message. The model still writes its own decline, since
the schema requires the field, but that text goes to `debug.original_answer` rather than to
the customer.

Retrieval is deliberately simple: cosine over in-memory vectors, no vector database. At this
corpus size a database would be infrastructure without a problem to solve, and the index is
a seam (`rag/index.py`) that can be swapped once there is one.

## The grounding check

Two tiers, cheapest first, in `rag/judge.py`.

**Figures, in code.** Every number in the answer has to appear in the cited excerpts.
No model call, nothing billed, and it covers the class of error that costs the most,
because a number is the part a customer acts on. It is strict about presence and silent
about meaning: it catches 45 where the source says 30, and cannot catch "within 30 days"
turning into "after 30 days".

**Claims, by model.** A second, cheaper model breaks the answer into claims and marks each
one against the cited excerpts only. Three things make this more than a vibe check:

- *The inputs stay narrow.* The checker sees the cited excerpts and nothing else. Hand it
  the whole knowledge base and it will mark a claim supported by some article the answer
  never cited, which is the failure it exists to catch.
- *There is no overall boolean in the schema.* Two independently generated fields can
  contradict each other, and the boolean is the one that gets read. The summary is computed
  in code from the claim list instead.
- *Field order is load-bearing.* Each claim is written before it is marked. Ask for the
  verdict first and the model commits before it has written down what it is judging, then
  rationalises whatever it picked.

**It fails closed.** A checker that errors reports unsupported with the reason attached,
and a run with no claim checker configured says so even when the figures passed. A partial
check that reads as a full one is worse than no check.

**Modes.** `off`, `shadow`, `enforce`. Shadow runs the check, records the finding, and
still returns the answer, so the thing can be measured on live traffic before it is allowed
to withhold anything. That staging is the point: the check costs an extra call per answered
question, and whether that is worth paying is a question about catch rate, not a question
about architecture.

**Validating it.** `python evaluate.py --judge` runs six known-bad examples, each pairing a
correct answer with a mutation of it: a changed figure, a dropped condition, a widened
scope, an invented detail, a flipped rule, and a plausible addition. Measuring a
self-report on answers that are already correct only shows it is agreeable, so the pair
gives the two numbers that matter, how often it flags something fine and how often it
misses something that is not.

## Not built, and why

- **Stuffing the whole knowledge base into the prompt.** It fits with room to spare and
  would beat retrieval on accuracy today. Retrieval is here because it survives the
  knowledge base growing, but at 2.1 KB it is not what is carrying correctness.

- **Multi-turn follow-ups, streaming, auth, rate limiting, persistence, a reindex
  endpoint.**

## Evaluation

`python evaluate.py` runs the labelled questions through the real pipeline and measures two
things separately, because they trade off: **coverage** (does it answer what it should) and
**restraint** (does it refuse what it should).

The question set matters more than its size. In-KB questions are worded the way a customer
asks, never the way the article is written, so retrieval has to do semantic work. Out-of-KB
questions sit next to a real article: "is there a free trial" beside `billing.md` is what
breaks a refusal guardrail, while "who won the 1998 world cup" dies at the gate.

The set is split. `tune` (20 covered, 10 uncovered) is what `top_k` and `min_score` were
chosen against. `holdout` (8 covered, 5 uncovered) was written first and left untouched
while tuning, and `--sweep` refuses to run on it.

> **Numbers pending.** An earlier revision of this pipeline was measured against a different
> knowledge base. Those results do not transfer to the corpus in `data/`, so they have been
> removed rather than reprinted as if they still held. Re-run `evaluate.py` against this
> corpus before quoting any figure.
>
> What to run, in order: `--sweep` on the tuning split to pick the gate, `--retrieval` for
> recall@k, then the full pipeline on `tune`, and finally `--holdout` once, at the end.

What the eval reports, and why each number is there:

- **recall@k**, embeddings only. The ceiling on everything downstream: an article that is
  never retrieved cannot be cited.
- **coverage**, how many covered questions get answered rather than falsely refused. A false
  refusal becomes a support ticket, so it has a real cost.
- **restraint**, how many uncovered questions get refused. Worth separating by cause: a
  refusal from the retrieval gate is a code guarantee, one from the model declining is a
  judgement a weaker model might not repeat.
- **wrong citations**, expected to be zero on any model. Guardrail 3 uses no model
  judgement, so it holds regardless of what is generating.

`evaluate.py --sweep` picks the gate: the highest threshold that blocks zero covered
questions, which also stops a good share of uncovered ones before any chat call.

Reporting a score on the questions the thresholds were tuned against measures fit to those
questions. That is what the `holdout` split is for, and why it is read once rather than
during tuning. Real support tickets would be a better source than invented questions either
way.

## Roadmap

1. **Decide whether to enforce the grounding check.** It ships in shadow. Enforcing it
   means a false alarm becomes a refused answer, so that call wants data from real traffic,
   not six hand-written mutations.

2. **A cheaper first pass than a second model.** An NLI classifier (a fine-tuned DeBERTa,
   or Vectara's HHEM, which is purpose-built for this) takes a premise and a hypothesis and
   returns entailment in one forward pass. Cheaper, faster, deterministic, self-hostable,
   and weaker on multi-step reasoning. The shape that probably wins is both: NLI as the
   cheap pass, escalating only uncertain claims to the model. Same tiering as the figure
   check, one level up.

3. **Handle knowledge base changes without a deploy.** Reindex on change, version the index
   so an answer traces to a revision, and add a reindex endpoint.

4. **Precompute embeddings at publish time.** Start-up currently calls the embeddings API,
   so no replica can boot while that API is down, and every replica repeats the same work.
   Embedding once when an article changes and writing the vectors to object storage turns
   boot into a file read.

5. **Operability.** Log the retrieval trace per request, since a bad answer is not
   diagnosable without knowing what was retrieved and how it scored. Alert on `unciteable`,
   because that path firing at all means the model is claiming answers it cannot attribute.
   Track the shadow grounding findings as a rate over time, since a climb means the answerer
   is drifting even while every citation still resolves.

## Known rough edges

- The gate threshold should be re-swept whenever articles are added.
- There is no timeout budget across the whole request, only per-call timeouts, so worst-case
  retries can run past four minutes while the caller gave up long ago.
