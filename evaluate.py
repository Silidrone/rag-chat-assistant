"""Offline evaluation against a labelled question set.

Measures two things separately, because they trade off:

    coverage:  a question the knowledge base answers gets answered, cited to
               the right article
    restraint: a question it does not answer gets refused, with no sources

Refusing everything scores 100% on restraint, so neither number means anything
alone.

Usage:
    python evaluate.py --retrieval   # embeddings only, no chat calls, cheap
    python evaluate.py --sweep       # pick the gate threshold from data
    python evaluate.py               # the whole pipeline, one chat call each
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from app import _make_generator, load_documents
from llm import LlmError, OpenAIClient
from rag.answer import answer_question
from rag.config import ConfigurationError, load_settings
from rag.index import Hit, build_index, chunk_document

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "eval_questions.json")

# The thresholds the sweep walks. Cosine scores from text-embedding-3-small sit
# in a much narrower band than the theoretical 0..1, hence the fine steps.
SWEEP_THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

PASS = "pass"
FAIL = "FAIL"


def load_questions(
    split: str = "tune", path: str = QUESTIONS_PATH
) -> tuple[list[dict], list[dict]]:
    """Load one labelled split.

    `tune` is what top_k and min_score were chosen against, so a score on it
    reports fit to that set. `holdout` was written first and never looked at
    while tuning, so it is the only split that says anything about
    generalisation.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if split not in data:
        raise SystemExit(f"unknown split {split!r}, expected one of tune, holdout")
    return data[split]["in_kb"], data[split]["out_of_kb"]


def load_grounding_cases(path: str = QUESTIONS_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["grounding_cases"]


def evaluate_grounding(documents, judge) -> bool:
    """Does the grounding check catch a mutated answer, and leave a good one alone?

    Two numbers, because they fail in opposite directions. A check that flags
    everything scores perfectly on catches and is useless. One that flags
    nothing scores perfectly on false alarms and is also useless.
    """
    from rag.index import Hit, build_index, chunk_document
    from rag.judge import run_check

    by_source = {name: text for name, text in documents}
    cases = load_grounding_cases()

    print(f"\nGrounding check, {len(cases)} known-bad examples")
    print("-" * 72)

    false_alarms, missed = 0, 0
    for case in cases:
        source = case["source"]
        chunks = chunk_document(source, by_source[source])
        hits = [Hit(chunk=chunk, score=1.0) for chunk in chunks]

        clean = run_check(case["correct"], [source], hits, mode="shadow", judge=judge)
        dirty = run_check(case["mutated"], [source], hits, mode="shadow", judge=judge)

        if not clean.supported:
            false_alarms += 1
        if dirty.supported:
            missed += 1

        clean_mark = PASS if clean.supported else FAIL
        dirty_mark = PASS if not dirty.supported else FAIL
        print(f"[{clean_mark}] correct kept    {source:20} {case['mutation']}")
        print(f"[{dirty_mark}] mutation caught {source:20} {case['mutation']}")
        if not dirty.supported:
            found = dirty.figures_not_in_source or dirty.unsupported_claims
            print(f"      caught by: {'figures' if dirty.figures_not_in_source else 'claims'} {found}")

    total = len(cases)
    print("-" * 72)
    print(f"false alarms on correct answers: {false_alarms}/{total}")
    print(f"mutations missed:                {missed}/{total}")
    return false_alarms == 0 and missed == 0


def evaluate_retrieval(index, embedder, in_kb, out_of_kb, top_k) -> None:
    """Is the right article retrieved at all, before the model is involved?

    Retrieval is the ceiling on everything downstream: an article that never
    makes it into the prompt cannot be cited no matter how good the model is.
    """
    questions = [q["question"] for q in in_kb + out_of_kb]
    vectors = embedder(questions)  # one batched call for the whole set

    print("\n=== Retrieval (no chat calls) ===\n")
    hits_at_k = 0
    for item, vector in zip(in_kb, vectors):
        results = index.search(vector, top_k)
        sources = [hit.chunk.source for hit in results]
        found = item["source"] in sources
        hits_at_k += found
        rank = sources.index(item["source"]) + 1 if found else None
        print(
            f"  [{PASS if found else FAIL}] rank={rank or '-'} "
            f"top={results[0].score:.3f}  {item['question'][:58]}"
        )
        if not found:
            print(f"          wanted {item['source']}, got {sources}")

    total = len(in_kb)
    print(f"\n  recall@{top_k}: {hits_at_k}/{total} ({hits_at_k / total:.0%})")

    in_scores = [index.search(v, 1)[0].score for v in vectors[:total]]
    out_scores = [index.search(v, 1)[0].score for v in vectors[total:]]
    print(f"  in-KB  top score: min {min(in_scores):.3f}  max {max(in_scores):.3f}")
    print(f"  out-KB top score: min {min(out_scores):.3f}  max {max(out_scores):.3f}")


def sweep_threshold(index, embedder, in_kb, out_of_kb) -> None:
    """Show what each gate threshold would cost and buy.

    The gate is a blunt instrument sitting in front of a sharp one. Its job is
    to discard the obviously unrelated for free, so it should be set at the
    highest value that still lets *every* covered question through. Blocking
    covered questions here is unrecoverable: the model never gets to look.
    """
    in_scores = [index.search(v, 1)[0].score for v in embedder([q["question"] for q in in_kb])]
    out_scores = [
        index.search(v, 1)[0].score for v in embedder([q["question"] for q in out_of_kb])
    ]

    print("\n=== Gate threshold sweep ===\n")
    print("  threshold   covered blocked (must be 0)   uncovered stopped free")
    best = 0.0
    for threshold in SWEEP_THRESHOLDS:
        blocked = sum(1 for s in in_scores if s < threshold)
        stopped = sum(1 for s in out_scores if s < threshold)
        if blocked == 0:
            best = threshold
        marker = "  <-- safe" if blocked == 0 else ""
        print(
            f"  {threshold:>9.2f}   {blocked:>21}   {stopped:>2}/{len(out_scores):<18}{marker}"
        )

    print(
        f"\n  Highest threshold that blocks no covered question: {best:.2f}\n"
        "  Everything the gate does not stop is left to the model's own\n"
        "  refusal, which is the accurate half of the guardrail."
    )


def evaluate_end_to_end(index, embedder, generator, settings, in_kb, out_of_kb) -> bool:
    """The full pipeline. One chat call per question."""
    print("\n=== End to end ===\n")

    correct_answers, wrong_source, false_refusals = 0, 0, 0
    for item in in_kb:
        result = answer_question(
            item["question"],
            index=index,
            embedder=embedder,
            generator=generator,
            settings=settings,
        )
        if not result.sources:
            false_refusals += 1
            status, note = FAIL, f"refused by {result.refused_by}"
        elif item["source"] in result.sources:
            correct_answers += 1
            status, note = PASS, ", ".join(result.sources)
        else:
            wrong_source += 1
            status, note = FAIL, f"cited {result.sources}, wanted {item['source']}"
        print(f"  [{status}] {item['question'][:52]:<54} {note}")

    leaks = 0
    print()
    for item in out_of_kb:
        result = answer_question(
            item["question"],
            index=index,
            embedder=embedder,
            generator=generator,
            settings=settings,
        )
        refused = not result.sources
        leaks += not refused
        note = f"refused by {result.refused_by}" if refused else f"LEAKED {result.sources}"
        print(f"  [{PASS if refused else FAIL}] {item['question'][:52]:<54} {note}")

    covered, uncovered = len(in_kb), len(out_of_kb)
    print(
        f"\n  coverage  {correct_answers}/{covered} ({correct_answers / covered:.0%})"
        f"   false refusals {false_refusals}   wrong citation {wrong_source}"
    )
    print(
        f"  restraint {uncovered - leaks}/{uncovered} "
        f"({(uncovered - leaks) / uncovered:.0%})   leaks {leaks}"
    )
    return false_refusals == 0 and wrong_source == 0 and leaks == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval", action="store_true", help="retrieval only, no chat calls"
    )
    parser.add_argument(
        "--sweep", action="store_true", help="print the gate threshold sweep"
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="validate the grounding check against known-bad examples",
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="run against the held-out split instead of the tuning split",
    )
    args = parser.parse_args()

    if args.holdout and args.sweep:
        parser.error("--sweep tunes the gate, so it must not run on the holdout split")

    try:
        settings = load_settings()
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    client = OpenAIClient(settings)
    split = "holdout" if args.holdout else "tune"
    in_kb, out_of_kb = load_questions(split)

    try:
        documents = load_documents()

        if args.judge:
            from app import _make_judge

            return 0 if evaluate_grounding(documents, _make_judge(client, settings)) else 1

        index = build_index(documents, client.embed)
        print(f"Indexed {len(index)} chunks from {len(documents)} articles.")
        print(f"model={settings.chat_model}  top_k={settings.top_k}  gate={settings.min_score}")
        print(f"split={split}  ({len(in_kb)} covered, {len(out_of_kb)} uncovered)")

        if args.sweep:
            sweep_threshold(index, client.embed, in_kb, out_of_kb)
            return 0

        evaluate_retrieval(index, client.embed, in_kb, out_of_kb, settings.top_k)
        if args.retrieval:
            return 0

        ok = evaluate_end_to_end(
            index, client.embed, _make_generator(client), settings, in_kb, out_of_kb
        )
        return 0 if ok else 1
    except LlmError as error:
        print(f"\nLLM error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
