"""End-to-end sanity check against a running instance.

Starts nothing itself. Run the app first (docker compose up, or python app.py),
then in another shell:

    python smoke_test.py                      # defaults to localhost:5000
    BASE_URL=http://localhost:5001 python smoke_test.py   # docker compose

It checks three things against the response contract in the README:
  1. /health responds
  2. a covered question returns an answer with a source
  3. an uncovered question declines with empty sources
"""

import os
import sys

import requests

BASE = os.environ.get("BASE_URL", "http://localhost:5000").rstrip("/")

COVERED = "Is it cheaper if I pay for the whole year?"
UNCOVERED = "Is there a free trial?"


def ask(question):
    r = requests.post(f"{BASE}/ask", json={"question": question}, timeout=60)
    r.raise_for_status()
    return r.json()


def main():
    ok = True

    try:
        h = requests.get(f"{BASE}/health", timeout=10)
        h.raise_for_status()
        print("[pass] /health responded:", h.json())
    except Exception as e:
        print(f"[FAIL] /health did not respond at {BASE}. Is the app running?", e)
        sys.exit(1)

    try:
        res = ask(COVERED)
        ans, sources = res.get("answer", ""), res.get("sources", [])
        print(f"\nQ: {COVERED}")
        print("   answer :", ans)
        print("   sources:", sources)
        if sources:
            print("[pass] covered question returned at least one source")
        else:
            print("[FAIL] expected a non-empty 'sources' for a covered question")
            ok = False
        if "20" not in str(ans):
            print("[warn] answer did not mention the 20% annual discount, check grounding")
    except Exception as e:
        print("[FAIL] covered question errored:", e)
        ok = False

    try:
        res = ask(UNCOVERED)
        ans, sources = res.get("answer", ""), res.get("sources", [])
        print(f"\nQ: {UNCOVERED}  (not in the knowledge base)")
        print("   answer :", ans)
        print("   sources:", sources)
        if not sources:
            print("[pass] uncovered question declined with empty sources")
        else:
            print("[FAIL] expected empty 'sources', the guardrail is leaking")
            ok = False
    except Exception as e:
        print("[FAIL] uncovered question errored:", e)
        ok = False

    print("\n" + ("All checks passed." if ok else "Some checks failed, see above."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
