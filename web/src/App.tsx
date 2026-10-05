import { useEffect, useRef, useState } from "react";

type Retrieved = { id: string; source: string; score: number };

type Grounding = {
  mode: string;
  supported: boolean;
  unsupported_claims: string[];
  figures_not_in_source: string[];
  claims_checked: number;
  error: string | null;
};

type AskResponse = {
  answer: string;
  sources: string[];
  debug: {
    refused_by: "retrieval_gate" | "model" | "unciteable" | "ungrounded" | null;
    top_score: number;
    original_answer: string | null;
    retrieved: Retrieved[];
    grounding: Grounding | null;
  };
};

type Health = {
  chunks_indexed: number;
  documents_loaded: number;
  top_k: number;
  min_score: number;
  judge_mode: string;
};

const EXAMPLES = [
  "Is it cheaper if I pay for the whole year?",
  "Can a normal team member invite someone?",
  "Is there a free trial?",
];

// Cosine similarity runs 0 to 1, so the axis is the real range rather than a
// fitted one. A fixed axis keeps the gate marker in the same place between
// questions, which is the comparison worth making.
const AXIS_MAX = 1;

function why(result: AskResponse, health: Health | null): string {
  const gate = health ? health.min_score.toFixed(2) : "the gate";
  const top = result.debug.top_score.toFixed(3);
  switch (result.debug.refused_by) {
    case "retrieval_gate":
      return `Nothing retrieved cleared the gate. The closest article scored ${top} against a threshold of ${gate}, so the question was declined without calling the model at all.`;
    case "model":
      return `The articles below cleared the gate and went to the model, which judged that they do not settle this question. Scoring close to an article is not the same as being answered by it.`;
    case "unciteable":
      return `The model wrote an answer but cited nothing that resolves to a retrieved article, so the answer was dropped rather than returned without a source.`;
    case "ungrounded":
      return `An answer was written and correctly cited, but the grounding check found claims the cited article does not support, so it was withheld rather than shown.`;
    default:
      return "";
  }
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<AskResponse | null>(null);
  // `fromServer` separates "the service answered and said no" from "nothing
  // answered at all". They need different copy: one is a rate limit the viewer
  // should wait out, the other usually means the API is not running.
  const [failure, setFailure] = useState<{ message: string; fromServer: boolean } | null>(null);
  const [pending, setPending] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch("/health")
      .then((r) => (r.ok ? r.json() : null))
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  async function ask(text: string) {
    const trimmed = text.trim();
    if (!trimmed || pending) return;
    setPending(true);
    setFailure(null);
    setResult(null);
    try {
      const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });
      const body = await response.json().catch(() => null);
      if (!response.ok) {
        setFailure({
          message: body?.error ?? `The service returned ${response.status}.`,
          fromServer: true,
        });
        return;
      }
      setResult(body as AskResponse);
    } catch {
      setFailure({ message: "Nothing answered at that address.", fromServer: false });
    } finally {
      setPending(false);
    }
  }

  const declined = result !== null && result.debug.refused_by !== null;
  const gatePercent = health ? (health.min_score / AXIS_MAX) * 100 : null;

  return (
    <main className="page">
      <h1 className="wordmark">rag-chat-assistant</h1>
      <p className="standfirst">
        A support assistant that answers only from six help-centre articles for Tessera, a
        fictional team-collaboration tool. It names the article behind every answer, and
        declines anything those articles do not cover.
      </p>
      {health && (
        <p className="config">
          {health.documents_loaded} articles · {health.chunks_indexed} chunks · top_k{" "}
          {health.top_k} · gate {health.min_score} · grounding {health.judge_mode}
        </p>
      )}

      <form
        className="ask"
        onSubmit={(event) => {
          event.preventDefault();
          void ask(question);
        }}
      >
        <input
          ref={inputRef}
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask about billing, cancelling, exports, uptime, seats or the API"
          aria-label="Your question"
        />
        <button type="submit" disabled={pending || question.trim() === ""}>
          {pending ? "Asking" : "Ask"}
        </button>
      </form>

      <div className="examples">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            className="example"
            onClick={() => {
              setQuestion(example);
              void ask(example);
              inputRef.current?.focus();
            }}
          >
            {example}
          </button>
        ))}
      </div>

      {failure && (
        <section className="band">
          <p className="band-name">no answer</p>
          <p className="answer failure">{failure.message}</p>
          {!failure.fromServer && (
            <p className="why">
              Nothing is listening. Start the API with <code>python app.py</code> on port
              5000, or point the dev server elsewhere with <code>API_URL</code>.
            </p>
          )}
        </section>
      )}

      {result && (
        <>
          <section className="band">
            <p className="band-name">{declined ? "declined" : "answer"}</p>
            <p className={declined ? "answer declined" : "answer"}>{result.answer}</p>

            {declined ? (
              <p className="why">{why(result, health)}</p>
            ) : (
              <div className="cites">
                <span className="cites-label">Drawn from</span>
                {result.sources.map((source) => (
                  <span key={source} className="cite">
                    {source}
                  </span>
                ))}
              </div>
            )}
          </section>

          {result.debug.grounding && (
            <section className="band">
              <p className="band-name">grounding</p>
              {result.debug.grounding.supported ? (
                <p className="verdict held">
                  Every claim checked against the cited article.{" "}
                  {result.debug.grounding.claims_checked} claim
                  {result.debug.grounding.claims_checked === 1 ? "" : "s"}, none unsupported.
                </p>
              ) : (
                <>
                  <p className="verdict flagged">
                    {result.debug.grounding.error
                      ? "The grounding check could not run, which is recorded as a failure rather than a pass."
                      : "The cited article does not support everything the answer says."}
                  </p>
                  {result.debug.grounding.figures_not_in_source.length > 0 && (
                    <p className="finding">
                      Figures found in no cited excerpt:{" "}
                      {result.debug.grounding.figures_not_in_source.map((figure) => (
                        <code key={figure}>{figure}</code>
                      ))}
                    </p>
                  )}
                  {result.debug.grounding.unsupported_claims.map((claim) => (
                    <p className="finding" key={claim}>
                      {claim}
                    </p>
                  ))}
                  {result.debug.grounding.error && (
                    <p className="finding">{result.debug.grounding.error}</p>
                  )}
                  {result.debug.grounding.mode === "shadow" && !declined && (
                    <p className="note">
                      Running in shadow, so the answer above was still returned. The check
                      is recorded, not enforced, until it has earned the extra call.
                    </p>
                  )}
                </>
              )}
            </section>
          )}

          {result.debug.retrieved.length > 0 && (
            <section className="band">
              <p className="band-name">retrieval</p>
              <div className="chart">
                {gatePercent !== null && (
                  <div className="gate-layer" aria-hidden="true">
                    <div className="gate-col">
                      <div className="gate" style={{ left: `${gatePercent}%` }}>
                        <span className="gate-flag">gate {health?.min_score}</span>
                      </div>
                    </div>
                  </div>
                )}
                {result.debug.retrieved.map((hit) => {
                  const cited = result.sources.includes(hit.source);
                  return (
                    <div className="row" key={hit.id}>
                      <span className={cited ? "row-source cited" : "row-source"} title={hit.source}>
                        {hit.source}
                      </span>
                      <span className="track">
                        <span
                          className={cited ? "fill cited" : "fill"}
                          style={{ width: `${Math.min(100, (hit.score / AXIS_MAX) * 100)}%` }}
                        />
                      </span>
                      <span className="row-score">{hit.score.toFixed(3)}</span>
                    </div>
                  );
                })}
                <div className="axis">
                  <span>0.0</span>
                  <span>cosine similarity</span>
                  <span>1.0</span>
                </div>
              </div>
              <p className="note">
                Every chunk the question was compared against, best first. Anything short of
                the gate is declined in code before the model is involved.{" "}
                {declined
                  ? "Nothing here was cited, so nothing is marked."
                  : "A green bar is a source the model cited and the server then resolved back to a real retrieved chunk."}
              </p>
            </section>
          )}
        </>
      )}
    </main>
  );
}
