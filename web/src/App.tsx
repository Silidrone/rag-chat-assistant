import { useEffect, useMemo, useRef, useState } from "react";

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

type Article = { source: string; text: string };

const STORAGE_KEY = "rag-chat-assistant.articles";

const EXAMPLES = [
  "Is it cheaper if I pay for the whole year?",
  "Can a normal team member invite someone?",
  "Is there a free trial?",
];

// Cosine similarity runs 0 to 1, so the axis is the real range rather than a
// fitted one. A fixed axis keeps the gate marker in the same place between
// questions, which is the comparison worth making.
const AXIS_MAX = 1;

function sameSet(a: Article[], b: Article[]): boolean {
  if (a.length !== b.length) return false;
  return a.every((x, i) => x.source === b[i].source && x.text === b[i].text);
}

function why(result: AskResponse, health: Health | null): string {
  const gate = health ? health.min_score.toFixed(2) : "the gate";
  const top = result.debug.top_score.toFixed(3);
  switch (result.debug.refused_by) {
    case "retrieval_gate":
      return `Nothing retrieved cleared the gate. The closest article scored ${top} against a threshold of ${gate}, so the question was declined without calling the model at all.`;
    case "model":
      return "The articles below cleared the gate and went to the model, which judged that they do not settle this question. Scoring close to an article is not the same as being answered by it.";
    case "unciteable":
      return "The model wrote an answer but cited nothing that resolves to a retrieved article, so the answer was dropped rather than returned without a source.";
    case "ungrounded":
      return "An answer was written and correctly cited, but the grounding check found claims the cited article does not support, so it was withheld rather than shown.";
    default:
      return "";
  }
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [defaults, setDefaults] = useState<Article[]>([]);
  const [articles, setArticles] = useState<Article[]>([]);
  const [editing, setEditing] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<AskResponse | null>(null);
  const [failure, setFailure] = useState<{ message: string; fromServer: boolean } | null>(
    null,
  );
  const [pending, setPending] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const edited = useMemo(
    () => defaults.length > 0 && !sameSet(articles, defaults),
    [articles, defaults],
  );

  useEffect(() => {
    fetch("/health")
      .then((r) => (r.ok ? r.json() : null))
      .then(setHealth)
      .catch(() => setHealth(null));

    fetch("/articles")
      .then((r) => (r.ok ? r.json() : []))
      .then((loaded: { source: string; text: string }[]) => {
        const pristine = loaded.map(({ source, text }) => ({ source, text }));
        setDefaults(pristine);
        let saved: Article[] | null = null;
        try {
          const raw = localStorage.getItem(STORAGE_KEY);
          if (raw) saved = JSON.parse(raw);
        } catch {
          saved = null;
        }
        setArticles(Array.isArray(saved) && saved.length > 0 ? saved : pristine);
      })
      .catch(() => setDefaults([]));
  }, []);

  useEffect(() => {
    if (articles.length === 0) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(articles));
    } catch {
      // A browser refusing storage is not a reason to break the page.
    }
  }, [articles]);

  async function ask(text: string) {
    const trimmed = text.trim();
    if (!trimmed || pending) return;
    setPending(true);
    setFailure(null);
    setResult(null);
    setEditing(null);
    try {
      const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // Only send the knowledge base when it differs from the shipped one,
        // so the common case costs no re-indexing at all.
        body: JSON.stringify(edited ? { question: trimmed, articles } : { question: trimmed }),
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

  function updateArticle(source: string, text: string) {
    setArticles((current) => current.map((a) => (a.source === source ? { ...a, text } : a)));
  }

  function renameArticle(source: string, next: string) {
    const name = next.trim();
    if (!name || articles.some((a) => a.source !== source && a.source === name)) return;
    setArticles((current) => current.map((a) => (a.source === source ? { ...a, source: name } : a)));
    setEditing(name);
  }

  function addArticle() {
    let name = "new-article.md";
    let n = 2;
    while (articles.some((a) => a.source === name)) name = `new-article-${n++}.md`;
    setArticles((current) => [...current, { source: name, text: "# New article\n\n" }]);
    setEditing(name);
  }

  function removeArticle(source: string) {
    setArticles((current) => current.filter((a) => a.source !== source));
    setEditing(null);
  }

  function exportArticles() {
    const blob = new Blob([JSON.stringify({ articles }, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "knowledge-base.json";
    link.click();
    URL.revokeObjectURL(url);
  }

  async function importArticles(files: FileList | null) {
    if (!files || files.length === 0) return;
    const incoming: Article[] = [];
    for (const file of Array.from(files)) {
      const text = await file.text();
      if (file.name.endsWith(".json")) {
        try {
          const parsed = JSON.parse(text);
          const list = Array.isArray(parsed) ? parsed : parsed.articles;
          if (Array.isArray(list)) {
            for (const item of list) {
              if (item?.source && item?.text) {
                incoming.push({ source: String(item.source), text: String(item.text) });
              }
            }
          }
        } catch {
          setFailure({ message: `${file.name} is not valid JSON.`, fromServer: false });
          return;
        }
      } else {
        incoming.push({ source: file.name, text });
      }
    }
    if (incoming.length > 0) {
      setArticles(incoming);
      setEditing(null);
      setResult(null);
    }
    if (fileInput.current) fileInput.current.value = "";
  }

  const open = editing ? articles.find((a) => a.source === editing) : null;
  const gatePercent = health ? (health.min_score / AXIS_MAX) * 100 : null;
  const declined = result !== null && result.debug.refused_by !== null;

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="side-head">
          <p className="side-title">knowledge base</p>
          {edited && <span className="side-flag">edited</span>}
        </div>

        <ul className="tree">
          {articles.map((article) => {
            const cited = result?.sources.includes(article.source) ?? false;
            return (
              <li key={article.source}>
                <button
                  type="button"
                  className={
                    "tree-row" +
                    (editing === article.source ? " open" : "") +
                    (cited ? " cited" : "")
                  }
                  onClick={() => setEditing(editing === article.source ? null : article.source)}
                >
                  <svg className="tree-icon" viewBox="0 0 16 16" aria-hidden="true">
                    <path
                      d="M4 1.5h5L12.5 5v9.5h-8.5z"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.2"
                      strokeLinejoin="round"
                    />
                    <path d="M9 1.5V5h3.5" fill="none" stroke="currentColor" strokeWidth="1.2" />
                  </svg>
                  <span className="tree-name">{article.source}</span>
                  {cited && <span className="tree-dot" aria-label="cited" />}
                </button>
              </li>
            );
          })}
        </ul>

        <div className="side-actions">
          <button type="button" onClick={addArticle}>
            Add article
          </button>
          <button type="button" onClick={() => fileInput.current?.click()}>
            Import
          </button>
          <button type="button" onClick={exportArticles}>
            Export
          </button>
          {edited && (
            <button
              type="button"
              onClick={() => {
                setArticles(defaults);
                setEditing(null);
                setResult(null);
              }}
            >
              Reset
            </button>
          )}
        </div>

        <input
          ref={fileInput}
          type="file"
          accept=".md,.json,.txt"
          multiple
          hidden
          onChange={(event) => void importArticles(event.target.files)}
        />

        <p className="side-note">
          Edits stay in this browser and travel with your question. Nobody else sees them.
        </p>
      </aside>

      <main className="page">
        <h1 className="wordmark">rag-chat-assistant</h1>
        <p className="standfirst">
          A support assistant that answers only from the articles on the left. It names the
          one it used, and declines anything they do not cover. Edit them and the answers
          change with them.
        </p>
        {health && (
          <p className="config">
            {articles.length} articles · top_k {health.top_k} · gate {health.min_score} ·
            grounding {health.judge_mode}
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
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask about anything in the knowledge base"
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
              }}
            >
              {example}
            </button>
          ))}
        </div>

        {open && (
          <section className="band">
            <div className="editor-head">
              <input
                className="editor-name"
                value={open.source}
                aria-label="Article file name"
                onChange={(event) => renameArticle(open.source, event.target.value)}
              />
              <button
                type="button"
                className="editor-remove"
                onClick={() => removeArticle(open.source)}
              >
                Delete
              </button>
            </div>
            <textarea
              className="editor-body"
              value={open.text}
              spellCheck={false}
              aria-label="Article text"
              onChange={(event) => updateArticle(open.source, event.target.value)}
            />
            <p className="note">
              The first heading is what gets embedded alongside the body, so it carries topic
              words the body leaves implicit. Changes apply to your next question.
            </p>
          </section>
        )}

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
                    <button
                      key={source}
                      type="button"
                      className="cite"
                      onClick={() => setEditing(source)}
                    >
                      {source}
                    </button>
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
                        Running in shadow, so the answer above was still returned. The check is
                        recorded, not enforced, until it has earned the extra call.
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
                        <span
                          className={cited ? "row-source cited" : "row-source"}
                          title={hit.source}
                        >
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
    </div>
  );
}
