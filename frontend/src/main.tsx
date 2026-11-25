import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type Release = {
  id: number;
  action: "init" | "promote" | "rollback";
  version: string;
  recorded_at: string;
};

type Trial = {
  version: string;
  ndcg_at_10: number;
  model_bytes: number;
  http: { median_http_p95_ms: number | null };
};

type Score = { ndcg_at_10: number; exact_top1_rate: number | null };

type Dashboard = {
  desired: Release;
  serving_version: string;
  events: Release[];
  release_check: {
    current: Trial;
    candidate: Trial;
    decision: { status: "eligible" | "blocked"; reasons: string[] };
  } | null;
  heldout: {
    model_version: string;
    queries: number;
    results: Record<"bm25" | "pretrained" | "release", Score>;
  } | null;
};

type Ranking = {
  model_version: string;
  batch_pairs: number;
  queue_ms: number;
  worker_ms: number;
  results: { id: string; score: number }[];
};

async function json<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, { ...options, cache: "no-store" });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.error ??
        body?.detail ??
        `Request failed with HTTP ${response.status}`,
    );
  }
  return response.json();
}

const short = (version: string) => version.slice(0, 12);
const ms = (value: number | null) =>
  value === null ? "Unavailable" : `${value.toFixed(2)} ms`;
const date = (value: string) =>
  new Date(value).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
const actionNames = {
  init: "Initialized",
  promote: "Promoted",
  rollback: "Rolled back",
};

function ReleaseCheck({ check }: { check: Dashboard["release_check"] }) {
  return (
    <section className="panel" id="comparison">
      <div className="section-heading">
        <h2>Candidate check</h2>
        {check && (
          <span className={`pill ${check.decision.status}`}>
            {check.decision.status}
          </span>
        )}
      </div>
      {check ? (
        <>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Evaluated bundle</th>
                  <th>Validation nDCG@10</th>
                  <th>HTTP p95</th>
                  <th>Model size</th>
                </tr>
              </thead>
              <tbody>
                {(
                  [
                    ["Current", check.current],
                    ["Candidate", check.candidate],
                  ] as const
                ).map(([label, row]) => (
                  <tr key={label}>
                    <th>
                      {label}
                      <small title={row.version}>{short(row.version)}</small>
                    </th>
                    <td>{row.ndcg_at_10.toFixed(6)}</td>
                    <td>{ms(row.http.median_http_p95_ms)}</td>
                    <td>{(row.model_bytes / 1_048_576).toFixed(2)} MiB</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {check.decision.reasons.length > 0 && (
            <ul className="decision-notes">
              {check.decision.reasons.map((reason, i) => (
                <li key={i}>{reason}</li>
              ))}
            </ul>
          )}
        </>
      ) : (
        <p className="muted">
          Run <code>release_check.py</code> to see a comparison.
        </p>
      )}
    </section>
  );
}

function Evaluation({ heldout }: { heldout: Dashboard["heldout"] }) {
  const labels = {
    bm25: "BM25",
    pretrained: "Pretrained TinyBERT",
    release: "Evaluated release",
  };
  const keys = ["bm25", "pretrained", "release"] as const;
  return (
    <section className="panel" id="evaluation">
      <h2>Held out evaluation</h2>
      {heldout ? (
        <>
          <p className="muted evaluation-detail">
            {heldout.queries.toLocaleString()} queries{" "}
            <span aria-hidden="true">·</span>{" "}
            <code title={heldout.model_version}>
              {short(heldout.model_version)}
            </code>
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Test nDCG@10</th>
                  <th>Exact match first</th>
                </tr>
              </thead>
              <tbody>
                {keys.map((key) => {
                  const row = heldout.results[key];
                  return (
                    <tr
                      key={key}
                      className={key === "release" ? "highlight" : undefined}
                    >
                      <th>{labels[key]}</th>
                      <td>{row.ndcg_at_10.toFixed(6)}</td>
                      <td>
                        {row.exact_top1_rate === null
                          ? "Unavailable"
                          : `${(row.exact_top1_rate * 100).toFixed(2)}%`}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <p className="muted">
          Run <code>evaluate_test.py</code> to see held out results.
        </p>
      )}
    </section>
  );
}

function Playground() {
  const [query, setQuery] = useState("usb c cable");
  const [titles, setTitles] = useState(
    "High speed HDMI cable\nUSB C charging cable\nOffice chair",
  );
  const [ranking, setRanking] = useState<Ranking | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [labels, setLabels] = useState<Record<string, string>>({});

  async function rank(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const candidates = titles
      .split("\n")
      .map((title) => title.trim())
      .filter(Boolean)
      .map((title, index) => ({ id: String(index + 1), title }));
    if (!query.trim() || candidates.length < 1 || candidates.length > 32) {
      setError("Enter a query and 1 to 32 product titles.");
      return;
    }
    setError("");
    setRanking(null);
    setBusy(true);
    try {
      const result = await json<Ranking>("/rank", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query.trim(), candidates }),
      });
      setLabels(
        Object.fromEntries(candidates.map((item) => [item.id, item.title])),
      );
      setRanking(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ranking failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel" id="try-ranking">
      <h2>Try a ranking</h2>
      <form onSubmit={rank} className="ranking-form">
        <label>
          Shopping query
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            required
            maxLength={512}
          />
        </label>
        <label>
          Product titles, one per line
          <textarea
            value={titles}
            onChange={(e) => setTitles(e.target.value)}
            rows={4}
            required
          />
        </label>
        <button disabled={busy} type="submit">
          {busy ? "Ranking…" : "Rank products"}
        </button>
      </form>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {ranking && (
        <div className="ranking-output" aria-live="polite">
          <p className="muted">
            Model{" "}
            <code title={ranking.model_version}>
              {short(ranking.model_version)}
            </code>{" "}
            · batch {ranking.batch_pairs} · queue {ms(ranking.queue_ms)} ·
            worker {ms(ranking.worker_ms)}
          </p>
          <ol>
            {ranking.results.map((result) => (
              <li key={result.id}>
                <span>{labels[result.id]}</span>
                <code>{result.score.toFixed(3)}</code>
              </li>
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}

function App() {
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function refresh() {
    setBusy(true);
    setError("");
    try {
      setDashboard(await json<Dashboard>("/api/release"));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load releases.");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    void refresh();
  }, []);
  const synced = dashboard?.desired.version === dashboard?.serving_version;

  return (
    <>
      <header>
        <div className="frame header-row">
          <a href="#top" className="brand">
            Ranker Releases
          </a>
          <nav aria-label="Dashboard sections">
            <a href="#comparison">Comparison</a>
            <a href="#try-ranking">Try ranking</a>
          </nav>
        </div>
      </header>
      <main className="frame" id="top">
        <div className="page-heading">
          <h1>Release dashboard</h1>
          <button
            className="secondary"
            onClick={() => void refresh()}
            disabled={busy}
          >
            {busy ? "Refreshing…" : "Refresh"}
          </button>
        </div>
        {error && (
          <p role="alert" className="error">
            {error}
          </p>
        )}
        {dashboard && (
          <>
            <div className="summary" aria-label="Release summary">
              <div>
                <span>Serving model</span>
                <code title={dashboard.serving_version}>
                  {short(dashboard.serving_version)}
                </code>
              </div>
              <div>
                <span>Selected model</span>
                <code title={dashboard.desired.version}>
                  {short(dashboard.desired.version)}
                </code>
              </div>
              <div>
                <span>Release #{dashboard.desired.id}</span>
                <strong className={synced ? "synced" : ""}>
                  {synced
                    ? "Serving selected version"
                    : "Worker is catching up"}
                </strong>
              </div>
            </div>
            <div className="grid">
              <ReleaseCheck check={dashboard.release_check} />
              <Evaluation heldout={dashboard.heldout} />
            </div>
          </>
        )}
        <div className="grid bottom-grid">
          <Playground />
          {dashboard && (
            <section className="panel">
              <h2>Release history</h2>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Action</th>
                      <th>Version</th>
                      <th>Recorded</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dashboard.events.map((event) => (
                      <tr key={event.id}>
                        <th>{actionNames[event.action]}</th>
                        <td>
                          <code title={event.version}>
                            {short(event.version)}
                          </code>
                        </td>
                        <td>{date(event.recorded_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </div>
      </main>
    </>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
