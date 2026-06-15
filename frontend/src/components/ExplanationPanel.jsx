import React from "react";
import { buildExplanationSections } from "../lib/research.js";

export default function ExplanationPanel({
  streaming,
  text,
  citations,
  prediction,
  selectedModel,
  companyProfile,
  error,
  onRetry,
}) {
  const sections = buildExplanationSections({
    text,
    citations,
    prediction,
    model: selectedModel,
    companyProfile,
  });

  if (!prediction && !streaming && !sections.hasContent && !error) {
    return (
      <section className="panel explanation-panel explanation-empty" aria-label="Explanation panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Evidence review</p>
            <h2>Why this signal?</h2>
          </div>
        </div>
        <p className="lead">
          Run the explanation stream to see the narrative behind the signal, which sources were actually used,
          which ones were weak, and where the model should be treated with caution.
        </p>
      </section>
    );
  }

  return (
    <section className="panel explanation-panel" aria-label="RAG explanation">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Explainability</p>
          <h2>
            Why this signal? {streaming ? <span className="live-dot" aria-hidden="true" /> : null}
          </h2>
          <p className="muted">
            Grounded explanation, source confidence, and evidence for the selected ticker signal.
          </p>
        </div>
        {streaming ? <span className="status-chip status-live">Streaming evidence</span> : null}
      </div>

      {error ? (
        <div className="state-banner state-error" role="alert">
          <div>
            <strong>Explanation unavailable</strong>
            <p>{error}</p>
          </div>
          {onRetry ? (
            <button type="button" className="button-secondary" onClick={onRetry}>
              Retry explanation
            </button>
          ) : null}
        </div>
      ) : null}

      <div className="explanation-grid">
        <SectionCard
          title="Main drivers"
          subtitle="Highest-signal explanation points"
          emptyText="The explanation has not produced clear drivers yet."
          items={sections.mainDrivers}
        />

        <article className="subpanel">
          <div className="subpanel-header">
            <div>
              <h3>Source confidence</h3>
              <p className="muted small">{sections.sourceConfidence.detail}</p>
            </div>
            <span className={`status-chip tone-${sections.sourceConfidence.tone}`}>
              {sections.sourceConfidence.label}
            </span>
          </div>
          {streaming && !text ? (
            <div className="explanation-skeleton" aria-hidden="true">
              <span className="skeleton-line wide" />
              <span className="skeleton-line" />
              <span className="skeleton-line short" />
            </div>
          ) : (
            <div className="explanation-copy">{renderWithCitations(text)}</div>
          )}
        </article>

        <article className="subpanel">
          <div className="subpanel-header">
            <div>
              <h3>Supporting evidence</h3>
              <p className="muted small">Articles directly used in the explanation stream.</p>
            </div>
          </div>
          {sections.supportingEvidence.length ? (
            <div className="source-card-list">
              {sections.supportingEvidence.map((source) => (
                <SourceCard key={source.doc_id || source.number} source={source} />
              ))}
            </div>
          ) : (
            <p className="muted small">No strongly used citations yet.</p>
          )}
        </article>

        <article className="subpanel">
          <div className="subpanel-header">
            <div>
              <h3>Weak or unrelated evidence</h3>
              <p className="muted small">Contextual or lower-relevance articles that should not be over-weighted.</p>
            </div>
          </div>
          {sections.weakEvidence.length ? (
            <div className="source-card-list">
              {sections.weakEvidence.map((source) => (
                <SourceCard key={source.doc_id || source.number} source={source} />
              ))}
            </div>
          ) : (
            <p className="muted small">No weak-evidence sources were detected.</p>
          )}
        </article>

        <SectionCard
          title="Model limitations"
          subtitle="Reasons not to over-trust the signal"
          items={sections.limitationBullets}
          tone="warning"
          emptyText="No additional limitations surfaced beyond the standard model disclaimer."
        />
      </div>

      {sections.weakEvidence.length ? (
        <div className="state-banner state-warning" role="status">
          <div>
            <strong>Weak evidence detected</strong>
            <p>
              Some retrieved headlines appear weakly related or were not used directly in the final explanation.
              Treat the narrative with proportionate caution.
            </p>
          </div>
        </div>
      ) : null}

      <div className="section-stack">
        <div className="section-title-row">
          <div>
            <h3>All sources</h3>
            <p className="muted small">Citation numbering is preserved so in-text references remain understandable.</p>
          </div>
        </div>
        {sections.sourceCards.length ? (
          <div className="source-card-list">
            {sections.sourceCards.map((source) => (
              <SourceCard key={source.doc_id || source.number} source={source} />
            ))}
          </div>
        ) : (
          <p className="muted small">No source cards yet.</p>
        )}
      </div>
    </section>
  );
}

function SectionCard({ title, subtitle, items, emptyText, tone = "default" }) {
  return (
    <article className={`subpanel tone-${tone}`}>
      <div className="subpanel-header">
        <div>
          <h3>{title}</h3>
          <p className="muted small">{subtitle}</p>
        </div>
      </div>
      {items?.length ? (
        <ul className="insight-list">
          {items.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      ) : (
        <p className="muted small">{emptyText}</p>
      )}
    </article>
  );
}

function SourceCard({ source }) {
  return (
    <article className={`source-card tone-${source.tone}`} aria-label={`Source ${source.number}`}>
      <div className="source-card-header">
        <span className="source-number">[{source.number}]</span>
        <span className={`status-chip tone-${source.tone}`}>{source.status}</span>
      </div>
      <a
        className="source-link"
        href={source.url || "#"}
        target="_blank"
        rel="noreferrer"
        aria-label={`Open source ${source.number}: ${source.headline || "untitled"}`}
      >
        {source.headline || "(untitled article)"}
      </a>
      <div className="source-meta">
        <span>{source.source || "Unknown publisher"}</span>
        <span>{source.publishedLabel}</span>
        <span>Score {source.scoreLabel}</span>
      </div>
    </article>
  );
}

function renderWithCitations(text) {
  if (!text) return "Streaming explanation will appear here once the backend returns token output.";

  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, index) =>
    /^\[\d+\]$/.test(part) ? (
      <sup key={index} className="cite-marker">
        {part}
      </sup>
    ) : (
      <span key={index}>{part}</span>
    )
  );
}
