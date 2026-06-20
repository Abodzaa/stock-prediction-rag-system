import React, { useMemo, useRef, useState } from "react";

export default function ExplanationPanel({
  streaming,
  text,
  prediction,
  selectedModel,
  companyProfile,
  sections,
  error,
  onRetry,
}) {
  const [highlightedSource, setHighlightedSource] = useState(null);
  const [tooltipOpen, setTooltipOpen] = useState(false);
  const sourceRefs = useRef(new Map());
  const tooltipId = "sentiment-weighting-tip";

  const evidenceSections = sections || {
    hasContent: false,
    bullishCatalysts: [],
    bearishRisks: [],
    supportingEvidence: [],
    weakEvidence: [],
    limitationBullets: [],
    sentimentMeter: null,
    sourceConfidence: null,
  };

  const hasEvidence = evidenceSections.supportingEvidence.length > 0 || evidenceSections.weakEvidence.length > 0;
  const showEmpty = !prediction && !streaming && !evidenceSections.hasContent && !error;

  const citationHandler = useMemo(
    () => (number) => {
      const node = sourceRefs.current.get(number);
      setHighlightedSource(number);
      node?.scrollIntoView({ behavior: "smooth", block: "center" });
      if (node?.focus) node.focus();
    },
    []
  );

  if (showEmpty) {
    return (
      <section className="panel explanation-panel explanation-empty" aria-label="Explanation panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Evidence review</p>
            <h2>Why this signal?</h2>
          </div>
        </div>
        <p className="lead">
          Run the explanation stream to see which articles directly support the signal, which ones were excluded,
          and whether the news flow agrees with the model's conviction.
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
            News/evidence lean is shown separately from the model's own price forecast.
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

      {evidenceSections.sentimentMeter ? (
        <div className="explanation-summary-grid">
          <article className="subpanel net-lean-card">
            <div className="subpanel-header">
              <div>
                <h3>Net directional score</h3>
                <p className="muted small">This is the news/evidence lean, not the quant model output.</p>
              </div>
            </div>
            <strong className={`net-lean-value tone-${evidenceSections.sentimentMeter.netDirection.tone}`}>
              {evidenceSections.sentimentMeter.netLabel}
            </strong>
            <p className="metric-footnote">
              {evidenceSections.sourceConfidence?.detail || "Directly relevant sources drive this view."}
            </p>
          </article>

          <article className="subpanel sentiment-meter-card">
            <div className="subpanel-header">
              <div>
                <h3>News sentiment meter</h3>
                <p className="muted small">Only directly relevant sources marked Used are counted.</p>
              </div>
              <button
                type="button"
                className="tooltip-trigger"
                aria-label="Sentiment weighting details"
                aria-describedby={tooltipOpen ? tooltipId : undefined}
                aria-expanded={tooltipOpen}
                onClick={() => setTooltipOpen((current) => !current)}
                onBlur={(event) => {
                  if (!event.currentTarget.contains(event.relatedTarget)) {
                    setTooltipOpen(false);
                  }
                }}
              >
                i
              </button>
            </div>

            {tooltipOpen ? (
              <div id={tooltipId} role="tooltip" className="tooltip-panel">
                More recent articles count more toward this score.
              </div>
            ) : null}

            <div className="sentiment-meter" aria-label={`Bullish ${evidenceSections.sentimentMeter.bullishPercent} percent and bearish ${evidenceSections.sentimentMeter.bearishPercent} percent`}>
              <div
                className="sentiment-meter-bullish"
                style={{ width: `${evidenceSections.sentimentMeter.bullishPercent}%` }}
              />
              <div
                className="sentiment-meter-bearish"
                style={{ width: `${evidenceSections.sentimentMeter.bearishPercent}%` }}
              />
            </div>
            <div className="sentiment-meter-labels">
              <span className="tone-bullish">{evidenceSections.sentimentMeter.bullishPercent}% Bullish</span>
              <span className="tone-bearish">{evidenceSections.sentimentMeter.bearishPercent}% Bearish</span>
            </div>
            <p className="metric-footnote">
              {evidenceSections.sentimentMeter.caption}{" "}
              {evidenceSections.sentimentMeter.supportingNumbers.map((number) => (
                <CitationButton key={`meter-${number}`} number={number} onActivate={citationHandler} />
              ))}
              {evidenceSections.sentimentMeter.excludedNumbers.length ? (
                <>
                  {" "}Excluded{" "}
                  {evidenceSections.sentimentMeter.excludedNumbers.map((number) => (
                    <CitationButton key={`excluded-${number}`} number={number} onActivate={citationHandler} />
                  ))}
                </>
              ) : null}
            </p>
          </article>
        </div>
      ) : null}

      <div className="explanation-grid">
        <CatalystCard
          title="Bullish catalysts"
          subtitle="Concise evidence that supports upside"
          tone="bullish"
          icon="▲"
          items={evidenceSections.bullishCatalysts}
          emptyText="No directly relevant bullish catalysts were strong enough to keep."
          onActivateSource={citationHandler}
        />

        <CatalystCard
          title="Bearish risks"
          subtitle="Concise evidence that pressures the thesis"
          tone="bearish"
          icon="▼"
          items={evidenceSections.bearishRisks}
          emptyText="No directly relevant bearish risks were strong enough to keep."
          onActivateSource={citationHandler}
        />

        <article className="subpanel">
          <div className="subpanel-header">
            <div>
              <h3>Supporting evidence</h3>
              <p className="muted small">Articles directly used in the explanation stream.</p>
            </div>
          </div>
          {evidenceSections.supportingEvidence.length ? (
            <div className="source-card-list">
              {evidenceSections.supportingEvidence.map((source) => (
                <SourceCard
                  key={source.doc_id || source.number}
                  source={source}
                  highlighted={highlightedSource === source.number}
                  setNode={(node) => {
                    if (node) sourceRefs.current.set(source.number, node);
                  }}
                />
              ))}
            </div>
          ) : (
            <p className="muted small">
              {streaming && !text ? "Waiting for directly relevant evidence..." : "No strongly used citations yet."}
            </p>
          )}
        </article>

        <article className="subpanel">
          <div className="subpanel-header">
            <div>
              <h3>Weak or unrelated evidence</h3>
              <p className="muted small">These sources were intentionally excluded from the sentiment score.</p>
            </div>
          </div>
          {evidenceSections.weakEvidence.length ? (
            <div className="source-card-list">
              {evidenceSections.weakEvidence.map((source) => (
                <SourceCard
                  key={source.doc_id || source.number}
                  source={source}
                  highlighted={highlightedSource === source.number}
                  setNode={(node) => {
                    if (node) sourceRefs.current.set(source.number, node);
                  }}
                />
              ))}
            </div>
          ) : (
            <p className="muted small">No weak or unrelated sources were detected.</p>
          )}
        </article>

        <article className="subpanel tone-warning">
          <div className="subpanel-header">
            <div>
              <h3>Model limitations</h3>
              <p className="muted small">Reasons not to over-trust the signal.</p>
            </div>
          </div>
          {evidenceSections.limitationBullets.length ? (
            <ul className="insight-list">
              {evidenceSections.limitationBullets.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          ) : (
            <p className="muted small">No additional limitations surfaced yet.</p>
          )}
        </article>
      </div>

      {!hasEvidence && streaming ? (
        <div className="state-banner tone-loading" role="status">
          <div>
            <strong>Reading the article set</strong>
            <p>Used and excluded evidence will appear here as the explanation stream finishes.</p>
          </div>
        </div>
      ) : null}
    </section>
  );
}

function CatalystCard({ title, subtitle, tone, icon, items, emptyText, onActivateSource }) {
  return (
    <article className={`subpanel catalyst-card tone-${tone}`}>
      <div className="subpanel-header">
        <div>
          <h3>
            <span className="catalyst-icon" aria-hidden="true">{icon}</span> {title}
          </h3>
          <p className="muted small">{subtitle}</p>
        </div>
      </div>
      {items?.length ? (
        <ul className="insight-list catalyst-list">
          {items.map((item) => (
            <li key={item.id}>
              <span>{item.text}</span>{" "}
              {item.citations.map((number) => (
                <CitationButton key={`${item.id}-${number}`} number={number} onActivate={onActivateSource} />
              ))}
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted small">{emptyText}</p>
      )}
    </article>
  );
}

function SourceCard({ source, highlighted, setNode }) {
  return (
    <article
      id={`source-${source.number}`}
      ref={setNode}
      tabIndex={-1}
      className={`source-card tone-${source.tone}${highlighted ? " is-highlighted" : ""}`}
      aria-label={`Source ${source.number}`}
    >
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
      {source.summary ? <p className="source-summary">{source.summary}</p> : null}
      {source.tone !== "used" && source.exclusionReason ? (
        <p className="source-exclusion-note">{source.exclusionReason}</p>
      ) : null}
    </article>
  );
}

function CitationButton({ number, onActivate }) {
  return (
    <button
      type="button"
      className="cite-button"
      onClick={() => onActivate(number)}
      onMouseEnter={() => onActivate(number)}
      onFocus={() => onActivate(number)}
      aria-label={`Jump to source ${number}`}
    >
      [{number}]
    </button>
  );
}
