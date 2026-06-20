import React from "react";
import {
  buildPredictionSummary,
  formatDate,
  formatScore,
  HORIZON_LABEL,
} from "../lib/research.js";

export default function PredictionCard({
  prediction,
  busy,
  companyProfile,
  selectedModel,
  selectedSymbol,
  marketContext,
  marketContextLoading,
  evidenceSummary,
  onRetry,
  error,
}) {
  const displaySymbol = selectedSymbol || prediction?.symbol;

  if (busy && !prediction) {
    return (
      <section className="panel prediction-card" aria-label="Prediction loading">
        <div className="prediction-skeleton" aria-hidden="true">
          <span className="skeleton-pill" />
          <span className="skeleton-display" />
          <span className="skeleton-caption" />
        </div>
      </section>
    );
  }

  if (!prediction) {
    return (
      <section className="panel prediction-card prediction-empty" aria-label="Decision panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Decision panel</p>
            <h2>Ready to evaluate {companyProfile?.symbol || selectedSymbol || "the selected ticker"}</h2>
          </div>
          <span className="status-chip tone-ready">Ready</span>
        </div>

        <div className="ready-evaluation-grid" aria-label="Selected evaluation context">
          <article className="ready-metric-card">
            <span className="metric-label">Ticker</span>
            <strong>{companyProfile?.symbol || selectedSymbol || "None selected"}</strong>
            <p className="metric-footnote">{companyProfile?.security || "Company profile unavailable"}</p>
          </article>
          <article className="ready-metric-card">
            <span className="metric-label">Horizon</span>
            <strong>{HORIZON_LABEL[selectedModel?.horizon] || "Not selected"}</strong>
            <p className="metric-footnote">Selected forecast window</p>
          </article>
          <article className="ready-metric-card">
            <span className="metric-label">Model</span>
            <strong>{selectedModel?.model_name || "No setup selected"}</strong>
            <p className="metric-footnote">{selectedModel?.family || "Choose a setup"}</p>
          </article>
        </div>

        <div className="state-banner tone-ready" role="status">
          <div>
            <strong>Run the signal to see direction, confidence, price context, backtest quality, and explanation risk.</strong>
            <p>Model signal only. Not financial advice. Use alongside independent due diligence and risk management.</p>
          </div>
        </div>

        {error ? (
          <div className="state-banner state-error" role="alert">
            <div>
              <strong>Prediction unavailable</strong>
              <p>{error}</p>
            </div>
            {onRetry ? (
              <button type="button" className="button-secondary" onClick={onRetry}>
                Retry prediction
              </button>
            ) : null}
          </div>
        ) : null}
      </section>
    );
  }

  const summary = buildPredictionSummary(prediction, selectedModel, companyProfile, marketContext, evidenceSummary);
  const confidencePercent = Math.round((prediction.confidence ?? 0) * 100);
  const priceContext = summary.priceContext;

  return (
    <section className="panel prediction-card" aria-label="Prediction results">
      <div className="prediction-hero">
        <div className="prediction-copy">
          <div className={`signal-badge tone-${summary.badge.tone}`}>
            <span aria-hidden="true">{summary.badge.icon}</span>
            <span>{summary.badge.label}</span>
          </div>
          <p className="eyebrow">Selected signal</p>
          <h2>
            {displaySymbol}
            <span className="prediction-company">{summary.companyName}</span>
          </h2>
          <p className="prediction-subtitle">
            {HORIZON_LABEL[prediction.horizon] || prediction.horizon} / {summary.sector}
          </p>
          <p className="signal-disclaimer">Model signal only. Not financial advice.</p>
        </div>

        <div className="prediction-primary" aria-label="Predicted move">
          <span className="prediction-primary-label">Predicted move</span>
          <strong className={`prediction-value tone-${summary.badge.tone}`}>{summary.predictedMove}</strong>
          <span className="prediction-asof">As of {formatDate(prediction.as_of)}</span>
        </div>
      </div>

      <div className="price-context-grid" aria-label="Price and decision context">
        <article className="subpanel price-context-card">
          <div className="subpanel-header">
            <div>
              <h3>Price context</h3>
              <p className="muted small">Actual price levels, not just percentage move.</p>
            </div>
            {priceContext ? (
              <span className={`status-chip volatility-chip tone-${summary.badge.tone}`}>
                {priceContext.volatilityLabel}
              </span>
            ) : null}
          </div>

          {marketContextLoading && !priceContext ? (
            <div className="explanation-skeleton" aria-hidden="true">
              <span className="skeleton-line wide" />
              <span className="skeleton-line" />
              <span className="skeleton-line short" />
            </div>
          ) : priceContext ? (
            <div className="price-stack">
              <div className="price-stat-grid">
                <PriceStat label="Current" value={priceContext.currentPriceLabel} note={`Last price on ${priceContext.priceAsOfLabel}`} />
                <PriceStat label="Predicted" value={priceContext.predictedPriceLabel} note="Point estimate if the signal holds" />
                <PriceStat label="Likely range" value={priceContext.rangeLabel} note="Illustrative range, not a guarantee" />
              </div>

              <div className="price-support-grid">
                <article className="micro-card">
                  <span className="metric-label">Volatility</span>
                  <strong>{priceContext.volatilityLabel}</strong>
                  <p className="metric-footnote">
                    {priceContext.volatilityValueLabel} / {priceContext.volatilityDetail}
                  </p>
                </article>
                <article className="micro-card">
                  <span className="metric-label">Risk / reward example</span>
                  <strong className={`tone-${priceContext.illustrativeDirection}`}>{priceContext.illustrativeReturnLabel}</strong>
                  <p className="metric-footnote">If you invested $1,000 today and the prediction held exactly.</p>
                </article>
              </div>

              {priceContext.upcomingEvents.length ? (
                <div className="state-banner state-warning" role="alert">
                  <div>
                    <strong>Upcoming event within the forecast window</strong>
                    <p>
                      {priceContext.upcomingEvents.map((event) => `${event.label} (${formatDate(event.date)})`).join(" / ")}.
                      Scheduled events can invalidate the signal quickly.
                    </p>
                  </div>
                </div>
              ) : null}

              {priceContext.trackRecord ? (
                <div className="state-banner tone-context" role="status">
                  <div>
                    <strong>{priceContext.trackRecord.detail}</strong>
                  </div>
                </div>
              ) : null}
            </div>
          ) : (
            <p className="muted small">Price context is unavailable right now.</p>
          )}
        </article>

        <article className="subpanel price-chart-card">
          <div className="subpanel-header">
            <div>
              <h3>Last 90 days</h3>
              <p className="muted small">
                {priceContext?.benchmarkSymbol ? `Ticker vs ${priceContext.benchmarkSymbol}` : "Recent price path"}
              </p>
            </div>
          </div>
          {priceContext?.priceHistory?.length ? (
            <SparklinePanel
              history={priceContext.priceHistory}
              benchmark={priceContext.benchmarkHistory}
              benchmarkLabel={priceContext.benchmarkSymbol}
            />
          ) : (
            <p className="muted small">Historical chart unavailable.</p>
          )}
        </article>
      </div>

      <div className="metric-grid" aria-label="Prediction confidence and backtest metrics">
        <article className="metric-card emphasis">
          <div className="metric-header-line">
            <span className="metric-label">Confidence</span>
            <strong>{confidencePercent}%</strong>
          </div>
          <div
            className="confidence-meter"
            role="img"
            aria-label={`Confidence ${confidencePercent}%`}
          >
            <div
              className={`confidence-fill tone-${summary.confidenceState.tone}`}
              style={{ width: `${confidencePercent}%` }}
            />
          </div>
          <p className="metric-footnote">{summary.confidenceState.label}</p>
          {summary.reconciliation ? (
            <div className={`reconciliation-note tone-${summary.reconciliation.tone}`} role="status">
              <strong>{summary.reconciliation.text}</strong>
              {summary.reconciliation.reminder ? <p>{summary.reconciliation.reminder}</p> : null}
            </div>
          ) : null}
        </article>

        <MetricCard
          label="Directional accuracy"
          value={
            prediction.metrics?.directional_accuracy != null
              ? `${Math.round(prediction.metrics.directional_accuracy * 100)}%`
              : "--"
          }
          note="Backtest hit rate"
        />
        <MetricCard
          label="Sharpe"
          value={formatScore(prediction.metrics?.sharpe)}
          note="Risk-adjusted quality"
        />
        <MetricCard
          label="RMSE"
          value={formatScore(prediction.metrics?.rmse, 4)}
          note="Forecast error"
        />
        <MetricCard
          label="Data freshness"
          value={summary.freshness.label}
          note={summary.freshness.detail}
          tone={summary.freshness.tone}
        />
        <MetricCard
          label="Forecast setup"
          value={selectedModel?.model_name || prediction.model_id}
          note={`${selectedModel?.family || "Unknown family"} / ${selectedModel?.kind || "Unknown type"}`}
        />
      </div>

      <div className="state-banner tone-context" role="status">
        <div>
          <strong>Plain-language backtest translation</strong>
          <p>{summary.directionalAccuracyNarrative}</p>
        </div>
      </div>

      <div className={`state-banner tone-${summary.confidenceState.tone}`} role="status">
        <div>
          <strong>{summary.confidenceState.label}</strong>
          <p>{summary.confidenceState.message}</p>
        </div>
      </div>

      {summary.freshness.stale ? (
        <div className="state-banner state-warning" role="alert">
          <div>
            <strong>Stale snapshot</strong>
            <p>This signal may be outdated for the chosen horizon. Verify current market conditions before acting.</p>
          </div>
        </div>
      ) : null}

      {prediction.drivers?.length ? (
        <div className="section-stack">
          <div className="section-title-row">
            <div>
              <h3>Main quantitative drivers</h3>
              <p className="muted small">The inputs that moved this forecast the most.</p>
            </div>
          </div>
          <div className="driver-grid">
            {prediction.drivers.slice(0, 6).map((driver) => {
              const value = driver.contribution != null ? driver.contribution : driver.importance;
              const positive = driver.contribution != null ? driver.contribution >= 0 : true;
              return (
                <article className="driver-card" key={driver.feature}>
                  <span className="driver-name">{driver.feature}</span>
                  <strong className={positive ? "tone-bullish" : "tone-bearish"}>
                    {driver.contribution != null
                      ? `${value >= 0 ? "+" : ""}${value.toFixed(4)}`
                      : `Importance ${value?.toFixed?.(3) || "--"}`}
                  </strong>
                  <span className="muted small">
                    {driver.value != null ? `Observed ${formatScore(driver.value, 3)}` : "Feature importance only"}
                  </span>
                </article>
              );
            })}
          </div>
        </div>
      ) : null}

      <div className="section-stack">
        <div className="section-title-row">
          <div>
            <h3>Model limitations</h3>
            <p className="muted small">Use these caveats before turning the signal into a buy decision.</p>
          </div>
        </div>
        <ul className="insight-list" aria-label="Model limitations">
          {summary.limitations.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </div>
    </section>
  );
}

function MetricCard({ label, value, note, tone = "default" }) {
  return (
    <article className={`metric-card tone-${tone}`}>
      <span className="metric-label">{label}</span>
      <strong className="metric-value">{value}</strong>
      <span className="metric-footnote">{note}</span>
    </article>
  );
}

function PriceStat({ label, value, note }) {
  return (
    <article className="micro-card">
      <span className="metric-label">{label}</span>
      <strong>{value}</strong>
      <p className="metric-footnote">{note}</p>
    </article>
  );
}

function SparklinePanel({ history, benchmark, benchmarkLabel }) {
  const pricePath = buildPath(history.map((point) => point.close), 320, 140);
  const benchmarkPath = benchmark?.length ? buildPath(normalizeComparisonSeries(history, benchmark), 320, 140) : "";
  const firstPrice = history[0]?.close;
  const lastPrice = history[history.length - 1]?.close;

  return (
    <div className="sparkline-shell">
      <svg
        className="sparkline-chart"
        viewBox="0 0 320 140"
        role="img"
        aria-label={`Price chart for the last ${history.length} sessions${benchmarkLabel ? ` with ${benchmarkLabel} comparison` : ""}`}
      >
        <path className="sparkline-benchmark" d={benchmarkPath} />
        <path className="sparkline-price" d={pricePath} />
      </svg>
      <div className="sparkline-legend">
        <span>Start {firstPrice != null ? formatScore(firstPrice, 2) : "--"}</span>
        <span>End {lastPrice != null ? formatScore(lastPrice, 2) : "--"}</span>
        {benchmarkLabel ? <span>Overlay {benchmarkLabel}</span> : null}
      </div>
    </div>
  );
}

function buildPath(values, width, height) {
  if (!values?.length) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  return values
    .map((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * width;
      const y = height - ((value - min) / range) * height;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function normalizeComparisonSeries(history, benchmark) {
  if (!history?.length || !benchmark?.length) return [];
  const basePrice = history[0].close || 1;
  const benchmarkBase = benchmark[0].close || 1;
  return benchmark.map((point) => (point.close / benchmarkBase) * basePrice);
}
