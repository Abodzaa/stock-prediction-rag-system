import React, { useMemo, useState } from "react";
import { HORIZON_LABEL } from "../lib/research.js";

export default function TickerSelector({
  constituents,
  selectedSymbol,
  selectedIndex,
  selectedHorizon,
  onIndexChange,
  onHorizonChange,
  onSelect,
}) {
  const [query, setQuery] = useState("");
  const activeSymbol = selectedSymbol;

  const visibleConstituents = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    const byIndex = constituents.filter((item) => item.indexes.includes(selectedIndex));

    if (!normalizedQuery) return byIndex;

    return byIndex.filter((item) => {
      return (
        item.symbol.toLowerCase().includes(normalizedQuery) ||
        item.security.toLowerCase().includes(normalizedQuery)
      );
    });
  }, [constituents, query, selectedIndex]);

  return (
    <section className="panel ticker-selector-panel" aria-label="Ticker and horizon selection">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Research setup</p>
          <h2>Start with the company</h2>
          <p className="muted">Choose a ticker and horizon, then run the selected forecast setup.</p>
        </div>
      </div>

      <div className="section-stack">
        <label className="field">
          <span>Search ticker or company</span>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="AAPL or Apple"
            aria-label="Search ticker or company"
          />
        </label>

        <div className="segmented-block">
          <span className="field-label">Forecast horizon</span>
          <div className="segmented-control" role="tablist" aria-label="Forecast horizon">
            {["h1", "h5"].map((option) => (
              <button
                key={option}
                type="button"
                className={option === selectedHorizon ? "is-active" : ""}
                onClick={() => onHorizonChange(option)}
                aria-selected={option === selectedHorizon}
              >
                {HORIZON_LABEL[option]}
              </button>
            ))}
          </div>
        </div>

        <div className="segmented-block">
          <span className="field-label">Universe</span>
          <div className="segmented-control" role="tablist" aria-label="Ticker universe">
            {[
              { id: "sp500", label: "S&P 500" },
              { id: "nasdaq100", label: "Nasdaq-100" },
            ].map((option) => (
              <button
                key={option.id}
                type="button"
                className={option.id === selectedIndex ? "is-active" : ""}
                onClick={() => onIndexChange(option.id)}
                aria-selected={option.id === selectedIndex}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        <div className="selected-company-chip" aria-label="Selected ticker">
          Active ticker <strong>{activeSymbol || "None selected"}</strong>
        </div>
      </div>

      {visibleConstituents.length ? (
        <div className="ticker-list-modern" role="list" aria-label="Available companies">
          {visibleConstituents.slice(0, 80).map((item) => {
            const isSelected = item.symbol === activeSymbol;
            return (
              <button
                key={item.symbol}
                type="button"
                className={`ticker-card${isSelected ? " is-selected" : ""}`}
                onClick={() => onSelect(item.symbol)}
                aria-pressed={isSelected}
              >
                <div className="ticker-card-topline">
                  <strong>{item.symbol}</strong>
                  {item.hasDedicatedModel ? <span className="status-chip tone-used">Dedicated model</span> : null}
                </div>
                <span className="ticker-card-name">{item.security}</span>
                <div className="ticker-card-meta">
                  <span>{item.sector || "Sector unavailable"}</span>
                  <span>{item.indexLabels.join(" / ")}</span>
                </div>
              </button>
            );
          })}
        </div>
      ) : (
        <div className="empty-state-box">
          <strong>No companies match this search</strong>
          <p>Try another ticker, company name, or index universe.</p>
        </div>
      )}
    </section>
  );
}
