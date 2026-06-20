import React, { startTransition, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api, withRetry } from "./api.js";
import ModelPicker, { ModelComparisonAccordion } from "./components/ModelPicker.jsx";
import PredictionCard from "./components/PredictionCard.jsx";
import ExplanationPanel from "./components/ExplanationPanel.jsx";
import TickerSelector from "./components/TickerSelector.jsx";
import constituentsData from "./data/constituents.json";
import {
  buildActionSummary,
  buildAvailableOptions,
  buildExplanationSections,
  buildFilterChips,
  filterModels,
  getCompanyProfile,
  getHealthPresentation,
  HORIZON_LABEL,
  sortModels,
} from "./lib/research.js";

const INITIAL_FILTERS = {
  horizon: "h1",
  kind: "",
  group: "",
  family: "",
  basis: "",
  symbol: "AAPL",
};

export default function App() {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);
  const [healthAttempt, setHealthAttempt] = useState(0);

  const [catalog, setCatalog] = useState([]);
  const [catalogError, setCatalogError] = useState("");
  const [catalogLoading, setCatalogLoading] = useState(false);

  const [filters, setFilters] = useState(INITIAL_FILTERS);
  const [debouncedFilters, setDebouncedFilters] = useState(INITIAL_FILTERS);
  const [modelId, setModelId] = useState("");
  const [symbol, setSymbol] = useState("AAPL");
  const [selectedIndex, setSelectedIndex] = useState("sp500");
  const [mobileControlsOpen, setMobileControlsOpen] = useState(false);
  const [modelComparisonOpen, setModelComparisonOpen] = useState(false);

  const [prediction, setPrediction] = useState(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState("");
  const [marketContext, setMarketContext] = useState(null);
  const [marketContextLoading, setMarketContextLoading] = useState(false);

  const [explanationText, setExplanationText] = useState("");
  const [citations, setCitations] = useState([]);
  const [explanationStreaming, setExplanationStreaming] = useState(false);
  const [explanationError, setExplanationError] = useState("");
  const streamRef = useRef(null);
  const requestRunRef = useRef(0);

  const constituents = constituentsData.constituents;

  useEffect(() => {
    const timeoutId = setTimeout(() => setDebouncedFilters(filters), 220);
    return () => clearTimeout(timeoutId);
  }, [filters]);

  useEffect(() => {
    let cancelled = false;
    let timerId = null;

    async function probeBackend(attempt = 0) {
      try {
        const nextHealth = await api.health();
        if (cancelled) return;
        setHealth(nextHealth);
        setHealthError(null);
      } catch (error) {
        if (cancelled) return;
        setHealthError(error);
        setHealthAttempt(attempt + 1);

        const shouldRetry = error instanceof ApiError ? error.retriable : true;
        if (shouldRetry) {
          const delay = Math.min(15000, 1000 * 2 ** attempt);
          timerId = setTimeout(() => probeBackend(attempt + 1), delay);
        }
      }
    }

    probeBackend();

    return () => {
      cancelled = true;
      if (timerId) clearTimeout(timerId);
    };
  }, []);

  async function loadCatalog() {
    setCatalogLoading(true);
    setCatalogError("");
    try {
      const [facetsResponse, modelsResponse] = await Promise.all([
        withRetry(() => api.facets(), { retries: 2, baseDelay: 900 }),
        withRetry(() => api.listModels({ limit: 1000 }), { retries: 2, baseDelay: 900 }),
      ]);
      if (!facetsResponse || !modelsResponse) return;

      startTransition(() => {
        setCatalog(modelsResponse.models || []);
      });
    } catch (error) {
      setCatalogError(error.message || "Failed to load forecast setups.");
    } finally {
      setCatalogLoading(false);
    }
  }

  useEffect(() => {
    if (!health || catalog.length || catalogLoading) return;
    loadCatalog();
  }, [health, catalog.length, catalogLoading]);

  const filteredModels = useMemo(() => {
    return sortModels(filterModels(catalog, debouncedFilters));
  }, [catalog, debouncedFilters]);

  const availableOptions = useMemo(() => buildAvailableOptions(catalog, debouncedFilters), [catalog, debouncedFilters]);
  const activeChips = useMemo(() => buildFilterChips(filters), [filters]);

  useEffect(() => {
    if (!filteredModels.length) {
      setModelId("");
      return;
    }
    if (!filteredModels.some((model) => model.model_id === modelId)) {
      setModelId(filteredModels[0].model_id);
    }
  }, [filteredModels, modelId]);

  const selectedModel = useMemo(
    () => filteredModels.find((model) => model.model_id === modelId) || null,
    [filteredModels, modelId]
  );

  useEffect(() => {
    clearStaleResults();
  }, [modelId]);

  const activeSymbol = symbol;
  const companyProfile = getCompanyProfile(constituents, activeSymbol);
  const healthPresentation = getHealthPresentation(health, healthError, healthAttempt);
  const explanationSections = useMemo(
    () =>
      buildExplanationSections({
        text: explanationText,
        citations,
        prediction,
        model: selectedModel,
        companyProfile,
      }),
    [citations, companyProfile, explanationText, prediction, selectedModel]
  );
  const actionSummary = useMemo(
    () =>
      buildActionSummary({
        prediction,
        selectedModel,
        companyProfile,
        evidenceSummary: explanationSections,
        marketContext,
      }),
    [companyProfile, explanationSections, marketContext, prediction, selectedModel]
  );

  const optionSets = useMemo(() => {
    return {
      horizon: [...new Set(catalog.map((model) => model.horizon))].sort(),
      kind: [...new Set(catalog.map((model) => model.kind))].sort(),
      group: [...new Set(catalog.map((model) => model.group))].sort(),
      family: [...new Set(catalog.map((model) => model.family))].sort(),
      basis: [...new Set(catalog.map((model) => model.basis))].sort(),
      symbol: [...new Set(catalog.map((model) => model.symbol).filter(Boolean))].sort(),
    };
  }, [catalog]);

  useEffect(() => {
    const match = constituents.find((item) => item.symbol === activeSymbol);
    if (match && !match.indexes.includes(selectedIndex)) {
      setSelectedIndex(match.indexes[0]);
    }
  }, [activeSymbol, constituents, selectedIndex]);

  function handleFilterChange(key, value) {
    setFilters((current) => ({ ...current, [key]: value }));
    if (key === "symbol" && value) {
      setSymbol(value);
      clearStaleResults();
    }
  }

  function handleTickerSelect(nextSymbol) {
    setSymbol(nextSymbol);
    setFilters((current) => ({ ...current, symbol: nextSymbol }));
    clearStaleResults();
  }

  function handleHorizonChange(nextHorizon) {
    setFilters((current) => ({ ...current, horizon: nextHorizon }));
  }

  function resetAdvancedFilters() {
    setFilters((current) => ({
      ...current,
      kind: "",
      group: "",
      family: "",
      basis: "",
    }));
  }

  function clearStaleResults() {
    streamRef.current?.();
    streamRef.current = null;
    requestRunRef.current += 1;
    setPrediction(null);
    setPredictionLoading(false);
    setPredictionError("");
    setMarketContext(null);
    setMarketContextLoading(false);
    setExplanationText("");
    setCitations([]);
    setExplanationStreaming(false);
    setExplanationError("");
  }

  async function loadSignalContext(nextPrediction, runId) {
    if (!nextPrediction?.symbol || !nextPrediction?.horizon) return;

    setMarketContextLoading(true);

    try {
      const benchmarkSymbol = selectedIndex === "nasdaq100" ? "^NDX" : "^GSPC";
      const nextContext = await api.signalContext({
        symbol: nextPrediction.symbol,
        horizon: nextPrediction.horizon,
        predicted_pct: nextPrediction.predicted_pct,
        benchmark_symbol: benchmarkSymbol,
      });
      if (runId !== requestRunRef.current) return;
      setMarketContext(nextContext);
    } catch {
      if (runId !== requestRunRef.current) return;
      setMarketContext(null);
    } finally {
      if (runId === requestRunRef.current) {
        setMarketContextLoading(false);
      }
    }
  }

  async function runPrediction() {
    if (!selectedModel?.model_id) return;

    const runId = requestRunRef.current + 1;
    requestRunRef.current = runId;
    streamRef.current?.();
    streamRef.current = null;

    setPredictionLoading(true);
    setPredictionError("");
    setExplanationText("");
    setCitations([]);
    setExplanationStreaming(false);
    setExplanationError("");

    try {
      const payload = {
        model_id: selectedModel.model_id,
        symbol: activeSymbol,
      };
      const nextPrediction = await api.predict(payload);
      if (runId !== requestRunRef.current) return;
      setPrediction({ ...nextPrediction, symbol: activeSymbol });
      loadSignalContext({ ...nextPrediction, symbol: activeSymbol }, runId);
    } catch (error) {
      if (runId !== requestRunRef.current) return;
      setPredictionError(error.message || "Prediction failed.");
    } finally {
      if (runId === requestRunRef.current) {
        setPredictionLoading(false);
      }
    }
  }

  function runExplanation() {
    if (!selectedModel?.model_id) return;

    const runId = requestRunRef.current + 1;
    requestRunRef.current = runId;
    streamRef.current?.();

    setPredictionLoading(false);
    setExplanationStreaming(true);
    setExplanationText("");
    setCitations([]);
    setExplanationError("");
    setPredictionError("");

    streamRef.current = api.explainStream(
      {
        model_id: selectedModel.model_id,
        symbol: activeSymbol,
      },
      {
        prediction: (nextPrediction) => {
          if (runId !== requestRunRef.current) return;
          setPrediction({ ...nextPrediction, symbol: activeSymbol });
          loadSignalContext({ ...nextPrediction, symbol: activeSymbol }, runId);
        },
        citations: (nextCitations) => {
          if (runId !== requestRunRef.current) return;
          setCitations(nextCitations);
        },
        token: (token) => {
          if (runId !== requestRunRef.current) return;
          setExplanationText((current) => current + token);
        },
        done: () => {
          if (runId !== requestRunRef.current) return;
          setExplanationStreaming(false);
        },
        error: (message) => {
          if (runId !== requestRunRef.current) return;
          setExplanationError(message);
          setExplanationStreaming(false);
        },
      }
    );
  }

  useEffect(() => {
    return () => streamRef.current?.();
  }, []);

  return (
    <div className="app-shell">
      <header className="hero-shell">
        <div className="hero-copy">
          <p className="eyebrow">Trusted trading research terminal</p>
          <h1>Equity Prediction + RAG Explanations</h1>
          <p className="hero-text">
            Evaluate a company signal from ticker search to buy/no-buy confidence with backtest context,
            data freshness, and source-grounded explanation quality.
          </p>
        </div>

        <div className="hero-side">
          <div className={`status-overview tone-${healthPresentation.tone}`}>
            <span className="status-chip">System status</span>
            <strong>{healthPresentation.title}</strong>
            <p>{healthPresentation.message}</p>
            {health ? (
              <div className="overview-metrics">
                <span>{health.models} models</span>
                <span>{health.retrieval_mode} retrieval</span>
                <span>{health.groq_configured ? "LLM ready" : "LLM offline"}</span>
              </div>
            ) : null}
          </div>
        </div>
      </header>

      <div className="mobile-toolbar">
        <button
          type="button"
          className="button-secondary"
          onClick={() => setMobileControlsOpen((current) => !current)}
          aria-expanded={mobileControlsOpen}
          aria-controls="research-rail"
        >
          {mobileControlsOpen ? "Hide research setup" : "Open research setup"}
        </button>
      </div>

      <div className="workspace-shell">
        <aside
          id="research-rail"
          className={`research-rail${mobileControlsOpen ? " is-open" : ""}`}
          aria-label="Research setup"
        >
          <TickerSelector
            constituents={constituents}
            selectedSymbol={symbol}
            selectedIndex={selectedIndex}
            selectedHorizon={filters.horizon}
            onIndexChange={setSelectedIndex}
            onHorizonChange={handleHorizonChange}
            onSelect={handleTickerSelect}
          />

          {catalogError ? (
            <div className="panel">
              <div className="state-banner state-error" role="alert">
                <div>
                  <strong>Forecast setups failed to load</strong>
                  <p>{catalogError}</p>
                </div>
                <button type="button" className="button-secondary" onClick={loadCatalog}>
                  Retry loading
                </button>
              </div>
            </div>
          ) : null}

          <ModelPicker
            filters={filters}
            onFilterChange={handleFilterChange}
            onResetFilters={resetAdvancedFilters}
            activeChips={activeChips}
            models={filteredModels}
            modelId={modelId}
            onModelSelect={setModelId}
            loading={catalogLoading}
            optionSets={optionSets}
            availableOptions={availableOptions}
            selectedModel={selectedModel}
          />

          <section className="panel action-panel" aria-label="Run research actions">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Decision actions</p>
                <h2>Run the selected signal</h2>
                <p className="muted">
                  Use the prediction first, then stream the grounded explanation when you want source-backed context.
                </p>
              </div>
            </div>
            <div className="actions">
              <button type="button" onClick={runPrediction} disabled={!selectedModel || predictionLoading}>
                {predictionLoading ? "Running prediction..." : "Run prediction"}
              </button>
              <button
                type="button"
                className="button-secondary"
                onClick={runExplanation}
                disabled={!selectedModel || explanationStreaming}
              >
                {explanationStreaming ? "Streaming explanation..." : "Run prediction + explanation"}
              </button>
            </div>
            <p className="muted small">
              Current focus: <strong>{activeSymbol}</strong> / {HORIZON_LABEL[filters.horizon] || filters.horizon}
            </p>
          </section>

          <ModelComparisonAccordion
            models={filteredModels}
            modelId={modelId}
            onModelSelect={setModelId}
            loading={catalogLoading}
            open={modelComparisonOpen}
            onToggle={setModelComparisonOpen}
          />
        </aside>

        <main className="results-stage">
          <section className={`panel action-summary-banner tone-${actionSummary.tone}`} aria-label="Action summary">
            <p className="eyebrow">Action summary</p>
            <p className="lead">{actionSummary.text}</p>
          </section>

          <PredictionCard
            prediction={prediction}
            busy={predictionLoading}
            companyProfile={companyProfile}
            selectedModel={selectedModel}
            selectedSymbol={activeSymbol}
            marketContext={marketContext}
            marketContextLoading={marketContextLoading}
            evidenceSummary={explanationSections}
            onRetry={runPrediction}
            error={predictionError}
          />

          <ExplanationPanel
            streaming={explanationStreaming}
            text={explanationText}
            prediction={prediction}
            selectedModel={selectedModel}
            companyProfile={companyProfile}
            selectedSymbol={activeSymbol}
            sections={explanationSections}
            error={explanationError}
            onRetry={runExplanation}
          />
        </main>
      </div>

      <footer className="page-footer">
        <p>Model signal only. Not financial advice. Use alongside independent due diligence and risk management.</p>
      </footer>
    </div>
  );
}
