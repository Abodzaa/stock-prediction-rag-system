import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import {
  buildAvailableOptions,
  buildExplanationSections,
  buildFilterChips,
  buildPredictionSummary,
  filterModels,
  getHealthPresentation,
  sortModels,
} from "../src/lib/research.js";

const catalog = [
  {
    model_id: "a",
    model_name: "Alpha Ridge",
    horizon: "h1",
    kind: "classical",
    group: "G2",
    family: "ridge",
    basis: "panel",
    needs_symbol: true,
    symbol: "AAPL",
    metrics: { directional_accuracy: 0.62, sharpe: 1.1, rmse: 0.02 },
  },
  {
    model_id: "b",
    model_name: "Beta LSTM",
    horizon: "h1",
    kind: "deep",
    group: "G4",
    family: "lstm",
    basis: "panel",
    needs_symbol: true,
    symbol: "MSFT",
    metrics: { directional_accuracy: 0.69, sharpe: 1.25, rmse: 0.04 },
  },
  {
    model_id: "c",
    model_name: "Gamma Index",
    horizon: "h5",
    kind: "classical",
    group: "G3",
    family: "xgboost",
    basis: "index",
    needs_symbol: false,
    symbol: null,
    metrics: { directional_accuracy: 0.57, sharpe: 0.88, rmse: 0.03 },
  },
];

test("health presentation shows warming state for retryable startup failure", () => {
  const state = getHealthPresentation(null, new Error("502"), 2);
  assert.equal(state.tone, "warming");
  assert.match(state.title, /System warming up/i);
});

test("model filtering and ranking preserve strongest setup first", () => {
  const filtered = filterModels(catalog, {
    horizon: "h1",
    kind: "",
    group: "",
    family: "",
    basis: "",
    symbol: "AAPL",
  });
  assert.equal(filtered.length, 1);

  const ranked = sortModels(filtered);
  assert.equal(ranked[0].model_id, "a");
});

test("available filter options narrow based on current selections", () => {
  const available = buildAvailableOptions(catalog, {
    horizon: "h1",
    kind: "",
    group: "",
    family: "",
    basis: "",
    symbol: "AAPL",
  });

  assert.equal(available.family.has("ridge"), true);
  assert.equal(available.family.has("lstm"), false);
  assert.equal(available.family.has("xgboost"), false);
});

test("active filter chips are customer readable", () => {
  const chips = buildFilterChips({
    horizon: "h1",
    kind: "deep",
    group: "",
    family: "lstm",
    basis: "",
    symbol: "AAPL",
  });

  assert.deepEqual(
    chips.map((chip) => chip.label),
    ["Forecast horizon", "Model type", "Model family", "Ticker"]
  );
});

test("prediction summary flags low confidence and stale snapshots", () => {
  const summary = buildPredictionSummary(
    {
      symbol: "AAPL",
      direction: "up",
      confidence: 0.42,
      horizon: "h1",
      as_of: "2026-06-10",
      predicted_pct: 2.4,
      warnings: ["News coverage is sparse."],
    },
    { basis: "panel" },
    { security: "Apple Inc.", sector: "Information Technology" },
    new Date("2026-06-15")
  );

  assert.equal(summary.confidenceState.tone, "low");
  assert.equal(summary.freshness.stale, true);
  assert.equal(summary.limitations.includes("News coverage is sparse."), true);
});

test("explanation sections separate supporting and weak evidence", () => {
  const sections = buildExplanationSections({
    text: "Apple demand trends improved [1]. However, one market-wide headline was only loosely related [2].",
    citations: [
      { doc_id: "1", headline: "iPhone demand rises", source: "Reuters", url: "https://example.com/1", published: "2026-06-14", score: 0.88 },
      { doc_id: "2", headline: "Macro markets mixed", source: "Bloomberg", url: "https://example.com/2", published: "2026-06-14", score: 0.2 },
    ],
    prediction: { as_of: "2026-06-14", horizon: "h1", confidence: 0.61, warnings: [] },
    model: { basis: "panel" },
    companyProfile: { security: "Apple Inc." },
  });

  assert.equal(sections.supportingEvidence.length, 1);
  assert.equal(sections.weakEvidence.length, 1);
  assert.equal(sections.limitationBullets.length > 0, true);
});

test("model picker uses accessible custom listbox controls instead of native select", async () => {
  const source = await readFile(new URL("../src/components/ModelPicker.jsx", import.meta.url), "utf8");

  assert.equal(source.includes("<select"), false);
  assert.match(source, /aria-haspopup="listbox"/);
  assert.match(source, /role="listbox"/);
  assert.match(source, /role="option"/);
  assert.match(source, /case "ArrowDown"/);
  assert.match(source, /case "Escape"/);
});
