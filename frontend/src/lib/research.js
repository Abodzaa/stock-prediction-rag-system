export const FILTER_META = {
  horizon: { label: "Forecast horizon", anyLabel: "Any horizon" },
  kind: { label: "Model type", anyLabel: "Any model type" },
  group: { label: "Universe group", anyLabel: "Any universe group" },
  family: { label: "Model family", anyLabel: "Any model family" },
  basis: { label: "Training basis", anyLabel: "Any training basis" },
  symbol: { label: "Ticker", anyLabel: "Any ticker" },
};

export const HUMAN_GROUP = {
  G1: "G1 - Price only",
  G2: "G2 - Technical",
  G3: "G3 - Breadth",
  G4: "G4 - Sentiment blend",
  G5: "G5 - Sentiment only",
};

export const HORIZON_LABEL = {
  h1: "Next trading day",
  h5: "Next 5 trading days",
};

const LIMITATION_KEYWORDS = [
  "however",
  "but",
  "uncertain",
  "uncertainty",
  "limited",
  "not directly",
  "indirect",
  "weak",
  "mixed",
  "caution",
  "may",
  "might",
  "not enough",
  "noise",
];

export function formatPercent(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return "--";
  const numeric = Number(value);
  return `${numeric >= 0 ? "+" : ""}${numeric.toFixed(digits)}%`;
}

export function formatScore(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(digits);
}

export function formatDate(value) {
  if (!value) return "Unavailable";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(date);
}

export function getCompanyProfile(constituents, symbol) {
  return constituents.find((item) => item.symbol === symbol) || null;
}

export function sortModels(models) {
  return [...models].sort((left, right) => {
    const directionalDiff =
      (right.metrics?.directional_accuracy ?? -Infinity) -
      (left.metrics?.directional_accuracy ?? -Infinity);
    if (directionalDiff !== 0) return directionalDiff;

    const sharpeDiff = (right.metrics?.sharpe ?? -Infinity) - (left.metrics?.sharpe ?? -Infinity);
    if (sharpeDiff !== 0) return sharpeDiff;

    const leftRmse = left.metrics?.rmse ?? Number.POSITIVE_INFINITY;
    const rightRmse = right.metrics?.rmse ?? Number.POSITIVE_INFINITY;
    if (leftRmse !== rightRmse) return leftRmse - rightRmse;

    return left.model_name.localeCompare(right.model_name);
  });
}

export function filterModels(catalog, filters) {
  return catalog.filter((model) => {
    if (filters.horizon && model.horizon !== filters.horizon) return false;
    if (filters.kind && model.kind !== filters.kind) return false;
    if (filters.group && model.group !== filters.group) return false;
    if (filters.family && model.family !== filters.family) return false;
    if (filters.basis && model.basis !== filters.basis) return false;
    if (filters.symbol && model.symbol && model.symbol !== filters.symbol) return false;
    return true;
  });
}

export function buildAvailableOptions(catalog, filters) {
  return Object.keys(FILTER_META).reduce((acc, key) => {
    const comparisonFilters = { ...filters, [key]: "" };
    const matching = filterModels(catalog, comparisonFilters);
    acc[key] = new Set(
      matching
        .map((model) => {
          if (key === "symbol") return model.symbol;
          return model[key];
        })
        .filter(Boolean)
    );
    return acc;
  }, {});
}

export function buildFilterChips(filters) {
  return Object.entries(filters)
    .filter(([, value]) => value)
    .map(([key, value]) => ({
      key,
      label: FILTER_META[key]?.label || key,
      value:
        key === "group"
          ? HUMAN_GROUP[value] || value
          : key === "horizon"
            ? HORIZON_LABEL[value] || value
            : value,
    }));
}

export function getConfidenceState(confidence = 0) {
  if (confidence >= 0.72) {
    return {
      tone: "strong",
      label: "Higher conviction",
      message: "Backtest support and model confidence are comparatively stronger than average.",
    };
  }
  if (confidence >= 0.55) {
    return {
      tone: "moderate",
      label: "Moderate conviction",
      message: "Useful as a signal, but it should be checked against other evidence before acting.",
    };
  }
  return {
    tone: "low",
    label: "Low conviction",
    message: "The model sees a weaker edge here, so this should not be treated as a strong trade signal.",
  };
}

export function getFreshnessStatus(asOf, horizon, now = new Date()) {
  const asOfDate = new Date(asOf);
  if (Number.isNaN(asOfDate.getTime())) {
    return {
      tone: "unknown",
      label: "Freshness unavailable",
      detail: "The backend did not return a valid as-of date.",
      stale: false,
    };
  }

  const msPerDay = 1000 * 60 * 60 * 24;
  const ageDays = Math.max(0, Math.floor((now.getTime() - asOfDate.getTime()) / msPerDay));
  const staleThreshold = horizon === "h5" ? 7 : 3;
  const tone = ageDays > staleThreshold ? "stale" : ageDays > 1 ? "aging" : "fresh";

  return {
    tone,
    label:
      tone === "fresh"
        ? "Fresh market snapshot"
        : tone === "aging"
          ? "Aging market snapshot"
          : "Stale market snapshot",
    detail:
      ageDays === 0
        ? "As-of date is current."
        : `${ageDays} day${ageDays === 1 ? "" : "s"} since the model snapshot.`,
    stale: tone === "stale",
    ageDays,
  };
}

export function getSignalBadge(direction) {
  if (direction === "up") return { label: "Bullish", tone: "bullish", icon: "↑" };
  if (direction === "down") return { label: "Bearish", tone: "bearish", icon: "↓" };
  return { label: "Neutral", tone: "neutral", icon: "→" };
}

export function buildPredictionSummary(prediction, model, companyProfile, now = new Date()) {
  if (!prediction) return null;

  const confidenceState = getConfidenceState(prediction.confidence ?? 0);
  const freshness = getFreshnessStatus(prediction.as_of, prediction.horizon, now);
  const badge = getSignalBadge(prediction.direction);
  const limitations = buildLimitations(prediction, model, freshness, confidenceState);

  return {
    companyName: companyProfile?.security || prediction.symbol,
    sector: companyProfile?.sector || "Sector unavailable",
    badge,
    predictedMove: formatPercent(prediction.predicted_pct),
    confidenceState,
    freshness,
    limitations,
  };
}

export function buildSourceCards(citations = [], explanationText = "") {
  const citedNumbers = new Set(
    [...explanationText.matchAll(/\[(\d+)\]/g)].map((match) => Number(match[1]))
  );

  return citations.map((citation, index) => {
    const number = index + 1;
    const score = citation.score ?? null;
    const cited = citedNumbers.has(number);
    let status = "Context only";
    let tone = "context";

    if (cited && (score == null || score >= 0.6)) {
      status = "Used";
      tone = "used";
    } else if (!cited || (score != null && score < 0.45)) {
      status = "Weak relevance";
      tone = "weak";
    }

    return {
      ...citation,
      number,
      status,
      tone,
      cited,
      publishedLabel: formatDate(citation.published),
      scoreLabel: score == null ? "n/a" : formatScore(score, 2),
    };
  });
}

export function buildExplanationSections({
  text,
  citations,
  prediction,
  model,
  companyProfile,
}) {
  const normalizedText = (text || "").trim();
  const sourceCards = buildSourceCards(citations, normalizedText);
  const sentences = normalizedText
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);

  const limitationSentences = sentences.filter((sentence) =>
    LIMITATION_KEYWORDS.some((keyword) => sentence.toLowerCase().includes(keyword))
  );
  const mainDrivers = sentences.filter((sentence) => !limitationSentences.includes(sentence)).slice(0, 3);

  const supportingEvidence = sourceCards.filter((source) => source.tone === "used");
  const weakEvidence = sourceCards.filter((source) => source.tone !== "used");
  const sourceConfidence = getSourceConfidence(sourceCards);
  const limitationBullets = buildLimitations(
    prediction,
    model,
    getFreshnessStatus(prediction?.as_of, prediction?.horizon),
    getConfidenceState(prediction?.confidence ?? 0),
    limitationSentences
  );

  return {
    hasContent: Boolean(normalizedText || sourceCards.length),
    mainDrivers:
      mainDrivers.length > 0
        ? mainDrivers
        : [
            `The research engine is still assembling a clean narrative for ${companyProfile?.security || prediction?.symbol || "this ticker"}.`,
          ],
    supportingEvidence,
    weakEvidence,
    limitationBullets,
    sourceConfidence,
    sourceCards,
    limitationSentences,
  };
}

export function getHealthPresentation(health, error, attempt = 0) {
  if (health) {
    return {
      tone: "ready",
      title: "Research system ready",
      message: `${health.models} models online, ${health.retrieval_mode} retrieval, news provider ${health.news_provider}.`,
    };
  }

  if (error) {
    return {
      tone: "warming",
      title: "System warming up. Retrying...",
      message:
        attempt > 0
          ? "The backend is still starting. We’ll keep retrying and hide transient gateway errors."
          : "Waiting for the backend to finish booting.",
    };
  }

  return {
    tone: "loading",
    title: "Connecting to research services",
    message: "Checking model registry, retrieval pipeline, and explanation services.",
  };
}

function getSourceConfidence(sourceCards) {
  const usedCount = sourceCards.filter((source) => source.tone === "used").length;
  const weakCount = sourceCards.filter((source) => source.tone === "weak").length;

  if (usedCount >= 3 && weakCount <= 1) {
    return {
      tone: "high",
      label: "Source confidence: higher",
      detail: "Multiple cited articles were directly used in the explanation.",
    };
  }
  if (usedCount >= 1) {
    return {
      tone: "medium",
      label: "Source confidence: moderate",
      detail: "Some retrieved articles clearly supported the explanation, but not all were strong matches.",
    };
  }
  return {
    tone: "low",
    label: "Source confidence: limited",
    detail: "The explanation did not strongly anchor itself in the retrieved article set.",
  };
}

function buildLimitations(prediction, model, freshness, confidenceState, explanationLimitations = []) {
  const limitations = [...explanationLimitations];

  if (confidenceState?.tone === "low") {
    limitations.push("Confidence is low, so the signal should be treated as weak supporting evidence only.");
  }
  if (freshness?.stale) {
    limitations.push("The market snapshot is stale relative to the selected horizon.");
  }
  if (model?.basis === "index") {
    limitations.push("This setup was trained on index-level behavior, so it may be less specific to one company.");
  }
  if (prediction?.warnings?.length) {
    limitations.push(...prediction.warnings);
  }
  if (prediction?.horizon === "h5") {
    limitations.push("Five-day forecasts can absorb more market noise and macro event drift than next-day signals.");
  }

  return [...new Set(limitations)].slice(0, 5);
}
