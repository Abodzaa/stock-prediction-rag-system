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

const POSITIVE_KEYWORDS = [
  "beat",
  "beats",
  "bullish",
  "upgrade",
  "surge",
  "gain",
  "gains",
  "growth",
  "strong",
  "record",
  "rebound",
  "expansion",
  "demand",
  "partnership",
  "momentum",
  "outperform",
  "profit",
  "optimistic",
  "lift",
  "improved",
];

const NEGATIVE_KEYWORDS = [
  "miss",
  "misses",
  "bearish",
  "downgrade",
  "drop",
  "selloff",
  "decline",
  "weak",
  "warning",
  "investigation",
  "lawsuit",
  "cut",
  "cuts",
  "pressure",
  "slump",
  "risk",
  "concern",
  "slowdown",
  "lowered",
  "loss",
];

const COMPANY_TOKEN_STOPWORDS = new Set([
  "inc",
  "inc.",
  "corp",
  "corp.",
  "corporation",
  "company",
  "co",
  "co.",
  "the",
  "and",
  "class",
  "holdings",
  "group",
  "plc",
  "ltd",
  "limited",
]);

export function formatPercent(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return "--";
  const numeric = Number(value);
  return `${numeric >= 0 ? "+" : ""}${numeric.toFixed(digits)}%`;
}

export function formatScore(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(digits);
}

export function formatCurrency(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return "--";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(Number(value));
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
  if (direction === "up") return { label: "Bullish", tone: "bullish", icon: "▲" };
  if (direction === "down") return { label: "Bearish", tone: "bearish", icon: "▼" };
  return { label: "Neutral", tone: "neutral", icon: "■" };
}

export function buildPredictionSummary(
  prediction,
  model,
  companyProfile,
  marketContext = null,
  evidenceSummary = null,
  now = new Date()
) {
  if (!prediction) return null;

  const confidenceState = getConfidenceState(prediction.confidence ?? 0);
  const freshness = getFreshnessStatus(prediction.as_of, prediction.horizon, now);
  const badge = getSignalBadge(prediction.direction);
  const priceContext = buildPriceContext(prediction, marketContext);
  const reconciliation = buildReconciliationNote(prediction, evidenceSummary);
  const limitations = buildLimitations(
    prediction,
    model,
    freshness,
    confidenceState,
    evidenceSummary?.limitationSentences || []
  );

  return {
    companyName: companyProfile?.security || prediction.symbol,
    sector: companyProfile?.sector || "Sector unavailable",
    badge,
    predictedMove: formatPercent(prediction.predicted_pct),
    confidenceState,
    freshness,
    limitations,
    priceContext,
    directionalAccuracyNarrative: translateDirectionalAccuracy(prediction.metrics?.directional_accuracy),
    reconciliation,
  };
}

export function buildSourceCards(citations = [], explanationText = "", companyProfile = null, symbol = "") {
  const citedNumbers = new Set(
    [...explanationText.matchAll(/\[(\d+)\]/g)].map((match) => Number(match[1]))
  );
  const tokens = buildCompanyTokens(companyProfile, symbol);

  return citations.map((citation, index) => {
    const number = index + 1;
    const score = citation.score ?? null;
    const headline = String(citation.headline || "");
    const summary = String(citation.summary || "");
    const combinedText = `${headline} ${summary}`.trim();
    const sentimentScore = scoreSentimentText(combinedText);
    const sentimentDirection = getSentimentDirection(sentimentScore);
    const cited = citedNumbers.has(number);
    const mentionsTicker = tokens.length
      ? tokens.some((token) => combinedText.toLowerCase().includes(token))
      : true;

    let status = "Weak or unrelated";
    let tone = "weak";
    let exclusionReason = "Not referenced in the final explanation.";

    if (cited) {
      status = "Used";
      tone = "used";
      exclusionReason = "";
    } else if (!mentionsTicker) {
      exclusionReason = `Broad or other-company context — not clearly specific to ${symbol || "this ticker"}.`;
    } else if (score != null && score < 0.45) {
      exclusionReason = "Low retrieval relevance for the final thesis.";
    }

    return {
      ...citation,
      headline,
      summary,
      number,
      status,
      tone,
      cited,
      mentionsTicker,
      exclusionReason,
      sentimentScore,
      sentimentDirection,
      publishedLabel: formatDate(citation.published),
      scoreLabel: score == null ? "n/a" : formatScore(score, 2),
      recencyWeight: getRecencyWeight(citation.published),
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
  const sourceCards = buildSourceCards(citations, normalizedText, companyProfile, prediction?.symbol || companyProfile?.symbol || "");
  const sentences = normalizedText
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);

  const limitationSentences = sentences.filter((sentence) =>
    LIMITATION_KEYWORDS.some((keyword) => sentence.toLowerCase().includes(keyword))
  );

  const supportingEvidence = sourceCards.filter((source) => source.tone === "used");
  const weakEvidence = sourceCards.filter((source) => source.tone !== "used");
  const sentimentMeter = buildSentimentMeter(supportingEvidence, weakEvidence);
  const catalystLists = buildCatalystLists({
    supportingEvidence,
    explanationText: normalizedText,
    prediction,
    companyProfile,
  });
  const sourceConfidence = getSourceConfidence(supportingEvidence, weakEvidence);
  const limitationBullets = buildLimitations(
    prediction,
    model,
    getFreshnessStatus(prediction?.as_of, prediction?.horizon),
    getConfidenceState(prediction?.confidence ?? 0),
    limitationSentences
  );
  const ragAnswer = buildRagAnswer({
    prediction,
    supportingEvidence,
    weakEvidence,
    sentimentMeter,
  });

  return {
    hasContent: Boolean(normalizedText || sourceCards.length),
    supportingEvidence,
    weakEvidence,
    bullishCatalysts: catalystLists.bullishCatalysts,
    bearishRisks: catalystLists.bearishRisks,
    sentimentMeter,
    netDirection: sentimentMeter.netDirection,
    sourceConfidence,
    sourceCards,
    limitationBullets,
    limitationSentences,
    ragAnswer,
    mainDrivers: [...catalystLists.bullishCatalysts, ...catalystLists.bearishRisks].map((item) => item.text),
  };
}

export function buildReconciliationNote(prediction, evidenceSummary) {
  if (!prediction || !evidenceSummary?.supportingEvidence?.length) return null;

  const confidenceState = getConfidenceState(prediction.confidence ?? 0);
  const newsDirection = evidenceSummary.netDirection?.direction || "neutral";
  const newsStrength = evidenceSummary.netDirection?.confidence ?? 0.5;
  const newsLabel = newsDirection === "bullish" ? "bullish" : newsDirection === "bearish" ? "bearish" : "mixed";
  const modelDirection = prediction.direction === "up" ? "bullish" : prediction.direction === "down" ? "bearish" : "neutral";

  const directionMismatch =
    newsDirection !== "neutral" && modelDirection !== "neutral" && newsDirection !== modelDirection;
  const strengthMismatch = confidenceState.tone === "low" && newsStrength >= 0.68;

  if (directionMismatch || strengthMismatch) {
    return {
      tone: "warning",
      aligned: false,
      text:
        directionMismatch
          ? `Note: news sentiment leans ${newsLabel}, but the model's price-based signal remains ${confidenceState.tone === "low" ? "weak" : "different"}. Treat this as a mismatch, not confirmation.`
          : `Note: news sentiment leans ${newsLabel} more decisively than the model's price-based signal. Treat this as supportive context, not confirmation.`,
      reminder: "Research support only — not investment advice.",
    };
  }

  if (newsDirection === modelDirection && newsDirection !== "neutral") {
    return {
      tone: "used",
      aligned: true,
      text: "News sentiment and the model signal are aligned.",
      reminder: confidenceState.tone === "low" ? "Still treat this as research support, not investment advice." : "",
    };
  }

  return {
    tone: "context",
    aligned: false,
    text: "News sentiment is mixed, so it should not be treated as clear confirmation of the model signal.",
    reminder: confidenceState.tone === "low" ? "Research support only — not investment advice." : "",
  };
}

export function buildActionSummary({
  prediction,
  selectedModel,
  companyProfile,
  evidenceSummary,
  marketContext,
  now = new Date(),
}) {
  if (!prediction) {
    return {
      tone: "ready",
      text: `Signal: Awaiting run. Selected ticker: ${companyProfile?.symbol || "None selected"}. Run the signal to compare price, conviction, and news evidence in one view.`,
    };
  }

  const summary = buildPredictionSummary(prediction, selectedModel, companyProfile, marketContext, evidenceSummary, now);
  const moveDescription = describeMoveDirection(prediction.predicted_pct);
  const newsPhrase = describeNewsLean(evidenceSummary, summary.reconciliation);
  const recommendation = describeRecommendation(summary);

  return {
    tone: summary.reconciliation?.aligned === false || summary.confidenceState.tone === "low" ? "warning" : summary.badge.tone,
    text: `Signal: ${summary.confidenceState.label.replace(" conviction", "")}. Predicted direction: ${moveDescription}. Confidence: ${summary.confidenceState.label}. News sentiment: ${newsPhrase}. Recommendation: ${recommendation}.`,
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
          ? "The backend is still starting. We'll keep retrying and hide transient gateway errors."
          : "Waiting for the backend to finish booting.",
    };
  }

  return {
    tone: "loading",
    title: "Connecting to research services",
    message: "Checking model registry, retrieval pipeline, and explanation services.",
  };
}

function buildPriceContext(prediction, marketContext) {
  if (!prediction || !marketContext?.current_price) return null;

  const illustrativeReturn = 1000 * ((prediction.predicted_pct ?? 0) / 100);

  return {
    currentPriceLabel: formatCurrency(marketContext.current_price),
    predictedPriceLabel: formatCurrency(marketContext.predicted_price),
    rangeLabel: `${formatCurrency(marketContext.range_low)} - ${formatCurrency(marketContext.range_high)}`,
    priceAsOfLabel: formatDate(marketContext.price_as_of),
    volatilityLabel: marketContext.volatility_label,
    volatilityDetail: marketContext.volatility_detail,
    volatilityValueLabel:
      marketContext.volatility_annualized_pct != null
        ? `${formatScore(marketContext.volatility_annualized_pct, 1)}% annualized`
        : "Unavailable",
    upcomingEvents: marketContext.upcoming_events || [],
    trackRecord: marketContext.track_record || null,
    priceHistory: marketContext.price_history || [],
    benchmarkHistory: marketContext.benchmark_history || [],
    benchmarkSymbol: marketContext.benchmark_symbol || null,
    illustrativeReturnLabel: `${illustrativeReturn >= 0 ? "Gain" : "Loss"} about ${formatCurrency(Math.abs(illustrativeReturn))}`,
    illustrativeDirection: illustrativeReturn >= 0 ? "bullish" : "bearish",
  };
}

function buildSentimentMeter(supportingEvidence, weakEvidence) {
  let bullishWeight = 0;
  let bearishWeight = 0;

  supportingEvidence.forEach((source) => {
    const weighted = Math.abs(source.sentimentScore || 0) * (source.recencyWeight || 1);
    if (source.sentimentDirection === "bullish") bullishWeight += weighted || 1 * (source.recencyWeight || 1);
    if (source.sentimentDirection === "bearish") bearishWeight += weighted || 1 * (source.recencyWeight || 1);
  });

  const totalWeight = bullishWeight + bearishWeight;
  const bullishPercent = totalWeight ? Math.round((bullishWeight / totalWeight) * 100) : 50;
  const bearishPercent = 100 - bullishPercent;
  const direction =
    bullishPercent >= 56 ? "bullish" : bearishPercent >= 56 ? "bearish" : "neutral";
  const dominantPercent = Math.max(bullishPercent, bearishPercent);
  const confidence = dominantPercent / 100;
  const intensity =
    dominantPercent >= 75 ? "Strongly" : dominantPercent >= 62 ? "Moderately" : dominantPercent >= 56 ? "Slightly" : "Mixed";
  const netLabel =
    direction === "neutral"
      ? "Net Lean: Mixed"
      : `Net Lean: ${intensity} ${direction === "bullish" ? "Bullish" : "Bearish"} (${bullishPercent}/${bearishPercent})`;

  return {
    bullishPercent,
    bearishPercent,
    direction,
    confidence,
    netLabel,
    directlyRelevantCount: supportingEvidence.length,
    excludedCount: weakEvidence.length,
    supportingNumbers: supportingEvidence.map((source) => source.number),
    excludedNumbers: weakEvidence.map((source) => source.number),
    caption: `Based on ${supportingEvidence.length} directly relevant source${supportingEvidence.length === 1 ? "" : "s"}. ${weakEvidence.length} weak or unrelated source${weakEvidence.length === 1 ? "" : "s"} excluded.`,
    tooltip: "More recent articles count more toward this score.",
    netDirection: {
      direction,
      confidence,
      tone: direction === "neutral" ? "context" : direction === "bullish" ? "bullish" : "bearish",
    },
  };
}

function buildCatalystLists({ supportingEvidence, explanationText, prediction, companyProfile }) {
  const supportingNumbers = new Set(supportingEvidence.map((source) => source.number));
  const sentenceBullets = extractCitedBullets(explanationText, supportingNumbers);
  const sourceBullets = supportingEvidence.map((source) => buildSourceBullet(source, prediction, companyProfile));

  const bullishCatalysts = uniqueBullets(
    [
      ...sentenceBullets.filter((item) => item.direction === "bullish"),
      ...sourceBullets.filter((item) => item.direction === "bullish"),
    ],
    4
  );
  const bearishRisks = uniqueBullets(
    [
      ...sentenceBullets.filter((item) => item.direction === "bearish"),
      ...sourceBullets.filter((item) => item.direction === "bearish"),
    ],
    4
  );

  return {
    bullishCatalysts,
    bearishRisks,
  };
}

function buildRagAnswer({ prediction, supportingEvidence, weakEvidence, sentimentMeter }) {
  if (!supportingEvidence.length) {
    return {
      available: false,
      tone: "context",
      sentences: [],
      emptyMessage:
        "No directly relevant news sources were found for this ticker/horizon, so no RAG answer is available.",
    };
  }

  const leadSources = supportingEvidence.slice(0, 2);
  const leadCitations = leadSources.map((source) => source.number);
  const combinedSummary = composeLeadSummary(leadSources);
  const evidenceDirection = sentimentMeter?.netDirection?.direction || "neutral";
  const modelDirection =
    prediction?.direction === "up" ? "bullish" : prediction?.direction === "down" ? "bearish" : "neutral";
  const confidenceTone = getConfidenceState(prediction?.confidence ?? 0).tone;
  const leanLabel =
    evidenceDirection === "bullish"
      ? "bullish"
      : evidenceDirection === "bearish"
        ? "bearish"
        : "mixed or inconclusive";
  const relationship =
    evidenceDirection !== "neutral" && modelDirection !== "neutral" && evidenceDirection === modelDirection
      ? confidenceTone === "low"
        ? "weak support rather than full confirmation"
        : "confirmation"
      : evidenceDirection !== "neutral" && modelDirection !== "neutral" && evidenceDirection !== modelDirection
        ? "contradiction"
        : "weak or inconclusive support";
  const comparison =
    evidenceDirection === "neutral"
      ? "The news flow is mixed, so it does not point clearly in the same direction as the model."
      : evidenceDirection === modelDirection
        ? "The news lean points in the same direction as the model signal."
        : "The news lean points in a different direction from the model signal.";
  const treatment =
    relationship === "confirmation"
      ? "Treat the news flow as confirmation of the signal, while still checking event risk."
      : relationship === "contradiction"
        ? "Treat this as a contradiction, not confirmation."
        : "Treat this as weak or inconclusive support rather than confirmation.";

  return {
    available: true,
    tone:
      relationship === "confirmation"
        ? "used"
        : relationship === "contradiction"
          ? "warning"
          : "context",
    sentences: [
      {
        id: "rag-1",
        text: combinedSummary,
        citations: leadCitations,
      },
      {
        id: "rag-2",
        text: `Overall, the directly relevant news evidence leans ${leanLabel}.`,
        citations: sentimentMeter?.supportingNumbers || leadCitations,
      },
      {
        id: "rag-3",
        text: `${comparison} ${treatment}`,
        citations: sentimentMeter?.supportingNumbers || leadCitations,
      },
      ...(weakEvidence.length
        ? [
            {
              id: "rag-4",
              text: `${weakEvidence.length} weak or unrelated source${weakEvidence.length === 1 ? "" : "s"} were excluded from this answer.`,
              citations: [],
            },
          ]
        : []),
    ],
  };
}

function extractCitedBullets(explanationText, supportingNumbers) {
  if (!explanationText) return [];

  return explanationText
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+/)
    .map((sentence, index) => {
      const citations = [...sentence.matchAll(/\[(\d+)\]/g)]
        .map((match) => Number(match[1]))
        .filter((number) => supportingNumbers.has(number));
      if (!citations.length) return null;

      const direction = getSentimentDirection(scoreSentimentText(sentence));
      if (direction === "neutral") return null;

      return {
        id: `sentence-${index}-${citations.join("-")}`,
        text: tightenBullet(sentence.replace(/\[\d+\]/g, "").trim()),
        citations,
        direction,
      };
    })
    .filter(Boolean);
}

function buildSourceBullet(source, prediction, companyProfile) {
  const snippet = chooseSourceSnippet(source).replace(/[.!?]+$/, "");
  const symbol = prediction?.symbol || companyProfile?.symbol || "the ticker";
  const horizonLabel = (HORIZON_LABEL[prediction?.horizon] || "selected horizon").toLowerCase();
  const direction =
    source.sentimentDirection === "neutral"
      ? prediction?.direction === "down"
        ? "bearish"
        : "bullish"
      : source.sentimentDirection;
  const tail =
    direction === "bullish"
      ? `this supports ${symbol} over the ${horizonLabel} window`
      : `this adds downside risk to ${symbol} over the ${horizonLabel} window`;

  return {
    id: `source-${source.number}`,
    text: tightenBullet(`${snippet}; ${tail}.`),
    citations: [source.number],
    direction,
  };
}

function uniqueBullets(items, limit) {
  const seen = new Set();
  const out = [];

  items.forEach((item) => {
    if (!item || !item.text) return;
    const key = `${item.direction}:${item.text.toLowerCase()}`;
    if (seen.has(key)) return;
    seen.add(key);
    out.push(item);
  });

  return out.slice(0, limit);
}

function getSourceConfidence(supportingEvidence, weakEvidence) {
  if (supportingEvidence.length >= 3 && weakEvidence.length <= 1) {
    return {
      tone: "high",
      label: "Direct evidence is strong",
      detail: "Multiple directly relevant sources were used in the final explanation.",
    };
  }
  if (supportingEvidence.length >= 1) {
    return {
      tone: "medium",
      label: "Direct evidence is moderate",
      detail: "Some directly relevant sources support the thesis, but the evidence base is not one-sided.",
    };
  }
  return {
    tone: "low",
    label: "Direct evidence is limited",
    detail: "The explanation did not anchor itself in clearly relevant retrieved articles.",
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

function translateDirectionalAccuracy(value) {
  if (value == null || Number.isNaN(Number(value))) {
    return "Historical directional hit-rate is unavailable for this setup.";
  }

  const percent = Math.round(Number(value) * 100);
  let ending = "roughly in line with a coin flip.";
  if (percent >= 65) ending = "meaningfully better than a coin flip, though still far from certain.";
  else if (percent >= 55) ending = "better than a coin flip, but still easy to over-trust.";
  else if (percent >= 50) ending = "only slightly better than a coin flip.";
  else ending = "worse than a coin flip over the sampled backtest cases.";

  return `In ${percent} out of 100 similar past situations, this model's direction call was correct — ${ending}`;
}

function describeMoveDirection(predictedPct = 0) {
  const numeric = Number(predictedPct) || 0;
  if (numeric >= 1.5) return "Up";
  if (numeric > 0) return "Slightly up";
  if (numeric <= -1.5) return "Down";
  if (numeric < 0) return "Slightly down";
  return "Flat";
}

function describeNewsLean(evidenceSummary, reconciliation) {
  if (!evidenceSummary?.supportingEvidence?.length) {
    return "awaiting directly relevant evidence";
  }
  if (reconciliation?.aligned === false) {
    if (evidenceSummary.netDirection.direction === "bullish") {
      return "bullish but unconfirmed by price data";
    }
    if (evidenceSummary.netDirection.direction === "bearish") {
      return "bearish but unconfirmed by price data";
    }
  }
  if (evidenceSummary.netDirection.direction === "bullish") return "bullish";
  if (evidenceSummary.netDirection.direction === "bearish") return "bearish";
  return "mixed";
}

function describeRecommendation(summary) {
  if (summary.freshness.stale || summary.reconciliation?.aligned === false || summary.confidenceState.tone === "low") {
    return "Treat as inconclusive — not a strong buy signal";
  }
  if (summary.confidenceState.tone === "strong" && summary.reconciliation?.aligned) {
    return "Constructive, but still verify event risk before acting";
  }
  return "Useful context, but not enough for a high-conviction decision";
}

function chooseSourceSnippet(source) {
  const summary = String(source.summary || "").trim();
  if (summary) {
    const firstSentence = summary.split(/(?<=[.!?])\s+/)[0];
    if (firstSentence && firstSentence.length >= 30) return firstSentence;
  }
  return String(source.headline || "(untitled article)");
}

function composeLeadSummary(sources) {
  const snippets = sources
    .map((source) => chooseSourceSnippet(source).replace(/[.!?]+$/, ""))
    .filter(Boolean);

  if (!snippets.length) return "No directly relevant used sources were available.";
  if (snippets.length === 1) return `${tightenBullet(snippets[0])}.`;

  return `${tightenBullet(`${snippets[0]}, while ${snippets[1].charAt(0).toLowerCase()}${snippets[1].slice(1)}`)}.`;
}

function tightenBullet(text) {
  if (!text) return "";
  const clean = text.replace(/\s+/g, " ").trim();
  if (clean.length <= 160) return clean;
  return `${clean.slice(0, 157).trimEnd()}...`;
}

function buildCompanyTokens(companyProfile, symbol) {
  const securityTokens = String(companyProfile?.security || "")
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token && !COMPANY_TOKEN_STOPWORDS.has(token) && token.length > 2);

  return [...new Set([String(symbol || "").toLowerCase(), ...securityTokens].filter(Boolean))];
}

function scoreSentimentText(text = "") {
  const lower = text.toLowerCase();
  let score = 0;

  POSITIVE_KEYWORDS.forEach((keyword) => {
    if (lower.includes(keyword)) score += 1;
  });
  NEGATIVE_KEYWORDS.forEach((keyword) => {
    if (lower.includes(keyword)) score -= 1;
  });

  return score;
}

function getSentimentDirection(score = 0) {
  if (score > 0) return "bullish";
  if (score < 0) return "bearish";
  return "neutral";
}

function getRecencyWeight(published) {
  if (!published) return 0.6;
  const date = new Date(published);
  if (Number.isNaN(date.getTime())) return 0.6;
  const ageMs = Math.max(0, Date.now() - date.getTime());
  const ageDays = ageMs / (1000 * 60 * 60 * 24);
  return Math.max(0.3, 1 / (1 + ageDays / 4));
}
