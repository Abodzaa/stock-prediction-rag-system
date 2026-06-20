import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";

const root = join(process.cwd(), "frontend", "dist");
const port = 4173;

const mime = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
};

const models = [
  {
    model_id: "mismatch-ridge",
    kind: "classical",
    family: "ridge",
    horizon: "h1",
    group: "G2",
    model_name: "Mismatch Ridge",
    basis: "panel",
    needs_symbol: true,
    needs_sentiment: false,
    n_features: 28,
    seq_len: 0,
    symbol: "AAPL",
    metrics: { directional_accuracy: 0.57, sharpe: 0.92, rmse: 0.0215 },
  },
  {
    model_id: "aligned-lstm",
    kind: "deep",
    family: "lstm",
    horizon: "h1",
    group: "G4",
    model_name: "Aligned LSTM",
    basis: "panel",
    needs_symbol: true,
    needs_sentiment: true,
    n_features: 64,
    seq_len: 64,
    symbol: "AAPL",
    metrics: { directional_accuracy: 0.66, sharpe: 1.18, rmse: 0.0191 },
  },
];

const mismatchPrediction = {
  model_id: "mismatch-ridge",
  symbol: "AAPL",
  horizon: "h1",
  as_of: "2026-06-20",
  predicted_log_return: -0.0021,
  predicted_pct: -0.21,
  direction: "down",
  confidence: 0.41,
  drivers: [
    { feature: "market_breadth", contribution: -0.0211, value: -0.33 },
    { feature: "close_to_ema20", contribution: 0.0118, value: 0.19 },
  ],
  feature_snapshot: {},
  metrics: { directional_accuracy: 0.57, sharpe: 0.92, rmse: 0.0215 },
  warnings: ["Short-horizon price signal is weak and easily overturned by event risk."],
};

const alignedPrediction = {
  model_id: "aligned-lstm",
  symbol: "AAPL",
  horizon: "h1",
  as_of: "2026-06-20",
  predicted_log_return: -0.0062,
  predicted_pct: -0.62,
  direction: "down",
  confidence: 0.67,
  drivers: [
    { feature: "news_breadth", importance: 0.312, value: -0.44 },
    { feature: "sector_relative_strength", importance: 0.244, value: -0.27 },
  ],
  feature_snapshot: {},
  metrics: { directional_accuracy: 0.66, sharpe: 1.18, rmse: 0.0191 },
  warnings: [],
};

const mismatchCitations = [
  {
    doc_id: "1",
    headline: "Apple services demand stays resilient ahead of launch cycle",
    summary: "Analysts highlighted stable upgrade demand and strong recurring revenue momentum.",
    source: "Reuters",
    url: "https://example.com/1",
    published: "2026-06-20T12:00:00Z",
    score: 0.91,
  },
  {
    doc_id: "2",
    headline: "Large-cap tech sentiment improves as risk appetite returns",
    summary: "Broad market tone improved and helped support mega-cap technology shares.",
    source: "Bloomberg",
    url: "https://example.com/2",
    published: "2026-06-19T12:00:00Z",
    score: 0.72,
  },
  {
    doc_id: "3",
    headline: "SpaceX supplier rallies after satellite contract win",
    summary: "The article focuses on a different company and is not directly about Apple.",
    source: "CNBC",
    url: "https://example.com/3",
    published: "2026-06-18T12:00:00Z",
    score: 0.21,
  },
];

const alignedCitations = [
  {
    doc_id: "1",
    headline: "Apple shipment checks soften as near-term demand cools",
    summary: "Channel checks pointed to softer unit demand and more cautious near-term ordering.",
    source: "Reuters",
    url: "https://example.com/1a",
    published: "2026-06-20T12:00:00Z",
    score: 0.93,
  },
  {
    doc_id: "2",
    headline: "Suppliers warn on margins and mixed handset inventory",
    summary: "Margin pressure and slower clearing inventory add downside risk around the next session.",
    source: "Bloomberg",
    url: "https://example.com/2a",
    published: "2026-06-19T12:00:00Z",
    score: 0.83,
  },
  {
    doc_id: "3",
    headline: "Federal contracts lift defense names broadly",
    summary: "Broad market context only and not directly relevant to Apple.",
    source: "WSJ",
    url: "https://example.com/3a",
    published: "2026-06-18T12:00:00Z",
    score: 0.19,
  },
];

const mismatchText =
  "Apple services demand stays resilient and supports near-term upside [1]. Improving large-cap risk appetite also helps the stock's near-term tone [2]. However, the price-based model still sees only a weak edge and could be overturned by noise.";
const alignedText =
  "Apple shipment checks softened into the next session and add downside pressure [1]. Supplier margin concerns reinforce that cautious setup [2]. This news flow matches the model's bearish price signal.";

const signalContext = {
  symbol: "AAPL",
  benchmark_symbol: "^GSPC",
  current_price: 195.2,
  price_as_of: "2026-06-20",
  predicted_price: 194.0,
  range_low: 193.1,
  range_high: 198.0,
  volatility_label: "Medium volatility",
  volatility_annualized_pct: 28.4,
  volatility_detail: "Recent price swings are meaningful but not extreme.",
  price_history: Array.from({ length: 90 }).map((_, index) => ({
    date: `2026-03-${String((index % 28) + 1).padStart(2, "0")}`,
    close: Number((182 + index * 0.22 + Math.sin(index / 5) * 3.2).toFixed(2)),
  })),
  benchmark_history: Array.from({ length: 90 }).map((_, index) => ({
    date: `2026-03-${String((index % 28) + 1).padStart(2, "0")}`,
    close: Number((5600 + index * 3.2 + Math.sin(index / 6) * 26).toFixed(2)),
  })),
  upcoming_events: [
    { kind: "earnings", label: "Upcoming earnings", date: "2026-06-21", within_horizon: true },
  ],
  track_record: {
    lookback: 10,
    correct: 6,
    total: 10,
    detail: "Last 10 predictions on AAPL: 6 correct.",
  },
};

function json(res, body) {
  res.writeHead(200, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(body));
}

function sse(res, events) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });
  let delay = 0;
  events.forEach((event) => {
    setTimeout(() => {
      res.write(`data: ${JSON.stringify(event)}\n\n`);
      if (event.type === "done") {
        res.end();
      }
    }, delay);
    delay += 60;
  });
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://127.0.0.1:${port}`);

  if (url.pathname === "/api/health") {
    return json(res, {
      status: "ok",
      models: models.length,
      groq_configured: true,
      news_provider: "mock",
      retrieval_mode: "mock",
    });
  }

  if (url.pathname === "/api/models/facets") {
    return json(res, {
      horizons: ["h1"],
      kinds: ["classical", "deep"],
      groups: ["G2", "G4"],
      families: ["ridge", "lstm"],
      bases: ["panel"],
      symbols: ["AAPL"],
    });
  }

  if (url.pathname === "/api/models") {
    return json(res, { models });
  }

  if (url.pathname === "/api/signal-context") {
    return json(res, signalContext);
  }

  if (url.pathname === "/api/predict" && req.method === "POST") {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
    });
    req.on("end", () => {
      const payload = JSON.parse(body || "{}");
      json(res, payload.model_id === "aligned-lstm" ? alignedPrediction : mismatchPrediction);
    });
    return;
  }

  if (url.pathname === "/api/explain/stream") {
    const modelId = url.searchParams.get("model_id");
    const prediction = modelId === "aligned-lstm" ? alignedPrediction : mismatchPrediction;
    const citations = modelId === "aligned-lstm" ? alignedCitations : mismatchCitations;
    const explanation = modelId === "aligned-lstm" ? alignedText : mismatchText;

    return sse(res, [
      { type: "prediction", data: prediction },
      { type: "citations", data: citations },
      { type: "token", data: explanation },
      { type: "done", data: {} },
    ]);
  }

  try {
    let pathname = decodeURIComponent(url.pathname);
    if (pathname === "/") pathname = "/index.html";
    const filePath = normalize(join(root, pathname));
    const ext = extname(filePath);
    const data = await readFile(filePath).catch(async () => {
      if (!ext) return readFile(join(root, "index.html"));
      throw new Error("Not found");
    });
    res.writeHead(200, { "Content-Type": mime[ext] || "application/octet-stream" });
    res.end(data);
  } catch {
    res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("Not found");
  }
});

server.listen(port, "127.0.0.1", () => {
  process.stdout.write(`mock preview ready http://127.0.0.1:${port}\n`);
});
