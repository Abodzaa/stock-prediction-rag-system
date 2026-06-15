# DECISIONS.md

Every resolved decision from `RAG_PIPELINE_TEMPLATE.md`, plus the model-serving
choices specific to this project.

## Domain
Inference interface over an existing **research model zoo** (equity return
prediction) with a **RAG layer that explains why** each prediction points up/down,
grounded in fetched market news. Models are served **read-only**; nothing is
retrained. We reuse the original training code for architectures and feature
formulas so inference matches training exactly.

## Model inventory & dedup
- Source: `src/artifacts/` (571 `.joblib` classical + 2197 `.pt` deep checkpoints).
- Deduplicated registry: **2,179 models** (`backend/app/model_registry.json`),
  both H1 and H5 where they exist.
  - 174 classical, 2,005 deep (incl. 1,940 per-symbol).
- **Dedup rules** (`build_registry.py`):
  - *Classical*: identity `(family, variant, horizon, group, algo)`; keep the
    **last checkpoint** = max `(epochs, patience, lag)`. (Honors the "use the last
    checkpoint" instruction; classical joblibs carry no metrics to rank by.)
  - *Deep*: identity `(family, [symbol], horizon, group, model)`; keep the **best**
    checkpoint = **min RMSE** from the metrics embedded in each `.pt`. This mirrors
    how the original sweep selected its own "best config", which is more meaningful
    than an arbitrary last-checkpoint for a hyperparameter grid.
- The 1,940 per-symbol deep models are exposed via one parameterized listing
  (`/api/models?symbol=...`), not 1,940 routes.

## Why architectures were needed (and how resolved)
`.pt` files store only a `state_dict` (named weight tensors), not a runnable
module. We **reuse the original `nn.Module` definitions** via
`src.models.model_factory.build_torch_model` and `load_state_dict(strict=True)`,
guaranteeing exact reconstruction. All 10 architectures (GRU, LSTM, Informer,
PatchTST, TFT, FEDformer, Autoformer, NBEATS, NHITS, MambaStock) load and run.

## Feature engineering (two bases)
Reuses `compute_symbol_features`, `add_cross_sectional_features`,
`compute_g1_features`, `compute_g2_technical`, `build_sequences`,
`aggregate_daily_sentiment` from the research repo.
- **panel basis** (`panel_full*`, `preprocessing_variants`, per-symbol deep):
  features computed on the chosen stock; **G3 = cross-sectional z-scores + breadth**
  computed from the live NASDAQ-100 universe.
- **index basis** (`sentiment_eval_*`, `deep_preprocessed_sentiment*`): features on
  the index (`^GSPC`), **G3 = market breadth**, **G4/G5 = daily news sentiment**.
- Classical models validate against their embedded `feature_names_in_` (authoritative
  ordering). Deep models use the canonical ordered list per `(family, group)`.

### Documented approximations (live vs. training)
- **Deep standardization**: training z-scored each fold on its train slice; that
  mean/std is not persisted, so inference standardizes feature-wise on the available
  **trailing history**. Surfaced as a per-prediction warning.
- **Breadth / cross-sectional / sentiment** use the live fetched universe & news
  rather than the exact historical training universe.

## Section 7 — Component → Pattern → Transport
| Component | Pattern | Transport | Notes |
|---|---|---|---|
| `/api/predict`, `/api/models` | Strategy + Facade | **REST** | one-shot request/response |
| `/api/explain/stream` | — | **SSE** | streamed LLM tokens + structured events |
| News ingestion | Command | **REST** enqueue | `/api/ingest`; idempotent upserts |
| Retrieval modes | **Strategy** | internal | hybrid/dense/sparse via `RETRIEVAL_MODE` |
| Provider clients (Groq, e5, Qdrant, news) | **Adapter + Factory** | internal | swappable |
| Config | **Singleton** | internal | `app/config.py` |

No WebSocket: there are no multi-subscriber/bidirectional surfaces in this app;
the only push is one-way streamed text → SSE per the template's hard rule.

## Retrieval (Section 6)
- Default **hybrid**: dense e5 (768, cosine, `passage:`/`query:` prefixes) + sparse
  hashed term-frequency, fused with **RRF**. Named vectors in Qdrant.
- Reranker: **skipped** for latency; RRF over top-K then `FINAL_K=5` to the LLM.
- Idempotency: point IDs = `uuid5(news_id)`.

## Orchestration (Section 8)
- **`none`** — the flows (predict; retrieve→prompt→generate) are linear. No
  LangChain/LangGraph overhead is justified.

## Generation (Section 9)
- Groq `llama-3.3-70b-versatile` behind an `LLMClient` Adapter, with an offline
  `EchoClient` fallback when `GROQ_API_KEY` is unset (keeps the endpoint usable).
- Guardrails: "explain only from the provided prediction + news context, cite
  `[n]`, say so if context is insufficient, no investment advice."

## Frontend (Section 10)
- React (Vite). Talks only to Controllers. Surfaces: model picker (faceted over the
  registry), stock input, prediction card (direction/confidence/drivers), streamed
  explanation with news citations.

## Deviations from defaults
- Vector size 768 / cosine / e5 prefixes: **kept**.
- Collection name `market_news` (domain-specific) instead of `documents`.
