# Equity Prediction + RAG Explanations

An inference interface over a deduplicated **model zoo** (classical sklearn +
deep PyTorch, H1 & H5 horizons) that predicts next-period equity returns, with a
**retrieval-augmented explanation layer** that narrates *why* each prediction
points up or down - grounded in freshly fetched market news with citations.

Models are served **read-only** from `src/artifacts/`. The backend reuses the
original research code (architectures + feature formulas) so inference matches
training. Nothing is retrained.

## Architecture (MVC)

```
React SPA --> REST --> /api/models, /api/predict        (request/response)
          --> SSE ----> /api/explain/stream             (streamed explanation)
              |
        Controllers (thin)
              |
        Services:  Inference • Ingest • Retrieval • Explanation
              |
        Repositories (Adapters): Model(joblib+torch) • Price(yfinance)
                                  News • Embedding(e5) • Qdrant • Groq
```

See `DECISIONS.md` for every resolved design choice and the documented
train-vs-live approximations.

## Quick start (Docker)

```bash
cp .env.example .env          # add GROQ_API_KEY (+ NEWS_API_KEY if using finnhub/newsapi)
docker compose up --build
# Frontend  -> http://localhost:5173
# API docs  -> http://localhost:8000/docs
```

The trained checkpoints in `src/artifacts/` are mounted into the backend
read-only (kept out of the image). e5 / FinBERT / SBERT weights download on first
use into the persisted `hf_cache` volume.

## Local dev (no Docker)

```bash
# 1. (once) build the model registry from the artifacts
cd backend && python build_registry.py --models-dir ../src/artifacts

# 2. backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# 3. a Qdrant (optional; explanation falls back to direct news if absent)
docker run -p 6333:6333 qdrant/qdrant

# 4. frontend
cd ../frontend && npm install && npm run dev   # http://localhost:5173
```

### Frontend ticker data

- The source CSV snapshots live outside the Vite app root in `src/data/metadata/sp500_constituents_snapshot.csv` and `src/data/metadata/nasdaq100_constituents_snapshot.csv`.
- The frontend uses a build-time generator at `frontend/scripts/build-constituents.mjs` to parse those CSVs and write `frontend/src/data/constituents.json`.
- `npm run dev` and `npm run build` both trigger `npm run gen:tickers` first, so the React app always imports fresh in-root JSON instead of reading files from outside `frontend/` at runtime.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | status, model count, config |
| GET | `/api/models/facets` | filter options (families, horizons, groups, symbols) |
| GET | `/api/models` | filtered, paginated model list |
| GET | `/api/models/{id}` | one model incl. feature names |
| POST | `/api/predict` | `{model_id, symbol?, as_of?}` -> prediction + drivers |
| POST | `/api/explain` | prediction + citations + explanation (blocking) |
| GET | `/api/explain/stream` | SSE: `prediction`, `citations`, `token`, `done` |
| POST | `/api/ingest` | fetch + embed + upsert news for a symbol |
| GET | `/api/news` | recent articles for a symbol |

### Example

```bash
curl -s localhost:8000/api/predict \
  -H 'content-type: application/json' \
  -d '{"model_id":"panel_full_h1_nosent__h1__g2__ridge","symbol":"AAPL"}'
```

## Notes
- **Model types**: `classical` (sklearn Pipeline) and `deep` (10 torch
  architectures). `panel`-basis models predict a chosen stock; `index`-basis models
  predict the market index.
- **Sentiment models (G4/G5)** require news; FinBERT/SBERT run locally on CPU.
- Research models only - **not investment advice**.
