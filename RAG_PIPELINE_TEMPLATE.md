# Agentic RAG Pipeline Template

> **Purpose.** This document is an instruction guide for an agentic AI ("the Agent"). The Agent reads this file, then reads the user's requirements/domain, and produces a tailored RAG project. The architecture, defaults, and contracts below are the *fixed spine*; everything marked **AGENT DECISION** is for the Agent to resolve per task.
>
> **Reading contract for the Agent.** Do not copy this file verbatim into the project. Treat it as a decision framework. For every component, (1) apply the fixed defaults, (2) resolve the AGENT DECISION points using the user's domain, (3) emit code + config + a short rationale note. Never silently skip a decision point — log each choice in `DECISIONS.md` at the project root.

---

## 0. Fixed Spine (non-negotiable defaults)

These hold across every instantiation unless the user *explicitly* overrides them.

| Concern | Default | Override env var |
|---|---|---|
| Vector DB | Qdrant | `QDRANT_URL`, `QDRANT_API_KEY` |
| Embeddings | `intfloat/multilingual-e5-base` | `EMBEDDING_MODEL` |
| Generation | Groq · `llama-3.3-70b-versatile` | `GROQ_MODEL`, `GROQ_API_KEY` |
| Default retrieval | Hybrid (dense + sparse) | `RETRIEVAL_MODE` |
| Architecture | MVC | — |
| Frontend | React | — |

**e5 prompt prefixes are mandatory** (the model was trained with them, skipping them silently degrades recall):
- Documents at index time: `"passage: " + text`
- Queries at search time: `"query: " + text`

Dimensionality: `multilingual-e5-base` → **768**. Set Qdrant collection vector size to 768, distance `Cosine`.

---

## 1. How the Agent Tailors This Template

The Agent runs this loop before writing any code:

1. **Parse domain & requirements.** Extract: data types (PDF, HTML, transcripts, tables, code…), languages, latency/throughput targets, multi-tenancy, freshness needs, compliance constraints, expected QPS, and whether the workload is read-heavy or write-heavy.
2. **Resolve decision points** (sections marked **AGENT DECISION**). Pick concrete tech, not categories.
3. **Pick orchestration tier** (Section 6).
4. **Map each backend component to a design pattern + transport** (Section 7). This is the part the user cares most about — *do not default everything to REST*.
5. **Emit** the MVC skeleton (Section 4), wire components, write `DECISIONS.md`.
6. **Self-check** against Section 11.

---

## 2. Reference Architecture (MVC)

```
                         React SPA (View)
                              │
            ┌─────────────────┼──────────────────┐
            │ REST/GraphQL    │ WebSocket/SSE     │  ← transport per component
            ▼                 ▼                   ▼
        Controllers  ───────────────────────────────┐   (Controller layer)
            │  thin: validation, auth, routing only  │
            ▼                                        │
        Services (Model layer — business logic)      │
   ┌────────┼─────────┬───────────────┬──────────────┘
   ▼        ▼         ▼               ▼
 Ingest  Retrieval  Generation   Notification / async
   │        │         │               │
   ▼        ▼         ▼               ▼
 Qdrant   Qdrant    Groq          Pub/Sub or WS hub
 (write)  (read)    (LLM API)
```

**MVC mapping (strict):**
- **Model** = domain entities + Services + repositories (Qdrant client, Groq client, embedding client). All business logic lives here. No HTTP awareness.
- **Controller** = thin transport adapters. Validate input, call a Service, shape the response. One controller per transport surface.
- **View** = React. Never talks to Qdrant/Groq directly; only to Controllers.

Rule: a Controller that contains business logic is a bug. A Service that imports an HTTP request object is a bug.

---

## 3. Directory Skeleton

```
project/
├── DECISIONS.md                 # Agent writes every resolved decision here
├── docker-compose.yml           # qdrant + backend + frontend
├── .env.example
├── backend/
│   ├── app/
│   │   ├── controllers/         # transport adapters (thin)
│   │   ├── models/              # entities + DTOs
│   │   ├── services/            # business logic
│   │   │   ├── ingest_service.py
│   │   │   ├── retrieval_service.py
│   │   │   ├── generation_service.py
│   │   │   └── notification_service.py
│   │   ├── repositories/        # qdrant, groq, embedding clients
│   │   ├── patterns/            # observer, strategy, factory impls
│   │   ├── orchestration/       # langchain/langgraph graphs (if used)
│   │   └── config.py
│   └── tests/
└── frontend/                    # React (View)
    └── src/
```

---

## 4. Configuration (`.env.example`)

```env
# --- Vector DB ---
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=documents

# --- Embeddings ---
EMBEDDING_MODEL=intfloat/multilingual-e5-base
EMBEDDING_DIM=768

# --- Generation ---
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile

# --- Retrieval ---
RETRIEVAL_MODE=hybrid          # hybrid | dense | sparse
HYBRID_ALPHA=0.5               # dense vs sparse weight (fusion)
TOP_K=20                       # candidates before rerank
FINAL_K=5                      # passages sent to LLM

# --- Orchestration (AGENT DECISION) ---
ORCHESTRATION=none             # none | langchain | langgraph
```

---

## 5. Ingestion Pipeline

Fixed stages: **load → clean → chunk → embed → upsert**.

### 5.1 Chunking — **AGENT DECISION**
Pick strategy by data type. Do not default to fixed-size for everything.

| Data type | Strategy | Notes |
|---|---|---|
| Prose / articles | Recursive by separators | 512–1024 tokens, 10–15% overlap |
| Code | Syntax-aware (by function/class) | preserve definitions intact |
| Transcripts | By speaker turn / timestamp window | keep speaker metadata |
| Tables / structured | Row-group or schema-aware | store table context in metadata |
| Q&A / FAQ | One chunk per pair | no splitting |

### 5.2 Embedding
- Prefix every chunk with `"passage: "` before encoding.
- Normalize vectors (cosine).
- Batch encode (default batch 32; tune to GPU/CPU).

### 5.3 Qdrant upsert
- Collection: vector size = `EMBEDDING_DIM` (768), distance `Cosine`.
- For hybrid: enable **named vectors** — a dense vector + a sparse vector (e.g. BM25/SPLADE). Use Qdrant sparse-vector support.
- Payload (metadata): `source`, `chunk_id`, `doc_id`, `lang`, `created_at`, plus any domain filters (tenant_id, category…). **AGENT DECISION:** which payload fields are filterable indexes.
- Idempotency: deterministic point IDs (hash of `doc_id + chunk_index`) so re-ingest updates instead of duplicating.

---

## 6. Retrieval

**Default = hybrid.** Dense (e5) + sparse (keyword), fused.

### 6.1 Fusion — **AGENT DECISION**
- Reciprocal Rank Fusion (RRF) by default; or weighted-sum with `HYBRID_ALPHA`.
- Switch to **dense-only** when corpus is conversational/semantic with little exact-keyword need; **sparse-only** when queries are highly lexical (codes, IDs, legal citations).

### 6.2 Reranking — **AGENT DECISION**
- Add a cross-encoder reranker (e.g. `bge-reranker`) when precision@k matters and latency budget allows. Skip for low-latency / high-QPS paths.
- Flow: retrieve `TOP_K=20` → rerank → keep `FINAL_K=5`.

### 6.3 Pattern: **Strategy**
Retrieval modes (hybrid/dense/sparse) are interchangeable `RetrievalStrategy` implementations selected at runtime from `RETRIEVAL_MODE`. The Service depends on the interface, not the concrete mode.

---

## 7. Component → Pattern → Transport Map

This is the core differentiation the user asked for. **Never blanket-apply REST.** For each component the Agent picks the pattern and transport that fit its interaction shape.

| Component | Interaction shape | Pattern (default) | Transport (default) |
|---|---|---|---|
| Query / retrieval-generation | request→response | Strategy + Facade | **REST** (or GraphQL if nested fetches) |
| Streaming answer tokens | server push, one-way | — | **SSE** (or WS if bidirectional) |
| Notifications / events | server push, many subscribers | **Observer** | **WebSocket** |
| Ingestion jobs | long-running, async | **Command + Queue/Worker** | enqueue via REST, status via WS/SSE |
| Document live-collab / chat | bidirectional realtime | Observer/Mediator | **WebSocket** |
| Embedding/LLM client swap | provider abstraction | **Adapter + Factory** | internal |
| Config/feature flags | cross-cutting | **Singleton** (config) | internal |

**Hard rule encoded for the Agent:**
- Anything that is *fire-and-subscribe* (notifications, presence, progress) → Observer + WebSocket. Not REST polling.
- Anything *one-shot request/response* → REST.
- Anything *streamed text* → SSE unless the client also sends mid-stream → WebSocket.
- Provider clients (Groq, embedding model, Qdrant) → Adapter behind a stable interface so they're swappable.

**AGENT DECISION** per project: confirm each row against the domain and record deviations in `DECISIONS.md` with a one-line reason.

---

## 8. Orchestration — **AGENT DECISION**

Choose the lightest tier that satisfies the requirements.

- **`none`** — single linear flow (retrieve → rerank → prompt → generate). Default for simple Q&A. No framework overhead.
- **`langchain`** — when you need composable chains, prompt templates, tool/retriever abstractions, memory, but the flow is still mostly linear/branching.
- **`langgraph`** — when the flow is a **stateful graph**: cyclic reasoning, multi-step agents, conditional routing, retries, human-in-the-loop, or multi-agent. Use when nodes need shared mutable state and edges are conditional.

Decision heuristic:
```
no branching, no tools, no memory      → none
linear chain + tools/memory            → langchain
loops / conditional routing / agents   → langgraph
```
Record the chosen tier and why in `DECISIONS.md`.

---

## 9. Generation

- Client: Groq, model `GROQ_MODEL`. Wrap in an **Adapter** (`LLMClient` interface) so a different provider can be swapped without touching Services.
- Prompt assembly: system instructions + retrieved `FINAL_K` passages (with source markers) + user query. Keep passages clearly delimited so the model can cite.
- Always include source attribution in the response payload (passage `doc_id`/`source`) for the View to render citations.
- Streaming: stream tokens over SSE/WS per Section 7.
- Guardrails — **AGENT DECISION:** add grounding/"answer only from context" instruction, refusal-on-no-context, and optional citation enforcement based on domain risk.

---

## 10. Frontend (React — fixed)

- Talks only to Controllers.
- Transport per feature mirrors Section 7: REST hooks for query, an SSE/WS client for streaming + notifications.
- Minimum surfaces: query input, streamed answer with citations, ingestion/upload + progress (via WS/SSE), notification toasts (Observer over WS).
- State/styling/router choices are **AGENT DECISION**, but keep the View free of business logic.

---

## 11. Agent Self-Check (run before finishing)

- [ ] Every AGENT DECISION point resolved and logged in `DECISIONS.md`.
- [ ] e5 `passage:`/`query:` prefixes applied; vector size 768, cosine.
- [ ] Default retrieval is hybrid unless domain justified otherwise.
- [ ] No Controller holds business logic; no Service imports HTTP objects.
- [ ] Notifications use Observer + WebSocket, **not** REST polling.
- [ ] Provider clients (Qdrant/Groq/embeddings) behind Adapters.
- [ ] Orchestration tier is the lightest that works.
- [ ] Re-ingest is idempotent (deterministic IDs).
- [ ] React talks only to Controllers; citations rendered.
- [ ] `.env.example` complete; `docker-compose` brings up qdrant + backend + frontend.

---

## 12. Minimal Reference Snippets

**Embedding with prefixes (repository):**
```python
def embed_passages(texts): return model.encode([f"passage: {t}" for t in texts], normalize_embeddings=True)
def embed_query(text):     return model.encode(f"query: {text}", normalize_embeddings=True)
```

**Retrieval Strategy interface:**
```python
class RetrievalStrategy(Protocol):
    def search(self, query: str, top_k: int) -> list[Hit]: ...

# HybridStrategy | DenseStrategy | SparseStrategy chosen from RETRIEVAL_MODE
```

**Observer for notifications:**
```python
class Subject:
    def __init__(self): self._subs = []
    def subscribe(self, ws): self._subs.append(ws)
    async def notify(self, event):
        for ws in self._subs: await ws.send_json(event)  # pushed over WebSocket
```

**LLM Adapter:**
```python
class LLMClient(Protocol):
    def generate(self, prompt: str, stream: bool) -> ...: ...
class GroqClient(LLMClient): ...   # swappable via Factory
```

---

*End of template. The Agent must produce `DECISIONS.md` alongside the generated project documenting each resolved choice.*
