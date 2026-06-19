# VDT Chatbot Stack

The stack has these services:

- `backend`: FastAPI backend that calls a custom chat-completions-compatible model API with `requests`, queries GraphDB first, and streams responses at `POST /api/chat`.
- `frontend`: Svelte/Vite chat UI at `http://localhost:5200`.
- `graphdb`: Ontotext GraphDB published at `http://localhost:7200`, using repository `VDT`.

## Run With Docker Compose

```bash
cd docker
docker compose up --build
```

Open:

- Frontend: http://localhost:5200
- Backend health: http://localhost:8090/health
- GraphDB: http://localhost:7200
- Qdrant: http://localhost:6363

## Build The Ontology URI Index

After the stack is running, build the phase-1 ontology RAG index manually:

```bash
docker compose exec backend python -m app.rag.index_ontology --dry-run
docker compose exec backend python -m app.rag.index_ontology --batch-size 8 --concurrency 4 --max-text-length 1024
```

The indexer reads `Ontology/ontology--DEV_type=parsed_sorted.nt`, embeds one document per ontology URI, and upserts it into the `ontology_uri` Qdrant collection. Runtime SPARQL generation uses this collection when `ONTOLOGY_RAG_ENABLED=true`.

## Configure The Model API

Copy the sample env file:

```bash
cp backend/.env-example backend/.env
```

Update `backend/.env` for your API:

```env
CHAT_API_BASE_URL=https://your-model-provider.example.com
CHAT_API_PATH=/v1/chat/completions
CHAT_API_KEY=your_api_key_here
CHAT_API_AUTH_HEADER=Authorization
CHAT_API_AUTH_PREFIX=Bearer
CHAT_MODEL=your-model-name
CHAT_API_STREAM=true
CHAT_API_CONNECT_TIMEOUT_SECONDS=10
CHAT_API_READ_TIMEOUT_SECONDS=120
CHAT_API_MAX_RETRIES=2
CHAT_API_RETRY_BACKOFF_SECONDS=1
GRAPHDB_URL=http://graphdb:7200
GRAPHDB_REPOSITORY=VDT
GRAPHDB_QUERY_TIMEOUT_SECONDS=120
QUESTION_TIMEOUT_SECONDS=1200
QUESTION_FINALIZATION_RESERVE_SECONDS=240
LLM_HISTORY_MAX_CHARS=50000
LLM_HISTORY_FIELD_MAX_CHARS=8000
```

The backend uses a two-agent flow. GraphDB query timeout has a configurable cap of 120 seconds per SPARQL query by default, and each query is also capped by the remaining `QUESTION_TIMEOUT_SECONDS` budget after `QUESTION_FINALIZATION_RESERVE_SECONDS`. The central agent first decides whether GraphDB is needed and writes a precise query description. If lookup is needed, the SPARQL coder agent turns that description into a read-only SELECT or ASK query. The backend executes the query, sends the GraphDB result back to the central agent, and the central agent writes the final answer. If GraphDB is not needed, times out, or returns no rows, the central agent can answer without GraphDB.

LLM calls retry transient connection failures, timeouts, and HTTP `429/5xx` responses before any stream chunk is emitted. HTTP `400` is not retried because it usually means the request payload or model configuration is invalid. By default the backend tries the initial call plus 2 retries, with a 1-second linear backoff.

The model API request uses an OpenAI-compatible chat-completions streaming format, but it does not use the OpenAI SDK. Set `CHAT_API_STREAM=false` if the provider only supports non-streaming JSON responses. Browser-like headers are sent by default: `User-Agent`, `Accept`, `Accept-Language`, `Origin`, `Referer`, `Sec-Fetch-*`, and `sec-ch-ua*`. If the API requires custom headers, add JSON to:

```env
CHAT_API_EXTRA_HEADERS={"X-Custom-Header":"value"}
```

If `CHAT_API_BASE_URL` is missing, the backend returns HTTP 500 so configuration errors are visible immediately.
