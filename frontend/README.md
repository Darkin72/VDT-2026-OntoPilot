# VDT Chatbot Stack

The stack has 3 services:

- `backend`: FastAPI backend that calls a custom chat-completions-compatible model API with `requests`, queries GraphDB first, and streams responses at `POST /api/chat`.
- `frontend`: Svelte/Vite chat UI at `http://localhost:5173`.
- `graphdb`: Ontotext GraphDB at `http://localhost:7200`, with the `Ontology` folder mounted as the import folder.

## Run With Docker Compose

```bash
cd docker
docker compose up --build
```

Open:

- Frontend: http://localhost:5173
- Backend health: http://localhost:8000/health
- GraphDB: http://localhost:7200

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
GRAPHDB_REPOSITORY=vdt
GRAPHDB_QUERY_TIMEOUT_SECONDS=200
QUESTION_TIMEOUT_SECONDS=1200
QUESTION_FINALIZATION_RESERVE_SECONDS=240
LLM_HISTORY_MAX_CHARS=50000
LLM_HISTORY_FIELD_MAX_CHARS=8000
```

The backend uses a two-agent flow. GraphDB query timeout has a configurable cap of 200 seconds per query, but each query is capped dynamically by the remaining `QUESTION_TIMEOUT_SECONDS` budget. With the default `QUESTION_TIMEOUT_SECONDS=1200`, `MAX_SPARQL_ATTEMPTS=5`, and `QUESTION_FINALIZATION_RESERVE_SECONDS=240`, the first GraphDB timeout is capped at 192s in the worst case of 5 possible queries: `(1200 - 240) / 5`. Later queries are recalculated from the remaining time. The central agent first decides whether GraphDB is needed and writes a precise query description. If lookup is needed, the SPARQL coder agent turns that description into a read-only SELECT or ASK query. The backend executes the query, sends the GraphDB result back to the central agent, and the central agent writes the final answer. If GraphDB is not needed, times out, or returns no rows, the central agent can answer without GraphDB.

LLM calls retry transient connection failures, timeouts, and HTTP `429/5xx` responses before any stream chunk is emitted. HTTP `400` is not retried because it usually means the request payload or model configuration is invalid. By default the backend tries the initial call plus 2 retries, with a 1-second linear backoff.

The model API request uses an OpenAI-compatible chat-completions streaming format, but it does not use the OpenAI SDK. Set `CHAT_API_STREAM=false` if the provider only supports non-streaming JSON responses. Browser-like headers are sent by default: `User-Agent`, `Accept`, `Accept-Language`, `Origin`, `Referer`, `Sec-Fetch-*`, and `sec-ch-ua*`. If the API requires custom headers, add JSON to:

```env
CHAT_API_EXTRA_HEADERS={"X-Custom-Header":"value"}
```

If `CHAT_API_BASE_URL` is missing, the backend returns HTTP 500 so configuration errors are visible immediately.
