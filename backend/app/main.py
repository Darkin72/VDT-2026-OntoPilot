import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app import chat_service, graphdb_service, llm_service
from app.schemas import ChatRequest

load_dotenv()

app = FastAPI(title="VDT Chatbot API", version="0.1.0")

frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin, "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "graphdb_url": os.getenv("GRAPHDB_URL", "http://graphdb:7200"),
        "graphdb_repository_url": graphdb_service.build_repository_url(),
        "chat_api_base_url": os.getenv("CHAT_API_BASE_URL", "not-configured"),
    }


@app.post("/api/chat")
def chat(payload: ChatRequest) -> StreamingResponse:
    if not llm_service.build_api_url():
        raise HTTPException(status_code=500, detail="CHAT_API_BASE_URL is not configured")

    return StreamingResponse(
        chat_service.agent_stream(payload.message),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
