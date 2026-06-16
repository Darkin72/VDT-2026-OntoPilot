import os
from typing import Any

import requests


def embedding_api_url() -> str:
    base_url = os.getenv("EMBEDDING_API_BASE_URL", "http://embedding.llm.mobifone.vn").strip().rstrip("/")
    path = os.getenv("EMBEDDING_API_PATH", "/v1/embeddings").strip() or "/v1/embeddings"
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{base_url}{path}"


def embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", "default").strip() or "default"


def embed_texts(texts: list[str], *, timeout_seconds: int = 120) -> list[list[float]]:
    if not texts:
        return []

    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(
        embedding_api_url(),
        json={"model": embedding_model(), "input": texts},
        headers=headers,
        timeout=(10, timeout_seconds),
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    data = payload.get("data", [])
    if not isinstance(data, list) or len(data) != len(texts):
        raise ValueError("Embedding API returned an unexpected data length")

    embeddings: list[list[float]] = []
    for item in data:
        embedding = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("Embedding API returned an invalid embedding")
        embeddings.append([float(value) for value in embedding])
    return embeddings


def embed_text(text: str, *, timeout_seconds: int = 120) -> list[float]:
    return embed_texts([text], timeout_seconds=timeout_seconds)[0]

