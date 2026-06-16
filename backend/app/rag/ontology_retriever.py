import os
from typing import Any

from app import logging_service
from app.rag import embedding_service, qdrant_store


def rag_enabled() -> bool:
    return os.getenv("ONTOLOGY_RAG_ENABLED", "true").strip().lower() in {"1", "true", "yes"}


def top_k() -> int:
    try:
        return max(1, int(os.getenv("ONTOLOGY_RAG_TOP_K", "8")))
    except ValueError:
        return 8


def retrieve_candidates(query: str) -> list[dict[str, Any]]:
    if not rag_enabled() or not query.strip():
        return []

    try:
        vector = embedding_service.embed_text(query)
        candidates = qdrant_store.search(vector, limit=top_k())
        logging_service.agent_step(
            "ontology_rag.retrieved",
            {
                "collection": qdrant_store.qdrant_collection(),
                "candidate_count": len(candidates),
                "candidates": [format_candidate(candidate) for candidate in candidates[:5]],
            },
            limit=4000,
        )
        return candidates
    except Exception as exc:  # noqa: BLE001 - RAG is optional and must not break chat.
        logging_service.agent_step(
            "ontology_rag.error",
            {"type": type(exc).__name__, "message": str(exc)},
            limit=2000,
        )
        return []


def format_candidate(candidate: dict[str, Any]) -> str:
    curie = str(candidate.get("curie") or candidate.get("uri") or "").strip()
    kind = str(candidate.get("kind") or "").strip()
    label = str(candidate.get("label") or "").strip()
    domain = str(candidate.get("domain") or "").strip()
    range_value = str(candidate.get("range") or "").strip()
    sub_class_of = str(candidate.get("sub_class_of") or "").strip()

    parts = [part for part in [curie, kind, f"label {label}" if label else ""] if part]
    if domain:
        parts.append(f"domain {domain}")
    if range_value:
        parts.append(f"range {range_value}")
    if sub_class_of:
        parts.append(f"subClassOf {sub_class_of}")
    return " | ".join(parts)


def format_candidates_block(query: str) -> str:
    candidates = retrieve_candidates(query)
    if not candidates:
        return ""

    lines = ["Retrieved ontology URI candidates:"]
    lines.extend(f"- {format_candidate(candidate)}" for candidate in candidates)
    return "\n".join(lines)

