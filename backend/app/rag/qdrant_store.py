import os
import uuid
from collections.abc import Iterable
from typing import Any


def qdrant_url() -> str:
    return os.getenv("QDRANT_URL", "http://qdrant:6333").strip().rstrip("/")


def qdrant_collection() -> str:
    return os.getenv("QDRANT_COLLECTION", "ontology_uri").strip() or "ontology_uri"


def point_id(uri: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, uri))


def make_client():
    from qdrant_client import QdrantClient

    return QdrantClient(url=qdrant_url())


def recreate_collection(*, vector_size: int) -> None:
    from qdrant_client import models

    client = make_client()
    client.recreate_collection(
        collection_name=qdrant_collection(),
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )


def upsert_points(points: Iterable[tuple[str, list[float], dict[str, Any]]]) -> None:
    from qdrant_client import models

    client = make_client()
    qdrant_points = [
        models.PointStruct(id=point_id(uri), vector=vector, payload=payload)
        for uri, vector, payload in points
    ]
    if qdrant_points:
        client.upsert(collection_name=qdrant_collection(), points=qdrant_points)


def search(vector: list[float], *, limit: int) -> list[dict[str, Any]]:
    client = make_client()
    collection = qdrant_collection()
    if hasattr(client, "search"):
        hits = client.search(collection_name=collection, query_vector=vector, limit=limit, with_payload=True)
    else:
        result = client.query_points(collection_name=collection, query=vector, limit=limit, with_payload=True)
        hits = result.points

    results: list[dict[str, Any]] = []
    for hit in hits:
        payload = dict(hit.payload or {})
        payload["score"] = float(hit.score)
        results.append(payload)
    return results

