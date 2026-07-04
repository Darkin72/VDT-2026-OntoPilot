import json
import os
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any

from app import logging_service


DEFAULT_LOOKUP_DB = Path("/app/Ontology/normalized/embedding_lookup.sqlite")
DEFAULT_DOCUMENTS = Path("/app/Ontology/normalized/embedding_documents.jsonl")
WORD_PATTERN = re.compile(r"[^\W\d_][\w.-]*", re.UNICODE)
QUOTED_PATTERN = re.compile(r"['\"]([^'\"]{2,120})['\"]")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "is", "of", "on", "or", "the", "to", "was", "were", "what", "when", "where", "which", "who", "whom", "whose", "with",
    "find", "get", "list", "return", "show", "identify", "determine", "query", "question", "answer", "resource", "entity", "predicate", "property", "relation", "relationship",
}
RELATION_HINTS = [
    (("born", "birth", "birthplace"), ("birth place", "birthplace", "birth date", "birthdate")),
    (("died", "death"), ("death place", "deathplace", "death date", "deathdate")),
    (("spouse", "wife", "husband", "married"), ("spouse",)),
    (("parent", "father", "mother"), ("parent", "father", "mother")),
    (("child", "children", "son", "daughter"), ("child", "children")),
    (("founder", "founded"), ("founder", "founding date")),
    (("capital",), ("capital",)),
    (("population",), ("population total", "populationtotal", "population")),
    (("area",), ("area total", "areatotal", "area")),
    (("award", "prize"), ("award",)),
    (("author", "wrote", "writer"), ("author", "writer")),
]


def lookup_db_path() -> Path:
    return Path(os.getenv("ONTOLOGY_LOOKUP_DB_PATH", str(DEFAULT_LOOKUP_DB)))


def documents_path() -> Path:
    return Path(os.getenv("ONTOLOGY_LOOKUP_DOCUMENTS_PATH", str(DEFAULT_DOCUMENTS)))


def lookup_enabled() -> bool:
    legacy_enabled = os.getenv("ONTOLOGY_RAG_ENABLED", "true").strip().lower() in {"1", "true", "yes"}
    lookup_enabled_value = os.getenv("ONTOLOGY_LOOKUP_ENABLED", "true").strip().lower() in {"1", "true", "yes"}
    return legacy_enabled and lookup_enabled_value


def top_k() -> int:
    try:
        return max(1, int(os.getenv("ONTOLOGY_RAG_TOP_K", "8")))
    except ValueError:
        return 8


def max_query_terms() -> int:
    try:
        return min(10, max(5, int(os.getenv("ONTOLOGY_LOOKUP_QUERY_COUNT", "10"))))
    except ValueError:
        return 10


def per_query_limit() -> int:
    try:
        return max(1, int(os.getenv("ONTOLOGY_LOOKUP_PER_QUERY_LIMIT", "3")))
    except ValueError:
        return 3


def normalize_term(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return " ".join(value.replace("_", " ").casefold().split())


def strip_punctuation(value: str) -> str:
    return value.strip(" \t\r\n.,;:!?()[]{}<>`~|/@#$%^&*+=\\")


def add_unique(items: list[str], value: str) -> None:
    value = strip_punctuation(value)
    if not value:
        return
    normalized = normalize_term(value)
    if len(normalized) < 2:
        return
    if normalized in {normalize_term(item) for item in items}:
        return
    items.append(value)


def capitalized_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for segment in re.split(r"[\n\r\t,;:!?()\[\]{}]+", text):
        words = WORD_PATTERN.findall(segment)
        index = 0
        while index < len(words):
            if not words[index][:1].isupper():
                index += 1
                continue
            phrase = [words[index]]
            index += 1
            while index < len(words) and len(phrase) < 6 and words[index][:1].isupper():
                phrase.append(words[index])
                index += 1
            if len(phrase) >= 2:
                add_unique(phrases, " ".join(phrase))
        
    return phrases


def meaningful_ngrams(text: str) -> list[str]:
    words = [word for word in WORD_PATTERN.findall(text) if normalize_term(word) not in STOP_WORDS]
    terms: list[str] = []
    for size in (4, 3, 2):
        for index in range(0, max(0, len(words) - size + 1)):
            phrase = " ".join(words[index:index + size])
            if any(word[:1].isupper() for word in words[index:index + size]) or size <= 3:
                add_unique(terms, phrase)
            if len(terms) >= 8:
                return terms
    return terms


def relation_terms(text: str) -> list[str]:
    normalized = normalize_term(text)
    terms: list[str] = []
    for triggers, candidates in RELATION_HINTS:
        if any(trigger in normalized for trigger in triggers):
            for candidate in candidates:
                add_unique(terms, candidate)
    return terms


def generate_lookup_terms(query: str) -> list[str]:
    terms: list[str] = []
    for match in QUOTED_PATTERN.findall(query):
        add_unique(terms, match)
    for phrase in capitalized_phrases(query):
        add_unique(terms, phrase)
    for term in relation_terms(query):
        add_unique(terms, term)
    for term in meaningful_ngrams(query):
        add_unique(terms, term)
    add_unique(terms, query)
    return terms[:max_query_terms()]


def fetch_document(handle, byte_offset: int) -> dict[str, Any]:
    handle.seek(byte_offset)
    document = json.loads(handle.readline())
    if not isinstance(document, dict):
        return {}
    payload = dict(document.get("payload") or {})
    for key in ("id", "parent_id", "chunk_index", "chunk_count", "kind", "uri", "curie", "label", "text"):
        if key in document:
            payload[key] = document[key]
    return payload


def lookup_term(connection: sqlite3.Connection, term: str, *, limit: int) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT d.* FROM terms t
        JOIN documents d ON d.doc_id = t.doc_id
        WHERE t.term = ?
        ORDER BY d.line_number
        LIMIT ?
        """,
        (normalize_term(term), limit),
    ).fetchall()


def retrieve_candidates_for_terms(terms: list[str], *, source_query: str = "") -> list[dict[str, Any]]:
    clean_terms: list[str] = []
    for term in terms:
        add_unique(clean_terms, str(term))
        if len(clean_terms) >= max_query_terms():
            break

    term_order = {term: index for index, term in enumerate(clean_terms)}
    clean_terms.sort(key=lambda term: (0 if normalize_term(term).startswith("dbr:") else 1, term_order[term]))

    if not lookup_enabled() or not clean_terms:
        return []

    db_path = lookup_db_path()
    source_path = documents_path()
    if not db_path.exists():
        logging_service.agent_step("ontology_lookup.missing_db", {"db_path": str(db_path)}, limit=1000)
        return []
    if not source_path.exists():
        logging_service.agent_step("ontology_lookup.missing_documents", {"documents_path": str(source_path)}, limit=1000)
        return []

    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    matched_terms: list[str] = []
    try:
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        with source_path.open("rb") as handle:
            for term in clean_terms:
                rows = lookup_term(connection, term, limit=per_query_limit())
                if rows:
                    matched_terms.append(term)
                for row in rows:
                    doc_id = str(row["doc_id"])
                    if doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)
                    candidate = fetch_document(handle, int(row["byte_offset"]))
                    candidate.setdefault("id", doc_id)
                    candidate.setdefault("kind", row["kind"])
                    candidate.setdefault("label", row["label"])
                    candidate.setdefault("curie", row["curie"])
                    candidate.setdefault("uri", row["uri"])
                    candidate["score"] = 1.0
                    candidate["matched_lookup_term"] = term
                    candidates.append(candidate)
                    if len(candidates) >= top_k():
                        break
                if len(candidates) >= top_k():
                    break
        connection.close()
        logging_service.trace_step(
            "ontology_lookup.retrieved",
            {
                "db_path": str(db_path),
                "source_query": source_query,
                "query_terms": clean_terms,
                "matched_terms": matched_terms,
                "candidate_count": len(candidates),
                "candidates": [format_candidate(candidate) for candidate in candidates[:5]],
            },
            limit=5000,
        )
        return candidates
    except Exception as exc:  # noqa: BLE001 - lookup is optional and must not break chat.
        logging_service.agent_step(
            "ontology_lookup.error",
            {"type": type(exc).__name__, "message": str(exc), "db_path": str(db_path)},
            limit=2000,
        )
        return []

def retrieve_candidates(query: str) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    return retrieve_candidates_for_terms(generate_lookup_terms(query), source_query=query)


def format_candidate(candidate: dict[str, Any]) -> str:
    curie = str(candidate.get("curie") or candidate.get("uri") or "").strip()
    kind = str(candidate.get("kind") or "").strip()
    label = str(candidate.get("label") or candidate.get("label_en") or "").strip()
    domain = str(candidate.get("domain") or candidate.get("domains_json") or "").strip()
    range_value = str(candidate.get("range") or candidate.get("ranges_json") or "").strip()
    sub_class_of = str(candidate.get("sub_class_of") or candidate.get("subclass_of_json") or "").strip()
    matched_term = str(candidate.get("matched_lookup_term") or "").strip()

    parts = [part for part in [curie, kind, f"label {label}" if label else ""] if part]
    if domain and domain != "[]":
        parts.append(f"domain {domain}")
    if range_value and range_value != "[]":
        parts.append(f"range {range_value}")
    if sub_class_of and sub_class_of != "[]":
        parts.append(f"subClassOf {sub_class_of}")
    if matched_term:
        parts.append(f"matched {matched_term}")
    return " | ".join(parts)


def format_candidates_block(query: str) -> str:
    candidates = retrieve_candidates(query)
    if not candidates:
        return ""

    lines = ["Retrieved ontology/resource URI candidates:"]
    lines.extend(f"- {format_candidate(candidate)}" for candidate in candidates)
    return "\n".join(lines)
