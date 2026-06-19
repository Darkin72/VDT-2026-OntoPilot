import json
import os
import time
from collections.abc import Iterable
from typing import Any

import requests

from app import logging_service

PRIORITY_PREDICATE_TERMS = (
    "place",
    "location",
    "date",
    "birth",
    "death",
    "resting",
    "burial",
    "subdivision",
    "country",
    "relation",
    "award",
    "founder",
)

def discover_repository(graphdb_url: str) -> str:
    try:
        response = requests.get(f"{graphdb_url}/rest/repositories", timeout=(3, 10))
        response.raise_for_status()
        repositories = response.json()
    except (requests.RequestException, json.JSONDecodeError):
        return ""

    if not isinstance(repositories, list):
        return ""

    for repository in repositories:
        repository_id = str(repository.get("id", "")).strip()
        if repository_id and repository_id.upper() != "SYSTEM":
            return repository_id
    return ""

def resolve_repository(graphdb_url: str, configured_repository: str) -> str:
    configured_repository = configured_repository.strip()
    if not configured_repository:
        return discover_repository(graphdb_url) or "DBPEDIA"

    try:
        response = requests.get(f"{graphdb_url}/rest/repositories", timeout=(3, 10))
        response.raise_for_status()
        repositories = response.json()
    except (requests.RequestException, json.JSONDecodeError):
        return configured_repository

    if not isinstance(repositories, list):
        return configured_repository

    for repository in repositories:
        repository_id = str(repository.get("id", "")).strip()
        if repository_id == configured_repository:
            return repository_id

    configured_lower = configured_repository.lower()
    for repository in repositories:
        repository_id = str(repository.get("id", "")).strip()
        if repository_id.lower() == configured_lower:
            return repository_id

    return configured_repository


def build_repository_url() -> str:
    explicit_url = os.getenv("GRAPHDB_REPOSITORY_URL", "").strip().rstrip("/")
    if explicit_url:
        return explicit_url

    graphdb_url = os.getenv("GRAPHDB_URL", "http://graphdb:7200").rstrip("/")
    repository = resolve_repository(graphdb_url, os.getenv("GRAPHDB_REPOSITORY", ""))
    return f"{graphdb_url}/repositories/{repository}"

def configured_query_timeout_seconds() -> int:
    return int(os.getenv("GRAPHDB_QUERY_TIMEOUT_SECONDS", "120"))

def query_max_attempts() -> int:
    return max(1, int(os.getenv("GRAPHDB_QUERY_MAX_ATTEMPTS", "3")))

def query_retry_delay_seconds() -> float:
    return max(0.0, float(os.getenv("GRAPHDB_QUERY_RETRY_DELAY_SECONDS", "10")))

def effective_query_timeout_seconds(
    *,
    remaining_question_seconds: float | None = None,
    remaining_graphdb_queries: int = 1,
) -> int:
    configured_timeout = configured_query_timeout_seconds()
    if remaining_question_seconds is None:
        return configured_timeout

    finalization_reserve = int(os.getenv("QUESTION_FINALIZATION_RESERVE_SECONDS", "240"))
    usable_seconds = int(remaining_question_seconds) - finalization_reserve
    if usable_seconds <= 0:
        return 0

    return max(1, min(configured_timeout, usable_seconds))

def sparql_one_line(sparql: str) -> str:
    return " ".join(sparql.split())

def query(sparql: str, *, timeout_seconds: int | None = None) -> dict[str, Any]:
    repository_url = build_repository_url()
    timeout_seconds = timeout_seconds if timeout_seconds is not None else configured_query_timeout_seconds()
    attempts = query_max_attempts()
    retry_delay_seconds = query_retry_delay_seconds()
    compact_sparql = sparql_one_line(sparql)

    for attempt in range(1, attempts + 1):
        logging_service.agent_step(
            "graphdb.query_start",
            {
                "attempt": attempt,
                "max_attempts": attempts,
                "repository_url": repository_url,
                "timeout_seconds": timeout_seconds,
                "sparql": compact_sparql,
            },
        )
        try:
            response = requests.post(
                repository_url,
                data={"query": sparql},
                headers={"Accept": "application/sparql-results+json"},
                timeout=(5, timeout_seconds),
            )
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                logging_service.agent_step(
                    "graphdb.query_error",
                    {
                        "attempt": attempt,
                        "max_attempts": attempts,
                        "retry": attempt < attempts,
                        "status_code": response.status_code,
                        "reason": response.reason,
                        "body": response.text,
                    },
                    limit=5000,
                )
                raise requests.HTTPError(
                    f"{exc}; GraphDB response body: {response.text}",
                    response=response,
                ) from exc

            result = response.json()
            logging_service.agent_step("graphdb.query_result_summary", summarize_result(result))
            return result
        except requests.RequestException as exc:
            if attempt >= attempts:
                raise
            logging_service.agent_step(
                "graphdb.query_retry",
                {
                    "attempt": attempt,
                    "max_attempts": attempts,
                    "delay_seconds": retry_delay_seconds,
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                limit=2000,
            )
            time.sleep(retry_delay_seconds)

    raise RuntimeError("unreachable graphdb query retry state")

def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    if "boolean" in result:
        return {"type": "ask", "value": bool(result["boolean"])}

    bindings = result.get("results", {}).get("bindings", [])
    sample = bindings[:2] if isinstance(bindings, list) else []
    return {
        "type": "select",
        "vars": result.get("head", {}).get("vars", []),
        "row_count": len(bindings) if isinstance(bindings, list) else 0,
        "sample": sample,
    }

def has_result(result: dict[str, Any]) -> bool:
    if "boolean" in result:
        return True
    bindings = result.get("results", {}).get("bindings", [])
    return isinstance(bindings, list) and len(bindings) > 0

def _binding_value(binding: dict[str, Any], variable: str) -> str:
    value = binding.get(variable, {})
    return str(value.get("value", "")) if isinstance(value, dict) else ""

def _binding_predicate_values(binding: dict[str, Any]) -> Iterable[str]:
    for variable in ("predicate", "p", "rel", "property"):
        value = _binding_value(binding, variable)
        if value:
            yield value

def _is_rdf_type_binding(binding: dict[str, Any]) -> bool:
    return any(value == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" for value in _binding_predicate_values(binding))

def _priority_score(binding: dict[str, Any]) -> int:
    predicate_text = " ".join(_binding_predicate_values(binding)).lower()
    if any(term in predicate_text for term in PRIORITY_PREDICATE_TERMS):
        return 0
    if _is_rdf_type_binding(binding):
        return 2
    return 1

def prioritized_bindings(bindings: list[Any], *, max_rows: int) -> list[dict[str, Any]]:
    typed_bindings = [binding for binding in bindings if isinstance(binding, dict)]
    if len(typed_bindings) <= max_rows:
        return typed_bindings

    indexed = list(enumerate(typed_bindings))
    indexed.sort(key=lambda item: (_priority_score(item[1]), item[0]))
    return [binding for _, binding in indexed[:max_rows]]

def compact_result(result: dict[str, Any], *, max_rows: int | None = None) -> dict[str, Any]:
    if "boolean" in result:
        return {"ask": bool(result["boolean"])}

    variables = result.get("head", {}).get("vars", [])
    bindings = result.get("results", {}).get("bindings", [])
    row_count = len(bindings) if isinstance(bindings, list) else 0
    max_rows = max_rows if max_rows is not None else int(os.getenv("GRAPHDB_SAMPLE_MAX_ROWS", os.getenv("GRAPHDB_MAX_ROWS", "20")))
    sample_bindings = prioritized_bindings(bindings, max_rows=max_rows) if isinstance(bindings, list) else []
    rows: list[dict[str, str]] = []
    for binding in sample_bindings:
        row: dict[str, str] = {}
        for variable in variables:
            row[variable] = _binding_value(binding, variable)
        rows.append(row)
    return {"rows": rows, "row_count": row_count}

def format_result(result: dict[str, Any]) -> str:
    return json.dumps(compact_result(result), ensure_ascii=False, indent=2)
