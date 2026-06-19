import json
import os
import time
from collections.abc import Iterator
from typing import Any

import requests

from app import graphdb_service, llm_service, logging_service
from app.agents import answer_formatter, central, sparql as sparql_agent
from app.agents import summary as summary_agent

def max_agent_rounds() -> int:
    return central.max_agent_rounds()

def max_subqueries_per_round() -> int:
    return central.max_subqueries_per_round()

def question_timeout_seconds() -> int:
    return int(os.getenv("QUESTION_TIMEOUT_SECONDS", "1200"))

def max_sparql_agent_attempts() -> int:
    return max(1, int(os.getenv("SPARQL_AGENT_MAX_ATTEMPTS", "3")))

def remaining_seconds(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())

def stream_with_optional_verbose_logging(messages: list[llm_service.ChatMessage], step: str) -> Iterator[str]:
    chunks: list[str] = []
    for chunk in llm_service.stream_messages(messages):
        chunks.append(chunk)
        yield chunk
    logging_service.verbose_text(step, "".join(chunks).strip())

def normalized_final_stream(
    message: str,
    raw_text: str,
    history: dict[str, Any],
) -> Iterator[str]:
    graphdb_result, graphdb_error = answer_formatter.latest_graphdb_evidence(history)
    final_text = central.normalize_answer_evidence_response(message, raw_text, graphdb_result, graphdb_error)
    yield final_text.strip()

def _remaining_graphdb_queries(round_index: int, subquery_index: int) -> int:
    current_round_remaining = max_subqueries_per_round() - subquery_index + 1
    later_rounds = max(0, max_agent_rounds() - round_index) * max_subqueries_per_round()
    return max(1, current_round_remaining + later_rounds)

def execute_subquery_step(
    message: str,
    history: dict[str, Any],
    round_data: dict[str, Any],
    subquery: dict[str, str],
    *,
    round_index: int,
    subquery_index: int,
    deadline: float,
) -> None:
    sparql = ""
    graphdb_result = None
    graphdb_error = None
    attempts: list[dict[str, Any]] = []
    graphdb_timeout_seconds = graphdb_service.effective_query_timeout_seconds(
        remaining_question_seconds=remaining_seconds(deadline),
        remaining_graphdb_queries=_remaining_graphdb_queries(round_index, subquery_index),
    )

    if graphdb_timeout_seconds <= 0:
        graphdb_error = "QUESTION_TIMEOUT: skipped GraphDB to reserve time for final answer."
    else:
        for sparql_attempt in range(1, max_sparql_agent_attempts() + 1):
            sparql = ""
            graphdb_result = None
            graphdb_error = None
            try:
                sparql = sparql_agent.generate_sparql(
                    message,
                    subquery["description"],
                    history=history,
                    subquery_id=subquery.get("id"),
                    round_context={
                        "round": round_index,
                        "subquery": subquery,
                        "sparql_attempt": sparql_attempt,
                        "max_sparql_attempts": max_sparql_agent_attempts(),
                        "current_subquery_attempts": attempts,
                        "current_round_executions": [
                            central.compact_execution_for_llm(execution)
                            for execution in round_data.get("executions", [])
                            if isinstance(execution, dict)
                        ],
                    },
                )
                if sparql:
                    graphdb_result = graphdb_service.query(sparql, timeout_seconds=graphdb_timeout_seconds)
                else:
                    graphdb_error = "SPARQL_GENERATION_EMPTY_OR_REJECTED"
            except requests.Timeout as exc:
                graphdb_error = "GRAPHDB_TIMEOUT: GraphDB query exceeded the configured timeout."
                logging_service.agent_step(
                    "agent_pipeline.graphdb_timeout",
                    {"type": type(exc).__name__, "message": str(exc), "sparql": sparql},
                    limit=5000,
                )
            except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
                graphdb_error = f"GRAPHDB_ERROR: {type(exc).__name__}: {exc}"
                logging_service.agent_step("agent_pipeline.error", {"type": type(exc).__name__, "message": str(exc)})

            attempt_record = {
                "attempt": sparql_attempt,
                "sparql": sparql,
                "graphdb_timeout_seconds": graphdb_timeout_seconds,
                "result": graphdb_result,
                "result_summary": graphdb_service.summarize_result(graphdb_result) if graphdb_result else None,
                "error": graphdb_error,
            }
            attempts.append(attempt_record)
            logging_service.agent_step(
                "agent_pipeline.subquery_sparql_attempt",
                {**attempt_record, "result": None},
                limit=5000,
            )
            if graphdb_error is None:
                break

    execution = {
        "round": round_index,
        "subquery_index": subquery_index,
        "subquery_id": subquery.get("id", f"q{subquery_index}"),
        "query_description": subquery["description"],
        "purpose": subquery.get("purpose", ""),
        "expected_evidence": subquery.get("expected_evidence", ""),
        "sparql": sparql,
        "graphdb_timeout_seconds": graphdb_timeout_seconds,
        "result": graphdb_result,
        "result_summary": graphdb_service.summarize_result(graphdb_result) if graphdb_result else None,
        "error": graphdb_error,
        "sparql_attempts": attempts,
    }
    round_data.setdefault("executions", []).append(execution)
    logging_service.agent_step("agent_pipeline.subquery_execution", central.compact_execution_for_llm(execution), limit=5000)

def fold_old_rounds_into_summary(history: dict[str, Any]) -> None:
    rounds = history.get("rounds", [])
    if not isinstance(rounds, list):
        return

    raw_window = central.raw_round_window()
    target_summarized = max(0, len(rounds) - raw_window)
    summarized_count = int(history.get("summarized_round_count") or 0)
    original_prompt = str(history.get("original_prompt", ""))
    while summarized_count < target_summarized:
        round_data = rounds[summarized_count]
        if isinstance(round_data, dict):
            history["accumulated_summary"] = summary_agent.summarize_round(
                original_prompt=original_prompt,
                accumulated_summary=history.get("accumulated_summary") if isinstance(history.get("accumulated_summary"), dict) else None,
                round_data=round_data,
            )
        summarized_count += 1
    history["summarized_round_count"] = summarized_count

def agent_stream(message: str) -> Iterator[str]:
    try:
        deadline = time.monotonic() + question_timeout_seconds()
        logging_service.agent_text("user.prompt", message)
        history = central.build_history(message)
        routing_decision = central.plan_graphdb_usage(message)
        central.add_history_step(history, "central_routing", routing_decision)

        if routing_decision["use_graphdb"]:
            focus = routing_decision["query_description"]
            for round_index in range(1, max_agent_rounds() + 1):
                round_data: dict[str, Any] = {
                    "round": round_index,
                    "focus": focus,
                    "plan": None,
                    "executions": [],
                    "evaluation": None,
                }
                history.setdefault("rounds", []).append(round_data)

                plan = central.plan_subqueries(message, history, round_index=round_index, focus=focus)
                round_data["plan"] = plan
                for subquery_index, subquery in enumerate(plan["subqueries"], start=1):
                    execute_subquery_step(
                        message,
                        history,
                        round_data,
                        subquery,
                        round_index=round_index,
                        subquery_index=subquery_index,
                        deadline=deadline,
                    )

                evaluation = central.evaluate_round(
                    message,
                    history,
                    round_index=round_index,
                    max_rounds=max_agent_rounds(),
                )
                round_data["evaluation"] = evaluation
                if evaluation["action"] != "continue":
                    break

                focus = evaluation["next_focus"]
                fold_old_rounds_into_summary(history)

        raw_answer = answer_formatter.format_final_answer(message, history)
        yield from normalized_final_stream(message, raw_answer, history)
    except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
        logging_service.logger.exception("agent_pipeline.llm_error")
        yield f"Xin loi, hien tai backend khong goi duoc model API ({type(exc).__name__})."
