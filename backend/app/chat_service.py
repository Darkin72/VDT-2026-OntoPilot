import json
import os
import re
import time
import unicodedata
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





def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _normalize_match_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("_", " ").replace("-", " ").casefold()
    return re.sub(r"\s+", " ", value).strip()


def _multiple_choice_options(message: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for match in re.finditer(r"(?m)^\s*([1-5])\.\s+(.+?)\s*$", message):
        options[match.group(1)] = match.group(2).strip()
    return options


def _iter_graphdb_texts(history: dict[str, Any]):
    for round_data in history.get("rounds", []) if isinstance(history.get("rounds"), list) else []:
        if not isinstance(round_data, dict):
            continue
        for execution in round_data.get("executions", []) if isinstance(round_data.get("executions"), list) else []:
            if not isinstance(execution, dict):
                continue
            result = execution.get("result")
            if isinstance(result, dict) and graphdb_service.has_result(result):
                yield graphdb_service.format_result(result)
                yield json.dumps(graphdb_service.compact_result(result, max_rows=50), ensure_ascii=False)


def _evaluation_supported_answer(history: dict[str, Any]) -> tuple[str | None, str | None]:
    for round_data in reversed(history.get("rounds", []) if isinstance(history.get("rounds"), list) else []):
        if not isinstance(round_data, dict):
            continue
        evaluation = round_data.get("evaluation")
        if not isinstance(evaluation, dict):
            continue
        reason = str(evaluation.get("reason", ""))
        if not reason:
            continue
        lowered = reason.casefold()
        if not any(word in lowered for word in ["matches option", "match option", "corresponds to", "direct match", "matches l?a ch?n", "option"]):
            continue
        match = re.search(r"(?:option|l?a ch?n)\s*([1-5])", reason, flags=re.IGNORECASE)
        if match:
            return match.group(1), reason
    return None, None


def _rescue_graphdb_evidence(message: str, final_text: str, history: dict[str, Any]) -> str:
    if not central.is_multiple_choice_prompt(message):
        return final_text
    data = central.extract_json_object(final_text)
    if not data:
        return final_text
    answer = str(data.get("answer", "")).strip()
    if not re.fullmatch(r"[1-5]", answer) or _parse_bool(data.get("graphDB_evidence")):
        return final_text

    eval_answer, eval_reason = _evaluation_supported_answer(history)
    if eval_answer:
        return json.dumps({
            "answer": eval_answer,
            "graphDB_evidence": True,
            "evidence": [eval_reason[:240]],
        }, ensure_ascii=False)

    option_text = _multiple_choice_options(message).get(answer, "")
    normalized_option = _normalize_match_text(option_text)
    option_tokens = [token for token in re.findall(r"[a-z0-9]+", normalized_option) if len(token) >= 3]
    if option_tokens:
        for evidence_text in _iter_graphdb_texts(history):
            normalized_evidence = _normalize_match_text(evidence_text)
            if all(token in normalized_evidence for token in option_tokens[-3:]):
                return json.dumps({
                    "answer": answer,
                    "graphDB_evidence": True,
                    "evidence": [f"GraphDB evidence mentions option {answer}: {option_text}"],
                }, ensure_ascii=False)
    return final_text


def normalized_final_stream(
    message: str,
    raw_text: str,
    history: dict[str, Any],
    debug_trace: list[dict[str, Any]] | None = None,
) -> Iterator[str]:
    graphdb_result, graphdb_error = answer_formatter.latest_graphdb_evidence(history)
    final_text = central.normalize_answer_evidence_response(message, raw_text, graphdb_result, graphdb_error)
    final_text = _rescue_graphdb_evidence(message, final_text, history)
    logging_service.trace_step(
        "answer_formatter.normalized_output",
        {
            "raw_answer": raw_text,
            "graphdb_evidence_summary": graphdb_service.summarize_result(graphdb_result) if graphdb_result else None,
            "graphdb_error": graphdb_error,
            "final_text": final_text,
        },
        limit=30000,
    )
    if debug_trace is not None and central.is_multiple_choice_prompt(message):
        data = central.extract_json_object(final_text)
        if data:
            data["backend_trace"] = debug_trace
            final_text = json.dumps(data, ensure_ascii=False)
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
                    logging_service.trace_step(
                        "graphdb.query_result_compact",
                        graphdb_service.compact_result(graphdb_result),
                        limit=30000,
                    )
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
    logging_service.trace_step("agent_pipeline.subquery_execution", central.compact_execution_for_llm(execution), limit=5000)

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

def agent_stream(message: str, *, debug: bool = False) -> Iterator[str]:
    trace_token = None
    debug_trace: list[dict[str, Any]] | None = None
    if debug:
        trace_token, debug_trace = logging_service.start_trace()
    try:
        deadline = time.monotonic() + question_timeout_seconds()
        logging_service.trace_text("user.prompt", message)
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
        yield from normalized_final_stream(message, raw_answer, history, debug_trace)
    except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
        logging_service.logger.exception("agent_pipeline.llm_error")
        yield f"Xin loi, hien tai backend khong goi duoc model API ({type(exc).__name__})."
    finally:
        if trace_token is not None:
            logging_service.stop_trace(trace_token)
