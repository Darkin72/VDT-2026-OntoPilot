from typing import Any

from app import graphdb_service, llm_service, logging_service
from app.agents.central.agent import history_to_text


def final_answer_messages(message: str, history: dict[str, Any]) -> list[llm_service.ChatMessage]:
    return [
        llm_service.system_message(
            "You are the answer formatting agent for a Vietnamese chatbot. "
            "Use the original user prompt and execution history to produce the final user-facing answer. "
            "For multiple-choice prompts, always return only valid JSON with this schema: "
            "{\"answer\":\"1\",\"graphDB_evidence\":true,\"evidence\":[\"short evidence text\"]}. "
            "The answer value must be the selected option ID as a string. "
            "graphDB_evidence must be true only when the selected option is directly supported by concrete GraphDB/SPARQL rows in the execution history; otherwise it must be false. "
            "Evidence must be a JSON array of short strings. "
            "For multiple-choice factual prompts, GraphDB is authoritative. Do not use external/common knowledge, assumptions, plausibility, or world knowledge to override GraphDB rows. "
            "Treat the execution history as a closed-book evidence set: derive the final answer only from concrete GraphDB rows, aggregate values, literals, URIs, labels, and errors shown in the history. "
            "If GraphDB rows are available in history, compare only those facts with the original choices and choose only the option supported by GraphDB evidence. "
            "Never choose an option because it is generally true, historically true, more plausible, likely, or supported by facts not present in the history. "
            "If GraphDB rows conflict with common knowledge, still choose the option supported by GraphDB. "
            "If an evidence row contains the exact value, date, count, label, URI local name, or ordered item matching one answer choice, choose that matching option even if another option seems more plausible outside GraphDB. "
            "If an evidence row label/URI contains a title plus a qualifier such as manga, comic, film, adaptation, edition, version, or date, and one answer choice names the base title, treat it as supporting that base-title option unless the question explicitly excludes adaptations/versions. Do not reject a GraphDB-supported adaptation merely because it is not an original work. "
            "For boolean prompts, answer Yes/True only when GraphDB history explicitly supports the positive claim; answer No/False only when GraphDB history explicitly contradicts it. Do not infer from real-world knowledge. "
            "For comparison, ordering, and superlative prompts, use only numeric/date values returned by GraphDB history; if such values are missing, do not invent or import values from memory. "
            "Every evidence item should cite a concrete value, entity, relationship, date, count, or literal from the history when possible. "
            "Never cite or rely on 'ki\u1ebfn th\u1ee9c chung', 'd\u1ef1a tr\u00ean suy lu\u1eadn', 'kh\u1ea3 d\u0129', 'ph\u1ed5 bi\u1ebfn', common knowledge, plausibility, likely facts, historical knowledge, or facts absent from the history as evidence. "
            "If the only support for an option is common knowledge or plausibility, that option is unsupported. "
            "If no usable SPARQL evidence exists after the central agent stopped, choose the least-bad option only as a fallback and evidence must contain 'No usable SPARQL evidence; selected as a best-effort fallback.'. "
            "For non-multiple-choice prompts, answer clearly and include GraphDB evidence when available. "
            "Always output clean UTF-8 Vietnamese text with proper diacritics. Never copy corrupted mojibake or provider status text into the final answer. "
            "For Vietnamese count questions such as 'bao nhi\u00eau', 'm\u1ea5y', or 's\u1ed1 l\u01b0\u1ee3ng', if GraphDB returns a count variable such as count, total, or n, use that aggregate value directly as the answer. "
            "Only count distinct concrete entities or literal values from returned rows when no aggregate count value is available; never treat row_count from a limited sample as the total answer. "
            "Do not refuse to answer a count only because the exact relationship label is imperfect; instead answer the count and qualify it when needed, for example 'GraphDB t\u00ecm th\u1ea5y N th\u1ef1c th\u1ec3 li\u00ean quan, nh\u01b0ng quan h\u1ec7 c\u1ee5 th\u1ec3 kh\u00f4ng \u0111\u01b0\u1ee3c chu\u1ea9n h\u00f3a'. "
            "Central-agent next_action reasons are routing notes, not final evidence; prefer the concrete SPARQL rows over a pessimistic routing reason."
        ),
        llm_service.user_message(
            f"Original prompt including any answer choices:\n{message}\n\n"
            f"Execution history:\n{history_to_text(history)}\n\n"
            "Before choosing, verify that the selected option is directly supported by the execution history. "
            "If your rationale mentions common knowledge, plausibility, or facts absent from the history, discard it and choose the option best supported by GraphDB evidence instead. "
            "Return the final answer to the user."
        ),
    ]

def format_final_answer(message: str, history: dict[str, Any]) -> str:
    messages = final_answer_messages(message, history)
    logging_service.trace_step(
        "answer_formatter.input",
        {"messages": messages},
        limit=30000,
    )
    raw_text = llm_service.complete_text(messages)
    logging_service.trace_text("answer_formatter.raw_answer", raw_text, limit=30000)
    return raw_text


def latest_graphdb_evidence(history: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    rounds = history.get("rounds", [])
    if isinstance(rounds, list) and rounds:
        latest_error = None
        for round_data in reversed(rounds):
            if not isinstance(round_data, dict):
                continue
            executions = round_data.get("executions", [])
            if not isinstance(executions, list):
                continue
            for execution in reversed(executions):
                if not isinstance(execution, dict):
                    continue
                error = execution.get("error")
                if error and not latest_error:
                    latest_error = str(error)
                result = execution.get("result")
                if isinstance(result, dict) and graphdb_service.has_result(result):
                    return result, None
        return None, latest_error

    steps = history.get("steps", [])
    if not isinstance(steps, list):
        return None, None

    latest_error = None
    for step in reversed(steps):
        if not isinstance(step, dict) or step.get("type") != "sparql_execution":
            continue
        error = step.get("error")
        if error and not latest_error:
            latest_error = str(error)
        result = step.get("result")
        if isinstance(result, dict) and graphdb_service.has_result(result):
            return result, None
    return None, latest_error
