from typing import Any

from app import graphdb_service, llm_service, logging_service
from app.agents.central.agent import history_to_text


def final_answer_messages(message: str, history: dict[str, Any]) -> list[llm_service.ChatMessage]:
    return [
        llm_service.system_message(
            "You are the answer formatting agent for a Vietnamese chatbot. "
            "Use the original user prompt and execution history to produce the final user-facing answer. "
            "For multiple-choice prompts, always return only valid JSON with this schema: "
            "{\"answer\":\"1\",\"evidence\":[\"short evidence text\"]}. "
            "The answer value must be the selected option ID as a string. "
            "Evidence must be a JSON array of short strings. "
            "If GraphDB rows are available in history, compare those facts with the original choices and choose only the option supported by GraphDB evidence. "
            "Every evidence item should cite a concrete value, entity, relationship, date, count, or literal from the history when possible. "
            "If no usable SPARQL evidence exists after the central agent stopped, choose the most likely answer from general knowledge and say evidence is a best-effort fallback. "
            "For non-multiple-choice prompts, answer clearly and include GraphDB evidence when available. "
            "For Vietnamese count questions such as 'bao nhiêu', 'mấy', or 'số lượng', count distinct concrete entities or literal values from the latest non-empty GraphDB rows when the rows contain plausible answer candidates. "
            "Do not refuse to answer a count only because the exact relationship label is imperfect; instead answer the count and qualify it when needed, for example 'GraphDB tìm thấy N thực thể liên quan, nhưng quan hệ cụ thể không được chuẩn hóa'. "
            "Central-agent next_action reasons are routing notes, not final evidence; prefer the concrete SPARQL rows over a pessimistic routing reason."
        ),
        llm_service.user_message(
            f"Original prompt including any answer choices:\n{message}\n\n"
            f"Execution history:\n{history_to_text(history)}\n\n"
            "Return the final answer to the user."
        ),
    ]


def format_final_answer(message: str, history: dict[str, Any]) -> str:
    raw_text = llm_service.complete_text(final_answer_messages(message, history))
    logging_service.verbose_text("answer_formatter.raw_answer", raw_text)
    return raw_text


def latest_graphdb_evidence(history: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
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
