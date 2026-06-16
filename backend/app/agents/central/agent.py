import json
import os
import re
from typing import Any

from app import graphdb_service, llm_service, logging_service

OPTIONS_BLOCK_PATTERN = re.compile(
    "(?is)(?:c(?:a|\u00e1)c\\s+)?(?:(?:\u0111|d)(?:a|\u00e1)p\\s*(?:a|\u00e1)n|answer\\s+options?|options?)\\s*:?\\s*\\n?.*$"
)
NUMBERED_OPTION_PATTERN = re.compile(r"(?m)^\s*(?:[1-5]|[A-Ea-e])[\).:-]\s+.+$")


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    object_match = fenced_match or re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not object_match:
        return {}

    json_text = object_match.group(1) if fenced_match else object_match.group(0)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def strip_answer_options(text: str) -> str:
    stripped = OPTIONS_BLOCK_PATTERN.sub("", text).strip()
    stripped = NUMBERED_OPTION_PATTERN.sub("", stripped).strip()
    return stripped or text

def build_history(original_prompt: str) -> dict[str, Any]:
    return {"original_prompt": original_prompt, "steps": []}

def add_history_step(history: dict[str, Any], step_type: str, payload: dict[str, Any]) -> None:
    steps = history.setdefault("steps", [])
    if isinstance(steps, list):
        steps.append({"type": step_type, **payload})

def max_history_chars() -> int:
    return max(1000, int(os.getenv("LLM_HISTORY_MAX_CHARS", "50000")))

def max_history_field_chars() -> int:
    return max(500, int(os.getenv("LLM_HISTORY_FIELD_MAX_CHARS", "8000")))

def shorten_for_history(value: Any) -> Any:
    if isinstance(value, str):
        limit = max_history_field_chars()
        return value if len(value) <= limit else f"{value[:limit]}...<truncated {len(value) - limit} chars>"
    if isinstance(value, list):
        return [shorten_for_history(item) for item in value]
    if isinstance(value, dict):
        return {key: shorten_for_history(item) for key, item in value.items()}
    return value

def compact_graphdb_result(result: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {"summary": graphdb_service.summarize_result(result)}
    try:
        compact["sample"] = json.loads(graphdb_service.format_result(result))
    except (TypeError, json.JSONDecodeError, ValueError):
        compact["sample"] = graphdb_service.format_result(result)
    return compact

def compact_history_for_llm(history: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {"original_prompt": shorten_for_history(history.get("original_prompt", ""))}
    steps = history.get("steps", [])
    compact_steps: list[dict[str, Any]] = []
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            compact_step: dict[str, Any] = {}
            for key, value in step.items():
                if key == "result" and isinstance(value, dict):
                    compact_step["result"] = compact_graphdb_result(value)
                else:
                    compact_step[key] = shorten_for_history(value)
            compact_steps.append(compact_step)
    compact["steps"] = compact_steps
    return compact

def history_to_text(history: dict[str, Any]) -> str:
    compact = compact_history_for_llm(history)
    text = json.dumps(compact, ensure_ascii=False, indent=2)
    limit = max_history_chars()
    if len(text) <= limit:
        return text

    steps = compact.get("steps", [])
    if isinstance(steps, list) and len(steps) > 6:
        compact["omitted_older_steps"] = len(steps) - 6
        compact["steps"] = steps[-6:]
        text = json.dumps(compact, ensure_ascii=False, indent=2)
        if len(text) <= limit:
            return text

    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"

def decide_next_action(
    message: str,
    history: dict[str, Any],
    sparql_attempts: int,
    max_sparql_attempts: int,
) -> dict[str, Any]:
    if sparql_attempts >= max_sparql_attempts:
        decision = {
            "action": "answer",
            "query_description": "",
            "reason": "max_sparql_attempts_reached",
        }
        logging_service.agent_step("central_agent.next_action", decision)
        return decision

    prompt = (
        "You are the central control agent for a Vietnamese chatbot backed by GraphDB/DBpedia.\n"
        "You can inspect the original user prompt and the full execution history.\n"
        "Decide whether the available information is enough to answer, or whether one more neutral GraphDB lookup is needed.\n\n"
        "Important rules:\n"
        "- If history contains enough concrete evidence or enough failed attempts to make progress unlikely, choose action=answer.\n"
        "- For count questions ('bao nhiêu', 'mấy', 'số lượng'), if a broad lookup returns distinct concrete candidate entities or values, choose action=answer and let the answer formatter count them, even when exact relationship labels are not perfectly normalized.\n"
        "- Do not mark evidence unusable solely because a relationship predicate is broad or non-standard; note that the final answer can qualify the count.\n"
        "- If one more lookup is useful, choose action=sparql and write a neutral query_description for the SPARQL coder.\n"
        "- Do not include answer option IDs or answer choices in query_description.\n"
        "- Avoid repeating failed or already-executed lookups from history.\n"
        "- Return only valid JSON with this schema: "
        "{\"action\":\"answer\",\"query_description\":\"\",\"reason\":\"short reason\"}.\n"
        "- action must be either answer or sparql.\n\n"
        f"SPARQL attempts used: {sparql_attempts}/{max_sparql_attempts}\n\n"
        f"Original prompt:\n{message}\n\n"
        f"Execution history:\n{history_to_text(history)}"
    )
    raw_text = llm_service.complete_text(
        [
            llm_service.system_message("You are a central control agent. Return only action JSON."),
            llm_service.user_message(prompt),
        ]
    )
    logging_service.verbose_text("central_agent.raw_next_action_response", raw_text)
    data = extract_json_object(raw_text)
    action = str(data.get("action", "answer")).strip().lower() if data else "answer"
    if action not in {"answer", "sparql"}:
        action = "answer"

    query_description = strip_answer_options(str(data.get("query_description", "")).strip()) if data else ""
    if action == "sparql" and not query_description:
        action = "answer"

    decision = {
        "action": action,
        "query_description": query_description if action == "sparql" else "",
        "reason": str(data.get("reason", "")).strip() if data else "next_action_json_parse_failed",
    }
    logging_service.agent_step("central_agent.next_action", decision)
    return decision


def plan_graphdb_usage(message: str) -> dict[str, Any]:
    lookup_prompt = strip_answer_options(message)
    prompt = (
        "You are the central routing agent for a Vietnamese chatbot backed by GraphDB/DBpedia.\n"
        "Decide whether the user's core question needs GraphDB lookup before answering.\n\n"
        "Important boundary: if the original prompt is multiple-choice, do not pass answer options or option IDs to the SPARQL coder. "
        "The SPARQL coder should only retrieve neutral facts needed to answer the core question. The central agent will compare facts with choices later.\n\n"
        "Use GraphDB for factual questions about entities, relationships, attributes, dates, places, people, organizations, "
        "classes, or multiple-choice questions that likely require stored knowledge.\n"
        "Do not use GraphDB for greetings, chitchat, writing/transformation tasks, pure math, or requests that can be answered "
        "without factual lookup.\n\n"
        "Return only valid JSON with this schema:\n"
        "{\"use_graphdb\":true,\"query_description\":\"neutral fact lookup description without answer options\",\"reason\":\"short reason\"}\n"
        "If GraphDB is not needed, set use_graphdb=false and query_description=\"\".\n\n"
        f"Original prompt:\n{message}\n\n"
        f"Core question without answer options:\n{lookup_prompt}"
    )
    raw_text = llm_service.complete_text(
        [
            llm_service.system_message("You are a central agent. Return only routing JSON."),
            llm_service.user_message(prompt),
        ]
    )
    logging_service.verbose_text("central_agent.raw_routing_response", raw_text)
    data = extract_json_object(raw_text)
    if not data:
        decision = {"use_graphdb": True, "query_description": lookup_prompt, "reason": "routing_json_parse_failed"}
        logging_service.agent_step("central_agent.parsed_decision", decision)
        return decision

    use_graphdb = parse_bool(data.get("use_graphdb"))
    query_description = strip_answer_options(str(data.get("query_description", "")).strip())
    if use_graphdb and not query_description:
        query_description = lookup_prompt

    decision = {
        "use_graphdb": use_graphdb,
        "query_description": query_description,
        "reason": str(data.get("reason", "")).strip(),
    }
    logging_service.agent_step("central_agent.parsed_decision", decision)
    return decision


def is_multiple_choice_prompt(message: str) -> bool:
    return bool(NUMBERED_OPTION_PATTERN.search(message))


def normalize_answer_evidence_response(
    message: str,
    raw_text: str,
    graphdb_result: dict[str, Any] | None = None,
    graphdb_error: str | None = None,
) -> str:
    if not is_multiple_choice_prompt(message):
        return raw_text

    data = extract_json_object(raw_text)
    answer = str(data.get("answer", "")).strip() if data else ""
    if not re.fullmatch(r"[1-5]", answer):
        return raw_text

    raw_evidence = data.get("evidence")
    evidence = [str(item).strip() for item in raw_evidence if str(item).strip()] if isinstance(raw_evidence, list) else []
    if not evidence:
        if graphdb_result and graphdb_service.has_result(graphdb_result):
            evidence = [f"SPARQL evidence: {graphdb_service.format_result(graphdb_result)}"]
        elif graphdb_error:
            evidence = [f"No usable SPARQL evidence ({graphdb_error}); selected as a best-effort guess."]
        else:
            evidence = ["No usable SPARQL evidence; selected as a best-effort guess."]

    return json.dumps({"answer": answer, "evidence": evidence}, ensure_ascii=False)
