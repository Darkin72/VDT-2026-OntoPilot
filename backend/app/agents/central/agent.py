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
    text = re.sub(r"(?im)^\s*:??\s*OPENROUTER PROCESSING\s*$", "", text).strip()

    def parse_candidate(candidate: str) -> dict[str, Any]:
        candidate = candidate.strip()
        candidates = [candidate]
        if r'\"' in candidate:
            candidates.append(candidate.replace(r'\"', '"'))
        if candidate.startswith('"') and candidate.endswith('"'):
            try:
                decoded = json.loads(candidate)
                if isinstance(decoded, str):
                    candidates.append(decoded)
            except json.JSONDecodeError:
                pass
        for item in candidates:
            try:
                data = json.loads(item)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
            if isinstance(data, str):
                try:
                    nested = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(nested, dict):
                    return nested
        return {}

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
    return parse_candidate(json_text)


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
    return {
        "original_prompt": original_prompt,
        "steps": [],
        "rounds": [],
        "accumulated_summary": None,
        "summarized_round_count": 0,
    }

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
    compact["sample"] = graphdb_service.compact_result(result)
    return compact

def compact_execution_for_llm(execution: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in execution.items():
        if key == "result" and isinstance(value, dict):
            compact[key] = compact_graphdb_result(value)
        else:
            compact[key] = shorten_for_history(value)
    return compact

def compact_round_for_llm(round_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "round": round_data.get("round"),
        "focus": shorten_for_history(round_data.get("focus", "")),
        "plan": shorten_for_history(round_data.get("plan")),
        "executions": [compact_execution_for_llm(execution) for execution in round_data.get("executions", []) if isinstance(execution, dict)],
        "evaluation": shorten_for_history(round_data.get("evaluation")),
    }

def raw_round_window() -> int:
    return max(1, int(os.getenv("GRAPHDB_RAW_ROUND_WINDOW", "1")))

def execution_context_for_llm(history: dict[str, Any], *, include_current_round: bool = True) -> dict[str, Any]:
    rounds = [round_data for round_data in history.get("rounds", []) if isinstance(round_data, dict)]
    summarized_count = max(0, int(history.get("summarized_round_count") or 0))
    recent_rounds = rounds[summarized_count:] if rounds else []
    if not recent_rounds and rounds:
        recent_rounds = rounds[-raw_round_window():]
    if not include_current_round and recent_rounds:
        recent_rounds = recent_rounds[:-1]
    return {
        "original_prompt": shorten_for_history(history.get("original_prompt", "")),
        "accumulated_summary": shorten_for_history(history.get("accumulated_summary")),
        "raw_rounds_not_yet_summarized": [compact_round_for_llm(round_data) for round_data in recent_rounds],
    }

def compact_history_for_llm(history: dict[str, Any]) -> dict[str, Any]:
    if isinstance(history.get("rounds"), list) and history.get("rounds"):
        return execution_context_for_llm(history)

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

def max_agent_rounds() -> int:
    return max(1, int(os.getenv("MAX_AGENT_ROUNDS", "5")))

def max_subqueries_per_round() -> int:
    return max(1, min(4, int(os.getenv("MAX_SUBQUERIES_PER_ROUND", "4"))))

def _fallback_subqueries(query_description: str) -> list[dict[str, str]]:
    description = strip_answer_options(query_description).strip() or "Look up neutral facts needed to answer the user question."
    return [
        {
            "id": "q1",
            "description": description,
            "purpose": "Fallback lookup when central planning did not return usable subqueries.",
            "expected_evidence": "Concrete GraphDB rows relevant to the original question.",
        }
    ]

def _normalize_subqueries(raw_subqueries: Any, fallback_description: str) -> list[dict[str, str]]:
    if not isinstance(raw_subqueries, list):
        return _fallback_subqueries(fallback_description)

    subqueries: list[dict[str, str]] = []
    for index, item in enumerate(raw_subqueries, start=1):
        if not isinstance(item, dict):
            continue
        description = strip_answer_options(str(item.get("description", "")).strip())
        if not description:
            continue
        subqueries.append(
            {
                "id": str(item.get("id") or f"q{index}").strip() or f"q{index}",
                "description": description,
                "purpose": str(item.get("purpose", "")).strip(),
                "expected_evidence": str(item.get("expected_evidence", "")).strip(),
            }
        )
        if len(subqueries) >= max_subqueries_per_round():
            break

    return subqueries or _fallback_subqueries(fallback_description)

def plan_subqueries(
    message: str,
    history: dict[str, Any],
    *,
    round_index: int,
    focus: str,
) -> dict[str, Any]:
    fallback_description = focus or strip_answer_options(message)
    prompt = (
        "You are the central planning agent for a Vietnamese chatbot backed by GraphDB/DBpedia.\n"
        "Create neutral subqueries for the next GraphDB round. Each subquery will be assigned to one SPARQL agent.\n\n"
        "Important rules:\n"
        f"- Return 1 to {max_subqueries_per_round()} subqueries.\n"
        "- Do not include answer option IDs or answer choices.\n"
        "- Write each subquery description as domain lookup text only, because it is used for ontology term retrieval. Do not put operational words in description such as SPARQL, query, COUNT, COUNT(DISTINCT), total number, perform, execute, aggregate, return, or single integer.\n"
        "- Put execution requirements such as aggregate count, COUNT(DISTINCT ?entity), single integer, samples, or verification only in purpose and expected_evidence, not in description.\n"
        "- Prefer a mix of focused lookup and discovery when predicate choice is uncertain.\n"
        "- Discovery subqueries should inspect useful non-type predicates and avoid rdf:type/label noise.\n"
        "- For count questions ('bao nhiêu', 'mấy', 'số lượng', 'how many', 'count'), the first subquery must require a numeric aggregate count, not a list of entities. Keep description semantic, e.g. 'Entities typed as dbo:Scientist or subclasses' or 'People with occupation labels scientist researcher scholar'. Put 'Use COUNT(DISTINCT ?entity); expected evidence is one integer' in expected_evidence.\n"
        "- If a count question may match more than 100 entities, expected_evidence must explicitly instruct the SPARQL coder to use COUNT(DISTINCT ...). Do not ask for listing/sample rows first, because LIMIT 100/200 rows is not the total answer.\n"
        "- For count questions over broad classes such as people, places, organizations, works, species, or events, assume the match set may exceed 100 and plan an aggregate count subquery first.\n"
        "- For count questions, do not write first-round descriptions such as 'Find all entities', 'list all entities', or expected_evidence='list of URIs' unless it is a later verification/sample subquery.\n"
        "- If exact death/birth/location predicates may be missing, include fallback relationships such as resting place, burial place, subdivision, location hierarchy, aliases, or broad non-type predicate inspection.\n"
        "- Make each subquery concrete enough for a SPARQL coder but do not write SPARQL.\n"
        "- Return only valid JSON with this schema: "
        "{\"subqueries\":[{\"id\":\"q1\",\"description\":\"...\",\"purpose\":\"...\",\"expected_evidence\":\"...\"}],\"reason\":\"short reason\"}.\n\n"
        f"Round: {round_index}/{max_agent_rounds()}\n"
        f"Current focus:\n{fallback_description}\n\n"
        f"Original prompt:\n{message}\n\n"
        f"Available execution context:\n{json.dumps(execution_context_for_llm(history, include_current_round=False), ensure_ascii=False, indent=2)}"
    )
    raw_text = llm_service.complete_text(
        [
            llm_service.system_message("You are a central query planner. Return only planning JSON."),
            llm_service.user_message(prompt),
        ]
    )
    logging_service.verbose_text("central_agent.raw_subquery_plan", raw_text)
    data = extract_json_object(raw_text)
    plan = {
        "subqueries": _normalize_subqueries(
            data.get("subqueries") if data else None,
            fallback_description,
        ),
        "reason": str(data.get("reason", "")).strip() if data else "subquery_plan_json_parse_failed",
    }
    logging_service.agent_step("central_agent.subquery_plan", plan, limit=5000)
    return plan

def evaluate_round(
    message: str,
    history: dict[str, Any],
    *,
    round_index: int,
    max_rounds: int,
) -> dict[str, Any]:
    if round_index >= max_rounds:
        decision = {
            "action": "answer",
            "reason": "max_agent_rounds_reached",
            "next_focus": "",
        }
        logging_service.agent_step("central_agent.round_evaluation", decision)
        return decision

    prompt = (
        "You are the central evaluation agent for a Vietnamese chatbot backed by GraphDB/DBpedia.\n"
        "Decide whether the accumulated evidence is enough to answer, or whether another GraphDB round is needed.\n\n"
        "Important rules:\n"
        "- Choose answer when recent rows or accumulated summary contain concrete evidence for the original question.\n"
        "- Do not reject evidence only because a predicate is broad, imperfect, or a fallback such as restingPlace for a death-location question; the final answer can qualify it.\n"
        "- Choose continue only when a specific missing piece remains and write next_focus for the next planner.\n"
        "- Avoid repeating failed paths listed in the summary or recent executions.\n"
        "- Return only valid JSON with this schema: {\"action\":\"answer\",\"reason\":\"short reason\",\"next_focus\":\"\"}.\n"
        "- action must be either answer or continue.\n\n"
        f"Round: {round_index}/{max_rounds}\n\n"
        f"Original prompt:\n{message}\n\n"
        f"Execution context:\n{json.dumps(execution_context_for_llm(history), ensure_ascii=False, indent=2)}"
    )
    raw_text = llm_service.complete_text(
        [
            llm_service.system_message("You are a central evaluator. Return only decision JSON."),
            llm_service.user_message(prompt),
        ]
    )
    logging_service.verbose_text("central_agent.raw_round_evaluation", raw_text)
    data = extract_json_object(raw_text)
    action = str(data.get("action", "answer")).strip().lower() if data else "answer"
    if action not in {"answer", "continue"}:
        action = "answer"

    decision = {
        "action": action,
        "reason": str(data.get("reason", "")).strip() if data else "round_evaluation_json_parse_failed",
        "next_focus": strip_answer_options(str(data.get("next_focus", "")).strip()) if action == "continue" else "",
    }
    if action == "continue" and not decision["next_focus"]:
        decision["next_focus"] = "Run a different neutral lookup that targets the remaining missing evidence."
    logging_service.agent_step("central_agent.round_evaluation", decision)
    return decision

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
        "- For count questions ('bao nhiêu', 'mấy', 'số lượng'), prefer an aggregate SPARQL result such as COUNT(DISTINCT ?entity) AS ?count. Do not treat row_count from a limited sample as the total.\n"
        "- For count questions, continue if the history only contains limited candidate rows and no aggregate count, unless enough failed attempts make progress unlikely.\n"
        "- Do not mark evidence unusable solely because a relationship predicate is broad or non-standard; note that the final answer can qualify the count.\n"
        "- If one more lookup is useful, choose action=sparql and write a neutral query_description for the SPARQL coder.\n"
        "- query_description is also used for ontology term retrieval, so keep it semantic and domain-focused. Do not include operational words such as SPARQL, query, COUNT, COUNT(DISTINCT), total number, perform, execute, aggregate, return, or single integer.\n"
        "- For count follow-up lookups, describe the entity set semantically in query_description and rely on the count-question rules above to require aggregate evidence.\n"
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
