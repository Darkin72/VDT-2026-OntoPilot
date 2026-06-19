import json
from typing import Any

from app import llm_service, logging_service
from app.agents.central.agent import compact_round_for_llm, extract_json_object, shorten_for_history

def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]

def summarize_round(
    *,
    original_prompt: str,
    accumulated_summary: dict[str, Any] | None,
    round_data: dict[str, Any],
) -> dict[str, Any]:
    prompt = (
        "You are a history summarization agent for a GraphDB/DBpedia QA pipeline.\n"
        "Fold one completed round of raw executions into the accumulated summary.\n\n"
        "Important rules:\n"
        "- Preserve concrete evidence: entity URIs, predicate URIs, object values, dates, counts, labels, and errors.\n"
        "- Preserve failed paths so later agents avoid repeating them.\n"
        "- If evidence is indirect or imperfect, keep it with a qualifier instead of discarding it.\n"
        "- Do not invent facts outside the provided round.\n"
        "- Return only valid JSON with this schema: "
        "{\"summary\":\"...\",\"key_evidence\":[\"...\"],\"failed_paths\":[\"...\"],\"open_questions\":[\"...\"]}.\n\n"
        f"Original prompt:\n{original_prompt}\n\n"
        f"Existing accumulated summary:\n{json.dumps(shorten_for_history(accumulated_summary), ensure_ascii=False, indent=2)}\n\n"
        f"Round to fold:\n{json.dumps(compact_round_for_llm(round_data), ensure_ascii=False, indent=2)}"
    )
    raw_text = llm_service.complete_text(
        [
            llm_service.system_message("You are a summarization agent. Return only summary JSON."),
            llm_service.user_message(prompt),
        ]
    )
    logging_service.verbose_text("summary_agent.raw_summary", raw_text)
    data = extract_json_object(raw_text)
    summary = {
        "summary": str(data.get("summary", "")).strip() if data else "",
        "key_evidence": _normalize_string_list(data.get("key_evidence") if data else None),
        "failed_paths": _normalize_string_list(data.get("failed_paths") if data else None),
        "open_questions": _normalize_string_list(data.get("open_questions") if data else None),
    }
    if not summary["summary"]:
        summary["summary"] = "Summary unavailable; inspect recent raw round evidence."
    logging_service.agent_step("summary_agent.summary", summary, limit=5000)
    return summary
