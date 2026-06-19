from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(BACKEND_DIR / ".env")

from app.agents.sparql import agent as sparql_agent  # noqa: E402

DEFAULT_USER_PROMPT = "Co bao nhieu nguoi la nha khoa hoc?"
DEFAULT_QUERY_DESCRIPTION = (
    "Perform a SPARQL COUNT query to find the total number of entities where "
    "occupation-related properties (such as dbo:occupation, dbp:occupation, or dbo:field) "
    "are linked to resources whose labels contain 'scientist', 'researcher', or 'scholar'."
)
DEFAULT_CENTRAL_OUTPUT: dict[str, Any] = {
    "subqueries": [
        {
            "id": "q1",
            "description": DEFAULT_QUERY_DESCRIPTION,
            "purpose": "To capture the total count of scientists identified via property-based occupation links, serving as a fallback for entities not explicitly typed as dbo:Scientist.",
            "expected_evidence": "A single integer representing the total count of entities matching the occupation-based string criteria.",
        }
    ],
    "reason": "Mocked central plan for isolated SPARQL-agent testing.",
}


def has_count(sparql: str) -> bool:
    return bool(re.search(r"(?is)\bCOUNT\s*\(", sparql))


def has_row_select(sparql: str) -> bool:
    body = re.sub(r"(?im)^\s*PREFIX\s+[^\n]+\n?", "", sparql).strip()
    return bool(re.match(r"(?is)^SELECT\s+\?", body))


def load_central_output(args: argparse.Namespace) -> dict[str, Any]:
    if args.central_json:
        data = json.loads(args.central_json)
    elif args.central_file:
        data = json.loads(Path(args.central_file).read_text(encoding="utf-8"))
    else:
        data = {
            **DEFAULT_CENTRAL_OUTPUT,
            "subqueries": [
                {
                    **DEFAULT_CENTRAL_OUTPUT["subqueries"][0],
                    "description": args.description,
                }
            ],
        }

    if not isinstance(data, dict):
        raise ValueError("central output must be a JSON object")
    return data


def normalize_subqueries(central_output: dict[str, Any], fallback_description: str) -> list[dict[str, str]]:
    raw_subqueries = central_output.get("subqueries")
    if not isinstance(raw_subqueries, list):
        raw_subqueries = []

    subqueries: list[dict[str, str]] = []
    for index, item in enumerate(raw_subqueries, start=1):
        if not isinstance(item, dict):
            continue
        description = str(item.get("description") or fallback_description).strip()
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

    return subqueries or [
        {
            "id": "q1",
            "description": fallback_description,
            "purpose": "Fallback mocked central subquery.",
            "expected_evidence": "Generated SPARQL only.",
        }
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mock central-agent output and call sparql_agent.generate_sparql in isolation."
    )
    parser.add_argument("--prompt", default=DEFAULT_USER_PROMPT, help="Original user prompt")
    parser.add_argument(
        "--description",
        default=DEFAULT_QUERY_DESCRIPTION,
        help="Fallback central query description when no central JSON/file is provided",
    )
    parser.add_argument(
        "--central-json",
        default="",
        help="Inline JSON that mimics central.plan_subqueries output",
    )
    parser.add_argument(
        "--central-file",
        default="",
        help="Path to a JSON file that mimics central.plan_subqueries output",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Run the same isolated call N times")
    return parser.parse_args()


def build_history(prompt: str, central_output: dict[str, Any], subqueries: list[dict[str, str]]) -> dict[str, Any]:
    plan = {**central_output, "subqueries": subqueries}
    return {
        "original_prompt": prompt,
        "steps": [{"type": "central_subquery_plan", **plan}],
        "rounds": [
            {
                "round": 1,
                "focus": subqueries[0]["description"] if subqueries else "",
                "plan": plan,
                "executions": [],
                "evaluation": None,
            }
        ],
        "accumulated_summary": None,
        "summarized_round_count": 0,
    }


def main() -> int:
    args = parse_args()
    central_output = load_central_output(args)
    subqueries = normalize_subqueries(central_output, args.description)

    for run_index in range(1, max(1, args.repeat) + 1):
        history = build_history(args.prompt, central_output, subqueries)
        round_data = history["rounds"][0]

        print(f"\n=== Isolated SPARQL run {run_index} ===")
        print(f"Prompt: {args.prompt}")
        print("Mock central output:")
        print(json.dumps(round_data["plan"], ensure_ascii=False, indent=2))

        for subquery_index, subquery in enumerate(subqueries, start=1):
            sparql = sparql_agent.generate_sparql(
                user_prompt=args.prompt,
                query_description=subquery["description"],
                history=history,
                subquery_id=subquery["id"],
                round_context={
                    "round": 1,
                    "subquery": subquery,
                    "current_round_executions": round_data["executions"],
                },
            )

            execution = {
                "round": 1,
                "subquery_index": subquery_index,
                "subquery_id": subquery["id"],
                "query_description": subquery["description"],
                "purpose": subquery.get("purpose", ""),
                "expected_evidence": subquery.get("expected_evidence", ""),
                "sparql": sparql,
                "result_summary": "<not executed; isolated SPARQL generation only>",
                "error": None if sparql else "SPARQL_GENERATION_EMPTY_OR_REJECTED",
            }
            round_data["executions"].append(execution)

            print(f"\n--- Subquery {subquery_index}: {subquery['id']} ---")
            print(f"Description: {subquery['description']}")
            print(f"Purpose: {subquery.get('purpose', '')}")
            print(f"Expected evidence: {subquery.get('expected_evidence', '')}")
            print(f"Has COUNT: {has_count(sparql)}")
            print(f"Looks like row SELECT: {has_row_select(sparql)}")
            print("SPARQL:")
            print(sparql or "<empty>")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
