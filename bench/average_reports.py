#!/usr/bin/env python3
"""Average benchmark reports from summary JSON and result CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "bench" / "results" / "planning"

QUESTION_TYPES = [
    "comparison",
    "multi-hop",
    "entity",
    "boolean",
    "attribute",
    "schema",
    "superlative",
    "list",
    "counting",
]
OUTPUT_COLUMNS = [
    "Accuracy",
    "Pass",
    "Comparison",
    "multi-hop",
    "entity",
    "boolean",
    "attribute",
    "schema",
    "superlative",
    "list",
    "counting",
    "Evidence count",
    "Time (second)",
    "Avg Time",
]


@dataclass(frozen=True)
class ReportMetrics:
    accuracy: float
    pass_count: float
    by_type: dict[str, float]
    evidence_count: float
    total_time: float
    avg_time: float


def percent(value: float) -> float:
    return value * 100


def read_latency(csv_path: Path) -> tuple[float, float]:
    total_time = 0.0
    row_count = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "latency_seconds" not in (reader.fieldnames or []):
            raise ValueError(f"Missing latency_seconds column: {csv_path}")
        for row in reader:
            value = (row.get("latency_seconds") or "").strip()
            if not value:
                continue
            total_time += float(value)
            row_count += 1

    avg_time = total_time / row_count if row_count else 0.0
    return total_time, avg_time


def load_report(summary_path: Path) -> ReportMetrics:
    timestamp = summary_path.stem.removeprefix("benchmark_summary_")
    csv_path = summary_path.with_name(f"benchmark_results_{timestamp}.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing matching CSV for {summary_path.name}: {csv_path.name}")

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    total_time, avg_time = read_latency(csv_path)
    metrics = summary.get("summary", summary)
    by_type = {
        question_type: float(metrics.get("by_type", {}).get(question_type, {}).get("correct", 0.0))
        for question_type in QUESTION_TYPES
    }

    return ReportMetrics(
        accuracy=float(metrics.get("accuracy", 0.0)),
        pass_count=float(metrics.get("correct", 0.0)),
        by_type=by_type,
        evidence_count=float(metrics.get("db_evidence_count", 0.0)),
        total_time=total_time,
        avg_time=avg_time,
    )


def average_reports(reports: list[ReportMetrics]) -> dict[str, float]:
    if not reports:
        raise ValueError("No reports to average")

    count = len(reports)
    averaged = {
        "Accuracy": sum(report.accuracy for report in reports) / count,
        "Pass": sum(report.pass_count for report in reports) / count,
        "Evidence count": sum(report.evidence_count for report in reports) / count,
        "Time (second)": sum(report.total_time for report in reports) / count,
        "Avg Time": sum(report.avg_time for report in reports) / count,
    }
    for question_type in QUESTION_TYPES:
        column = "Comparison" if question_type == "comparison" else question_type
        averaged[column] = sum(report.by_type[question_type] for report in reports) / count
    return averaged


def format_value(column: str, value: float) -> str:
    if column == "Accuracy":
        return f"{value:.4f}"
    if column in {"Pass", *QUESTION_TYPES, "Comparison"}:
        return f"{value:.1f}"
    if column == "Evidence count":
        return f"{value:.1f}"
    return f"{value:.2f}"


def print_table(values: dict[str, float]) -> None:
    rows = [(column, format_value(column, values[column])) for column in OUTPUT_COLUMNS]
    width = max(len(column) for column, _ in rows)
    for column, value in rows:
        print(f"{column:<{width}}  {value}")


def write_csv(csv_out: Path, values: dict[str, float]) -> None:
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerow({column: format_value(column, values[column]) for column in OUTPUT_COLUMNS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average benchmark summary reports.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--csv-out", type=Path, help="Optional output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_paths = sorted(args.results_dir.glob("benchmark_summary_*.json"))
    if not summary_paths:
        raise SystemExit(f"No benchmark_summary_*.json files found in {args.results_dir}")

    reports = [load_report(path) for path in summary_paths]
    values = average_reports(reports)

    print_table(values)
    if args.csv_out:
        write_csv(args.csv_out, values)
        print(f"Wrote CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
