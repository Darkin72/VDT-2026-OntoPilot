#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark backend chatbot bằng bộ câu hỏi trắc nghiệm trong Excel.

Script đọc file Ontology/test_questions_v1.0.xlsx, gửi từng câu hỏi kèm các
đáp án lên backend, trích JSON dạng {"answer":"1"}, rồi tính accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import openpyxl
import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XLSX = REPO_ROOT / "Ontology" / "test_questions_v1.0.xlsx"
DEFAULT_BACKEND_URL = "http://localhost:8090/api/chat"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "bench" / "results"


@dataclass
class QuestionCase:
    question_id: str
    question: str
    question_type: str
    resource: str
    answer: int
    options: list[str]


@dataclass
class BenchmarkResult:
    question_id: str
    question_type: str
    question: str
    correct_answer: int
    predicted_answer: int | None
    is_correct: bool
    has_db_evidence: bool
    valid_answer_count: int
    raw_response: str
    response_json: dict[str, object]
    trace_log: list[dict[str, Any]]
    error: str
    latency_seconds: float


def normalize_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_answer(value: object, row_number: int) -> int:
    if value is None or str(value).strip() == "":
        raise ValueError(f"Dòng {row_number}: cột answer bị trống")
    try:
        answer = int(float(str(value).strip()))
    except ValueError as exc:
        raise ValueError(f"Dòng {row_number}: answer không phải số: {value!r}") from exc
    if answer < 1 or answer > 5:
        raise ValueError(f"Dòng {row_number}: answer phải nằm trong khoảng 1-5, nhận {answer}")
    return answer


def load_questions(xlsx_path: Path) -> list[QuestionCase]:
    workbook = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    sheet = workbook.active
    cases: list[QuestionCase] = []

    for row_number, row in enumerate(sheet.iter_rows(values_only=True), start=1):
        if row_number == 1 and normalize_cell(row[0]).lower() in {"number", "question id", "id"}:
            continue

        row = tuple(row) + (None,) * max(0, 10 - len(row))
        question_id = normalize_cell(row[0])
        question = normalize_cell(row[1])
        question_type = normalize_cell(row[2])
        resource = normalize_cell(row[3])

        answer_cell = normalize_cell(row[4])
        option_cells = [normalize_cell(value) for value in row[5:10]]
        if not question:
            print(f"Bỏ qua dòng {row_number}: thiếu câu hỏi", file=sys.stderr)
            continue

        if not answer_cell:
            print(f"Bỏ qua dòng {row_number}: thiếu answer nên không thể chấm", file=sys.stderr)
            continue

        options = [value for value in option_cells if value]
        if not options:
            print(f"Bỏ qua dòng {row_number}: không có đáp án lựa chọn", file=sys.stderr)
            continue

        try:
            answer = parse_answer(row[4], row_number)
        except ValueError as exc:
            print(f"Bỏ qua dòng {row_number}: {exc}", file=sys.stderr)
            continue

        if answer > len(options):
            print(
                f"Bỏ qua dòng {row_number}: answer={answer} nhưng chỉ có {len(options)} đáp án không rỗng",
                file=sys.stderr,
            )
            continue

        cases.append(
            QuestionCase(
                question_id=question_id,
                question=question,
                question_type=question_type,
                resource=resource,
                answer=answer,
                options=options,
            )
        )

    return cases


def build_prompt(case: QuestionCase) -> str:
    options_text = "\n".join(f"{index}. {option}" for index, option in enumerate(case.options, start=1))
    return (
        "Bạn là hệ thống trả lời trắc nghiệm tiếng Việt.\n"
        "Chỉ trả về JSON hợp lệ, không giải thích, không markdown, không thêm ký tự khác.\n"
        "Schema bắt buộc: {\"answer\":\"1\"}.\n"
        f"Giá trị answer hợp lệ là chuỗi từ \"1\" đến \"{len(case.options)}\".\n\n"
        f"Câu hỏi: {case.question}\n\n"
        f"Các đáp án:\n{options_text}\n\n"
        "If SPARQL/GraphDB evidence is available, choose the option supported by that evidence and return {\"answer\":\"1\",\"evidence\":[\"short evidence\"]}; only guess the most likely option when no usable SPARQL evidence exists.\n"
        "JSON:"
    )


def call_backend(backend_url: str, prompt: str, timeout: float, echo_stream: bool = True) -> str:
    with requests.post(
        backend_url,
        json={"message": prompt},
        stream=True,
        timeout=(10, timeout),
    ) as response:
        response.raise_for_status()
        chunks: list[str] = []
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                chunks.append(chunk)
                if echo_stream:
                    print(chunk, end="", flush=True)
        if echo_stream:
            print(flush=True)
        return "".join(chunks).strip()


def extract_response_json(raw_response: str) -> dict[str, object]:
    text = raw_response.strip()
    json_text = text

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        json_text = fenced_match.group(1)
    elif not text.startswith("{"):
        object_match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if object_match:
            json_text = object_match.group(0)

    try:
        data = json.loads(json_text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def extract_answer(raw_response: str, valid_answer_count: int) -> int | None:
    text = raw_response.strip()
    data = extract_response_json(raw_response)
    if data:
        answer_value = str(data.get("answer", "")).strip()
        if re.fullmatch(r"[1-5]", answer_value):
            value = int(answer_value)
            return value if 1 <= value <= valid_answer_count else None

    candidates = [int(value) for value in re.findall(r"(?<!\d)([1-5])(?!\d)", text)]
    for value in candidates:
        if 1 <= value <= valid_answer_count:
            return value
    return None


def has_db_evidence(raw_response: str) -> bool:
    data = extract_response_json(raw_response)
    evidence = data.get("evidence") if data else None
    if not isinstance(evidence, list):
        return False

    evidence_text = "\n".join(str(item) for item in evidence).lower()
    fallback_markers = (
        "no usable sparql evidence",
        "best-effort guess",
        "fallback",
        "no_graphdb_result",
        "graphdb_error",
        "graphdb_timeout",
    )
    if any(marker in evidence_text for marker in fallback_markers):
        return False
    return bool(evidence_text.strip())

def build_trace_step(step: str, detail: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": step,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "detail": detail,
    }

def extract_backend_trace(response_json: dict[str, object]) -> list[dict[str, Any]]:
    trace = response_json.get("trace") or response_json.get("trace_log") or response_json.get("backend_trace")
    if isinstance(trace, list):
        return [item if isinstance(item, dict) else {"message": str(item)} for item in trace]
    return []


def run_single_case(
    index: int,
    case: QuestionCase,
    backend_url: str,
    timeout: float,
    fail_fast: bool,
    echo_stream: bool,
) -> tuple[int, BenchmarkResult]:
    prompt = build_prompt(case)
    trace_log: list[dict[str, Any]] = [
        build_trace_step(
            "benchmark.build_prompt",
            {
                "question_id": case.question_id,
                "question_type": case.question_type,
                "original_question": case.question,
                "resource": case.resource,
                "options": case.options,
                "correct_answer": case.answer,
                "prompt_sent_to_backend": prompt,
            },
        ),
        build_trace_step(
            "backend.request",
            {
                "method": "POST",
                "url": backend_url,
                "json": {"message": prompt},
                "timeout_seconds": timeout,
            },
        ),
    ]
    started_at = time.perf_counter()
    raw_response = ""
    error = ""
    predicted_answer: int | None = None
    response_json: dict[str, object] = {}

    try:
        if echo_stream:
            print(f"    Model stream [{index}]: ", end="", flush=True)
        raw_response = call_backend(backend_url, prompt, timeout, echo_stream=echo_stream)
        trace_log.append(
            build_trace_step(
                "backend.response_stream",
                {
                    "raw_response": raw_response,
                    "response_chars": len(raw_response),
                },
            )
        )
        response_json = extract_response_json(raw_response)
        trace_log.append(
            build_trace_step(
                "backend.response_json_extract",
                {
                    "parsed": bool(response_json),
                    "response_json": response_json,
                    "backend_trace_from_response": extract_backend_trace(response_json),
                },
            )
        )
        predicted_answer = extract_answer(raw_response, len(case.options))
        trace_log.append(
            build_trace_step(
                "benchmark.extract_answer",
                {
                    "predicted_answer": predicted_answer,
                    "valid_answer_count": len(case.options),
                },
            )
        )
    except Exception as exc:  # Ghi lỗi để benchmark tiếp tục chạy được.
        error = str(exc)
        trace_log.append(
            build_trace_step(
                "backend.error",
                {
                    "type": type(exc).__name__,
                    "message": error,
                },
            )
        )
        if fail_fast:
            raise
    finally:
        latency = time.perf_counter() - started_at

    is_correct = predicted_answer == case.answer
    evidence_found = has_db_evidence(raw_response)
    trace_log.append(
        build_trace_step(
            "benchmark.score",
            {
                "correct_answer": case.answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "has_db_evidence": evidence_found,
                "latency_seconds": round(latency, 4),
                "error": error,
            },
        )
    )
    return index, BenchmarkResult(
        question_id=case.question_id,
        question_type=case.question_type,
        question=prompt,
        correct_answer=case.answer,
        predicted_answer=predicted_answer,
        is_correct=is_correct,
        has_db_evidence=evidence_found,
        valid_answer_count=len(case.options),
        raw_response=raw_response,
        response_json=response_json,
        trace_log=trace_log,
        error=error,
        latency_seconds=latency,
    )


def print_case_result(index: int, case: QuestionCase, result: BenchmarkResult) -> None:
    status = "ĐÚNG" if result.is_correct else "SAI"
    predicted_text = "Không trích được" if result.predicted_answer is None else str(result.predicted_answer)
    print(
        f"[{index}] ID={case.question_id} {status} | đúng={case.answer} | "
        f"model={predicted_text} | {result.latency_seconds:.2f}s"
    )
    if result.error:
        print(f"    Lỗi: {result.error}")


def run_benchmark(
    cases: Iterable[QuestionCase],
    backend_url: str,
    timeout: float,
    delay: float,
    fail_fast: bool,
    echo_stream: bool,
    concurrency: int,
) -> list[BenchmarkResult]:
    indexed_cases = list(enumerate(cases, start=1))
    if concurrency <= 1:
        results: list[BenchmarkResult] = []
        progress = tqdm(indexed_cases, total=len(indexed_cases), desc="Benchmark", unit="q")
        for index, case in progress:
            _, result = run_single_case(index, case, backend_url, timeout, fail_fast, echo_stream)
            results.append(result)
            print_case_result(index, case, result)
            if delay > 0:
                time.sleep(delay)
        return results

    if echo_stream:
        print("Tắt stream echo vì đang chạy song song để tránh trộn output.")
        echo_stream = False

    results_by_index: dict[int, BenchmarkResult] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_case = {}
        for index, case in indexed_cases:
            if delay > 0 and index > 1:
                time.sleep(delay)
            future = executor.submit(
                run_single_case,
                index,
                case,
                backend_url,
                timeout,
                fail_fast,
                echo_stream,
            )
            future_to_case[future] = (index, case)

        for future in tqdm(as_completed(future_to_case), total=len(future_to_case), desc="Benchmark", unit="q"):
            index, case = future_to_case[future]
            completed_index, result = future.result()
            results_by_index[completed_index] = result
            print_case_result(completed_index, case, result)

    return [results_by_index[index] for index, _ in indexed_cases]


def summarize(results: list[BenchmarkResult]) -> dict[str, object]:
    total = len(results)
    correct = sum(1 for result in results if result.is_correct)
    errored = sum(1 for result in results if result.error)
    unparsed = sum(1 for result in results if result.predicted_answer is None and not result.error)
    db_evidence_count = sum(1 for result in results if result.has_db_evidence)
    accuracy = correct / total if total else 0.0

    by_type: dict[str, dict[str, object]] = {}
    for result in results:
        key = result.question_type or "unknown"
        stats = by_type.setdefault(key, {"total": 0, "correct": 0, "accuracy": 0.0})
        stats["total"] = int(stats["total"]) + 1
        stats["correct"] = int(stats["correct"]) + int(result.is_correct)

    for stats in by_type.values():
        stats["accuracy"] = int(stats["correct"]) / int(stats["total"]) if stats["total"] else 0.0

    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "errored": errored,
        "unparsed": unparsed,
        "db_evidence_count": db_evidence_count,
        "db_evidence_rate": db_evidence_count / total if total else 0.0,
        "accuracy": accuracy,
        "by_type": by_type,
    }


def write_outputs(results: list[BenchmarkResult], summary: dict[str, object], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"benchmark_results_{timestamp}.csv"
    json_path = output_dir / f"benchmark_summary_{timestamp}.json"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question_id",
                "question_type",
                "question",
                "correct_answer",
                "predicted_answer",
                "is_correct",
                "has_db_evidence",
                "valid_answer_count",
                "latency_seconds",
                "error",
                "raw_response",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "question_id": result.question_id,
                    "question_type": result.question_type,
                    "question": result.question,
                    "correct_answer": result.correct_answer,
                    "predicted_answer": result.predicted_answer or "",
                    "is_correct": result.is_correct,
                    "has_db_evidence": result.has_db_evidence,
                    "valid_answer_count": result.valid_answer_count,
                    "latency_seconds": round(result.latency_seconds, 4),
                    "error": result.error,
                    "raw_response": result.raw_response,
                }
            )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    return csv_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark backend chatbot bằng file Excel câu hỏi.")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX, help="Đường dẫn file .xlsx câu hỏi")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL, help="Endpoint backend /api/chat")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số câu hỏi, 0 là chạy toàn bộ")
    parser.add_argument("--offset", type=int, default=0, help="Bỏ qua N câu hỏi đầu tiên")
    parser.add_argument("--timeout", type=float, default=1200.0, help="Timeout đọc stream cho mỗi câu, tính bằng giây")
    parser.add_argument("--delay", type=float, default=0.0, help="Nghỉ giữa các request submit, tính bằng giây")
    parser.add_argument("--concurrency", type=int, default=1, help="Số câu benchmark chạy song song")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Thư mục ghi kết quả")
    parser.add_argument("--fail-fast", action="store_true", help="Dừng ngay khi request đầu tiên bị lỗi")
    parser.add_argument(
        "--no-stream-echo",
        action="store_true",
        help="Không in stream câu trả lời của model ra màn hình trong lúc benchmark",
    )
    return parser.parse_args()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    args = parse_args()
    cases = load_questions(args.xlsx)
    if args.offset:
        cases = cases[args.offset :]
    if args.limit:
        cases = cases[: args.limit]

    if not cases:
        print("Không có câu hỏi nào để benchmark.", file=sys.stderr)
        return 1

    print(f"Đang benchmark {len(cases)} câu hỏi")
    print(f"Backend: {args.backend_url}")
    print(f"Excel: {args.xlsx}")
    print(f"Concurrency: {max(1, args.concurrency)}")

    results = run_benchmark(
        cases=cases,
        backend_url=args.backend_url,
        timeout=args.timeout,
        delay=args.delay,
        fail_fast=args.fail_fast,
        echo_stream=not args.no_stream_echo,
        concurrency=max(1, args.concurrency),
    )
    summary = summarize(results)
    csv_path, json_path = write_outputs(results, summary, args.output_dir)

    print("\nTổng kết")
    print(f"- Tổng số câu: {summary['total']}")
    print(f"- Đúng: {summary['correct']}")
    print(f"- Sai: {summary['incorrect']}")
    print(f"- Lỗi request: {summary['errored']}")
    print(f"- Không trích được đáp án: {summary['unparsed']}")
    print(f"- DB evidence extracted: {summary['db_evidence_count']} ({summary['db_evidence_rate']:.2%})")
    print(f"- Accuracy: {summary['accuracy']:.2%}")
    print(f"- File CSV: {csv_path}")
    print(f"- File JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
