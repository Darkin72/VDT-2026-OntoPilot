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
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import openpyxl
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XLSX = REPO_ROOT / "Ontology" / "test_questions_v1.0.xlsx"
DEFAULT_BACKEND_URL = "http://localhost:8000/api/chat"
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
    valid_answer_count: int
    raw_response: str
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


def extract_answer(raw_response: str, valid_answer_count: int) -> int | None:
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
        if isinstance(data, dict):
            answer_value = str(data.get("answer", "")).strip()
            if re.fullmatch(r"[1-5]", answer_value):
                value = int(answer_value)
                return value if 1 <= value <= valid_answer_count else None
    except json.JSONDecodeError:
        pass

    candidates = [int(value) for value in re.findall(r"(?<!\d)([1-5])(?!\d)", text)]
    for value in candidates:
        if 1 <= value <= valid_answer_count:
            return value
    return None


def run_benchmark(
    cases: Iterable[QuestionCase],
    backend_url: str,
    timeout: float,
    delay: float,
    fail_fast: bool,
    echo_stream: bool,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    for index, case in enumerate(cases, start=1):
        prompt = build_prompt(case)
        started_at = time.perf_counter()
        raw_response = ""
        error = ""
        predicted_answer: int | None = None

        try:
            if echo_stream:
                print(f"    Model stream: ", end="", flush=True)
            raw_response = call_backend(backend_url, prompt, timeout, echo_stream=echo_stream)
            predicted_answer = extract_answer(raw_response, len(case.options))
        except Exception as exc:  # Ghi lỗi để benchmark tiếp tục chạy được.
            error = str(exc)
            if fail_fast:
                raise
        finally:
            latency = time.perf_counter() - started_at

        is_correct = predicted_answer == case.answer
        results.append(
            BenchmarkResult(
                question_id=case.question_id,
                question_type=case.question_type,
                question=case.question,
                correct_answer=case.answer,
                predicted_answer=predicted_answer,
                is_correct=is_correct,
                valid_answer_count=len(case.options),
                raw_response=raw_response,
                error=error,
                latency_seconds=latency,
            )
        )

        status = "ĐÚNG" if is_correct else "SAI"
        predicted_text = "Không trích được" if predicted_answer is None else str(predicted_answer)
        print(
            f"[{index}] ID={case.question_id} {status} | đúng={case.answer} | "
            f"model={predicted_text} | {latency:.2f}s"
        )
        if error:
            print(f"    Lỗi: {error}")

        if delay > 0:
            time.sleep(delay)

    return results


def summarize(results: list[BenchmarkResult]) -> dict[str, object]:
    total = len(results)
    correct = sum(1 for result in results if result.is_correct)
    errored = sum(1 for result in results if result.error)
    unparsed = sum(1 for result in results if result.predicted_answer is None and not result.error)
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
                    "valid_answer_count": result.valid_answer_count,
                    "latency_seconds": round(result.latency_seconds, 4),
                    "error": result.error,
                    "raw_response": result.raw_response,
                }
            )

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    return csv_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark backend chatbot bằng file Excel câu hỏi.")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX, help="Đường dẫn file .xlsx câu hỏi")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL, help="Endpoint backend /api/chat")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số câu hỏi, 0 là chạy toàn bộ")
    parser.add_argument("--offset", type=int, default=0, help="Bỏ qua N câu hỏi đầu tiên")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout đọc stream cho mỗi câu, tính bằng giây")
    parser.add_argument("--delay", type=float, default=0.0, help="Nghỉ giữa các request, tính bằng giây")
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

    results = run_benchmark(
        cases=cases,
        backend_url=args.backend_url,
        timeout=args.timeout,
        delay=args.delay,
        fail_fast=args.fail_fast,
        echo_stream=not args.no_stream_echo,
    )
    summary = summarize(results)
    csv_path, json_path = write_outputs(results, summary, args.output_dir)

    print("\nTổng kết")
    print(f"- Tổng số câu: {summary['total']}")
    print(f"- Đúng: {summary['correct']}")
    print(f"- Sai: {summary['incorrect']}")
    print(f"- Lỗi request: {summary['errored']}")
    print(f"- Không trích được đáp án: {summary['unparsed']}")
    print(f"- Accuracy: {summary['accuracy']:.2%}")
    print(f"- File CSV: {csv_path}")
    print(f"- File JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())









