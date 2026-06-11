# Benchmark backend

Script `benchmark_backend.py` đọc bộ câu hỏi trong `Ontology/test_questions_v1.0.xlsx`, gọi backend `/api/chat`, stream câu trả lời của model ra màn hình theo từng câu, trích JSON model trả về theo dạng `{ "answer": "1" }`, rồi tự động chấm accuracy.

## Cài dependency

```bash
pip install -r bench/requirements.txt
```

## Chạy benchmark

```bash
python bench/benchmark_backend.py --backend-url http://localhost:8000/api/chat
```

Chạy thử một vài câu:

```bash
python bench/benchmark_backend.py --limit 10
```

Mặc định script in stream câu trả lời của model ra màn hình. Nếu chỉ muốn chấm điểm và ghi file kết quả, tắt phần in stream:

```bash
python bench/benchmark_backend.py --limit 10 --no-stream-echo
```

Kết quả được ghi vào `bench/results/` gồm:

- `benchmark_results_*.csv`: chi tiết từng câu, đáp án đúng, đáp án model, raw response, lỗi nếu có.
- `benchmark_summary_*.json`: tổng kết accuracy và accuracy theo `question_type`.


Prompt benchmark yêu cầu model chỉ trả về JSON hợp lệ, ví dụ:

```json
{"answer":"1"}
```

Script chỉ gửi câu hỏi và các đáp án lựa chọn lên backend. Cột `resource` và `answer` không được gửi vào prompt để tránh lộ đáp án. Các dòng không đủ dữ liệu để chấm, ví dụ có `id` nhưng thiếu câu hỏi, thiếu `answer` hoặc thiếu đáp án lựa chọn, sẽ được bỏ qua và in cảnh báo.





