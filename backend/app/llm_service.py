import json
import os
import time
from collections.abc import Iterator

import requests

from app import logging_service

ChatMessage = dict[str, str]


def system_message(content: str) -> ChatMessage:
    return {"role": "system", "content": content}


def user_message(content: str) -> ChatMessage:
    return {"role": "user", "content": content}


def build_api_url() -> str:
    base_url = os.getenv("CHAT_API_BASE_URL", "").rstrip("/")
    api_path = os.getenv("CHAT_API_PATH", "/v1/chat/completions")
    if not base_url:
        return ""
    return f"{base_url}{api_path if api_path.startswith('/') else '/' + api_path}"


def build_browser_headers() -> dict[str, str]:
    headers = {
        "Accept": "text/event-stream, application/json, text/plain, */*",
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        "DNT": "1",
        "Origin": os.getenv("CHAT_API_ORIGIN", "https://chat.openai.com"),
        "Pragma": "no-cache",
        "Referer": os.getenv("CHAT_API_REFERER", "https://chat.openai.com/"),
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "User-Agent": os.getenv(
            "CHAT_API_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36",
        ),
        "sec-ch-ua": '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
    }

    api_key = os.getenv("CHAT_API_KEY", "")
    if api_key:
        auth_header = os.getenv("CHAT_API_AUTH_HEADER", "Authorization")
        auth_prefix = os.getenv("CHAT_API_AUTH_PREFIX", "Bearer")
        headers[auth_header] = f"{auth_prefix} {api_key}".strip()

    extra_headers = os.getenv("CHAT_API_EXTRA_HEADERS", "")
    if extra_headers:
        headers.update(json.loads(extra_headers))

    return headers

def stream_enabled() -> bool:
    return os.getenv("CHAT_API_STREAM", "true").strip().lower() in {"1", "true", "yes"}

def safe_payload_for_log(payload: dict) -> dict:
    messages = payload.get("messages") or []
    return {
        **{key: value for key, value in payload.items() if key != "messages"},
        "message_count": len(messages) if isinstance(messages, list) else None,
        "message_chars": sum(len(str(message.get("content", ""))) for message in messages)
        if isinstance(messages, list)
        else None,
    }


def build_payload(messages: list[ChatMessage], *, stream: bool = True) -> dict:
    return {
        "model": os.getenv("CHAT_MODEL", "default"),
        "messages": messages,
        "temperature": float(os.getenv("CHAT_TEMPERATURE", "0.2")),
        "stream": stream,
    }


def connect_timeout_seconds() -> float:
    return float(os.getenv("CHAT_API_CONNECT_TIMEOUT_SECONDS", "10"))


def read_timeout_seconds() -> float:
    return float(os.getenv("CHAT_API_READ_TIMEOUT_SECONDS", "120"))


def max_retries() -> int:
    return max(0, int(os.getenv("CHAT_API_MAX_RETRIES", "2")))


def retry_backoff_seconds() -> float:
    return max(0.0, float(os.getenv("CHAT_API_RETRY_BACKOFF_SECONDS", "1")))


def retry_status_codes() -> set[int]:
    raw_codes = os.getenv("CHAT_API_RETRY_STATUS_CODES", "429,500,502,503,504")
    codes: set[int] = set()
    for raw_code in raw_codes.split(","):
        try:
            codes.add(int(raw_code.strip()))
        except ValueError:
            continue
    return codes


def parse_stream_line(line: str) -> str:
    if not line:
        return ""
    if line.startswith(("event:", "id:", "retry:")):
        return ""
    if line.startswith("data:"):
        line = line.removeprefix("data:").strip()
    if line == "[DONE]":
        return ""

    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return line

    choices = data.get("choices") or []
    if choices:
        delta = choices[0].get("delta") or {}
        message = choices[0].get("message") or {}
        return delta.get("content") or message.get("content") or choices[0].get("text") or ""

    return data.get("content") or data.get("text") or ""


def is_retryable_llm_error(exc: requests.RequestException) -> bool:
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in retry_status_codes()
    return False


def stream_messages(messages: list[ChatMessage]) -> Iterator[str]:
    attempts = max_retries() + 1
    yielded_any = False
    api_url = build_api_url()
    request_payload = build_payload(messages, stream=stream_enabled())
    for attempt in range(1, attempts + 1):
        try:
            with requests.post(
                api_url,
                headers=build_browser_headers(),
                json=request_payload,
                stream=stream_enabled(),
                timeout=(connect_timeout_seconds(), read_timeout_seconds()),
            ) as response:
                response.raise_for_status()
                if stream_enabled():
                    for raw_line in response.iter_lines(decode_unicode=True):
                        chunk = parse_stream_line(raw_line or "")
                        if chunk:
                            yielded_any = True
                            yield chunk
                else:
                    chunk = parse_stream_line(response.text)
                    if chunk:
                        yielded_any = True
                        yield chunk
                return
        except requests.RequestException as exc:
            should_retry = attempt < attempts and not yielded_any and is_retryable_llm_error(exc)
            response = getattr(exc, "response", None)
            logging_service.agent_step(
                "llm.request_error",
                {
                    "attempt": attempt,
                    "max_attempts": attempts,
                    "retry": should_retry,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "url": api_url,
                    "status_code": response.status_code if response is not None else None,
                    "response_text": response.text if response is not None else None,
                    "payload": safe_payload_for_log(request_payload),
                },
                limit=3000,
            )
            if not should_retry:
                raise
            time.sleep(retry_backoff_seconds())


def complete_text(messages: list[ChatMessage]) -> str:
    return "".join(stream_messages(messages)).strip()
