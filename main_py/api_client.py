import json
import requests
from typing import Generator

API_BASE = "http://localhost:2026"


def stream_chat(query: str, history: list = None) -> Generator[dict, None, None]:
    payload = {"query": query, "history": history or []}
    with requests.post(
        f"{API_BASE}/chat/stream",
        json=payload,
        stream=True,
        timeout=300,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue


def ask_chat(query: str, history: list = None) -> dict:
    payload = {"query": query, "history": history or []}
    resp = requests.post(
        f"{API_BASE}/chat/ask",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def compute_metrics(question: str, answer: str, context: str, sources: list = None) -> dict:
    payload = {
        "question": question,
        "answer"  : answer,
        "context" : context,
        "sources" : sources or [],
    }
    try:
        resp = requests.post(
            f"{API_BASE}/metrics",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def is_server_ready() -> bool:
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        return resp.json().get("service_ready", False)
    except Exception:
        return False
