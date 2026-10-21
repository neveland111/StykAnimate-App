"""
AI proxy service for StykAnimate desktop codec.
Provides OpenAI-compatible forwarding so frontend can use any provider
without CORS issues and without exposing apiKey to browser history.

Endpoints (handled in server.py):
  POST /api/ai/chat   {baseUrl, model, apiKey, messages, temperature, maxTokens}
  POST /api/ai/generate {prompt, providerId?} -> uses stored provider? (future)

This module contains the forwarding logic; server.py wires HTTP.

Security: loopback only, requires X-Styk-Job-Id + token like other /api routes.
No persistence; apiKey is forwarded ephemerally, not logged.
Timeout: 30s. Max message payload: 64KB.

For browser (npm run serve) without token, backend still enforces loopback + origin check.
"""
from __future__ import annotations
import json
import urllib.request
import urllib.error
from typing import Any

MAX_BODY = 64 * 1024
TIMEOUT_S = 30

def _validate_base_url(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("baseUrl required")
    url = base_url.strip().rstrip("/")
    if not (url.startswith("https://") or url.startswith("http://")):
        raise ValueError("baseUrl must start with https:// or http://")
    # Allow only chat/completions style; normalize
    # Strip trailing /chat/completions if user pasted full endpoint
    if url.endswith("/chat/completions"):
        url = url[:-len("/chat/completions")]
    return url

def _validate_model(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model required")
    m = model.strip()
    if len(m) > 120:
        raise ValueError("model name too long")
    return m

def build_openai_request(base_url: str, model: str, api_key: str, messages: Any, temperature: float = 0.7, max_tokens: int = 1500) -> tuple[str, bytes, dict]:
    base = _validate_base_url(base_url)
    model = _validate_model(model)
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be non-empty list")
    if len(messages) > 30:
        raise ValueError("too many messages")
    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            raise ValueError("each message must have role and content")
        if msg["role"] not in ("system", "user", "assistant"):
            raise ValueError("invalid role")
        if not isinstance(msg["content"], str) or len(msg["content"]) > 8000:
            raise ValueError("content too long")
    temp = float(temperature) if isinstance(temperature, (int, float)) else 0.7
    temp = max(0.0, min(2.0, temp))
    mtok = int(max_tokens) if isinstance(max_tokens, int) else 1500
    mtok = max(1, min(4000, mtok))

    url = base + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temp,
        "max_tokens": mtok,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key and isinstance(api_key, str) and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    return url, body, headers

def forward_chat(base_url: str, model: str, api_key: str, messages: Any, temperature: float = 0.7, max_tokens: int = 1500) -> dict:
    """
    Synchronously forward to LLM provider and return parsed JSON.
    Raises urllib.error.HTTPError on failure.
    """
    url, body, headers = build_openai_request(base_url, model, api_key, messages, temperature, max_tokens)
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            raw = resp.read()
            if len(raw) > 512 * 1024:
                raise ValueError("LLM response too large")
            # Ensure valid JSON
            data = json.loads(raw.decode("utf-8"))
            return data
    except urllib.error.HTTPError as e:
        # Re-raise with body for server.py to format
        try:
            err_body = e.read().decode("utf-8")[:2000]
        except Exception:
            err_body = str(e)
        raise RuntimeError(f"LLM upstream {e.code}: {err_body}") from e

# For unit testing without network
def mock_chat_response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}
