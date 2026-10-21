# AI Animation Generation

StykAnimate can now generate stickman animation plans from a text prompt using any OpenAI-compatible LLM.

> **No model is bundled.** You bring your own provider — OpenAI, OpenRouter, Groq, Anthropic (via OpenAI compat), Ollama local, LM Studio, etc. Your API key never leaves your device except to your chosen provider (or via the local backend proxy).

## Where to configure

`Settings → AI Animation` (new tab, `lightbulb` icon).

Fields per provider:

| Field | Example | Notes |
| --- | --- | --- |
| Name | My OpenAI | Any label |
| Base URL | `https://api.openai.com/v1` | **Customizable** — any OpenAI-compatible endpoint. No trailing slash needed. |
| Model | `gpt-4o-mini` | **Customizable** — exact model ID your provider serves |
| API Key | `sk-...` | **Optional** — leave empty for Ollama/LM Studio local that needs no auth |

Multiple providers can be stored (up to 20). One is Active; Test button verifies connectivity with a tiny `OK` prompt.

Common Base URLs:

* OpenAI: `https://api.openai.com/v1`
* OpenRouter: `https://openrouter.ai/api/v1`
* Groq: `https://api.groq.com/openai/v1`
* Ollama: `http://localhost:11434/v1` (set `OLLAMA_ORIGINS=*` or run via proxy)
* LM Studio: `http://localhost:1234/v1`

## How it works

Prompt → LLM (via `ai-provider.mjs` `callChatCompletion`) → JSON plan → preview.

### LLM is enough (no vision model)

StykAnimate animations are not pixels but **structured joint moves**. The LLM acts as a director picking moves from a curated motion library (`src/ai-animation.mjs: MOTION_LIBRARY`):

`stand, walk, run, jump, punch, kick, salto, block, hadouken, dance, fall, getup`

System prompt teaches the LLM the exact JSON schema:

```json
{
  "title": "Punch Combo",
  "fps": 24,
  "durationFrames": 48,
  "figures": 1,
  "sequence": [
    {"move": "stand", "frames": 8, "easing": "easeOut"},
    {"move": "punch", "frames": 6, "easing": "easeIn"},
    {"move": "kick", "frames": 14, "easing": "easeInOut"}
  ]
}
```

* `move` must be in library; `frames` 3–30; `easing` validated.
* Duration 12–180 frames, 1–2 figures.
* LLM cannot hallucinate arbitrary geometry — we map library moves to safe poses, so rig constraints (`RIG-CONSTRAINTS.md`) remain respected.

### Backend proxy

Frontend first tries `POST /api/ai/chat` on the local Python codec (`backend/server.py` + `backend/ai_service.py`). This avoids CORS for local Ollama and hides the key from browser navigation history. If the proxy is unavailable (browser-only `npm run serve` without sidecar), it falls back to direct `fetch(baseUrl + "/chat/completions")` — which requires the provider to allow CORS.

Proxy payload:

```json
{
  "baseUrl": "https://api.openai.com/v1",
  "model": "gpt-4o-mini",
  "apiKey": "sk-...",
  "messages": [{"role":"system","content":"..."}, {"role":"user","content":"..."}],
  "temperature": 0.7,
  "maxTokens": 1500
}
```

Response is OpenAI shape (`choices[0].message.content`).

Security: loopback only, `Authorization: Bearer <token>` required when desktop token is set, 30s timeout, 64KB request cap, no logging.

### Local fallback (no LLM)

If no provider is configured or offline, `Generate Local (no LLM)` uses keyword heuristics (`ai-animation.mjs:generatePlanLocal`) — e.g. prompt containing "kick"/"tendang" → `kick` move. Useful for testing without API costs.

## Using the result

v1 returns a **plan preview** (JSON + human summary). Future versions will commit the plan as real keyframes via editor commands (`editor.mjs`).

Manual flow for now:

1. Configure provider in Settings → AI.
2. Enter prompt (e.g. `"2 stickman fight, tendangan salto"`).
3. Click `Generate (LLM)` — see JSON + summary in output area.
4. Copy moves to manually pose, or run Local fallback.

## Extensibility

* Add moves to `MOTION_LIBRARY` in `src/ai-animation.mjs` and update system prompt.
* Add `POST /api/ai/generate` for prompt→document (planned) that writes a full `.piv/.styk` via `native_format.py`.
* Future vision model: swap `callChatCompletion` for a motion diffusion model (MDM, MotionGPT) without changing storage — providers remain user-configurable.

## Privacy

* API keys stored in `localStorage` key `stykanimate.editor.settings.v1` alongside other editor settings; desktop build also persists via workspace preferences IPC. Not included in telemetry (`docs/telemetry.md`).
* No StykAnimate server receives prompts or keys.
* See `src/ai-provider.mjs` and `backend/ai_service.py` for auditable forwarding.
