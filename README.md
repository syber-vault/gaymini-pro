# Gaymini pro

> A compact AI API for chat, vision, images, video, music, code, and text tools.

## What it does

- Chat and multi-turn conversations
- Image generation, editing, and vision
- Video and music generation
- Translation, summaries, and rewriting
- Code assistance and deep research
- Model selection through the API and GoatBot commands

## Run locally

```bash
pip install -r requirements.txt
python3 app.py
```

The server runs on `http://localhost:5000`.

## Quick example

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Explain recursion simply","model":"gemini-3-flash"}'
```

## Documentation

- [API reference](API.md)
- Web docs: `http://localhost:5000/docs`

## Configuration

Add Gemini cookies through environment variables or `gaymini.txt`:

```text
__Secure-1PSID=your-value
__Secure-1PSIDTS=your-value
```

Never commit real cookie values.

## License

Built and maintained by [SIFAT](https://github.com/FX-SIF4T).

## Google session cookies expire — read this first

Image generation, image-to-video and chat all run through your Google
session (Gemini web + Flow). Google sessions expire after hours/days.
When they expire, `/health` reports `"status": "degraded"` and
`"gemini_authenticated": false`, the chat pill turns red ("কুকি মেয়াদোত্তীর্ণ"),
and image/video jobs fail fast with a clear message instead of hanging.

Fix: open the chat, tap the 🔑 button, and re-sync fresh cookies from
`gemini.google.com` (for chat + Nano Banana Pro images) and
`flow.google.com` (for Flow Imagen). Never paste cookies in a chat message
and never commit cookie files — `cookies*.json` and `accounts_vault.json`
are git-ignored for a reason; this repo is public.

## Video: what it really is

"ভিডিও স্টুডিও" makes a **motion video**: it takes a keyframe image
(from Nano Banana Pro / Flow Imagen, or a photo you upload) and animates
it with ffmpeg (slow zoom/pan) into a 5-second 720p mp4. Image-to-video
(upload a photo first) works even when the Google session is expired.
Text-to-video needs a working session to create the first frame. Real
Veo 3.1 generation via Flow is not wired up yet (`FlowClient.generate_video_flow`
only submits the job; the Veo polling loop is not implemented), so the UI
labels the output "মোশন ভিডিও", not "Veo 3.1".

## Use with Claude Code (Anthropic-compatible gateway)

```bash
pip install -r requirements.txt
bash setup.sh          # installs gemini-api, cld-argon, cld-hydroide
cld-argon              # Claude Code on the reasoning model
cld-hydroide           # Claude Code on the fast model
```

`gemini_api_server.py` serves `/v1/messages` (streaming + tool use). Tool use is emulated
through a strict prompt protocol (`anthropic_adapter.py`) because the Gemini web chat has no
native function calling. Multiple tool calls per turn, schema validation/repair, one automatic
corrective retry and Anthropic-spec SSE are supported.

Env vars: `GAYMINI_DIR`, `HOST` (default `127.0.0.1`), `PORT` (5000),
`GAYMINI_API_KEY` (require this key; set it if the server is reachable by others),
`GAYMINI_MAX_PROMPT_CHARS` (450000), `GAYMINI_TOOL_RETRIES` (1),
`GAYMINI_MAX_CONCURRENCY` (2, requests hitting your Google account at once), `GAYMINI_QUEUE_TIMEOUT` (120s), `GAYMINI_THREADS` (16).

Offline tests (no network/cookies): `python3 test_adapter.py && python3 test_server_e2e.py && python3 test_live_stream.py`

Live check with your real cookies (gateway running): `python3 selftest_live.py --model gemini-4-argon`

Tool-calling quality: few-shot examples built from your real tools, full JSON-Schema validation (enum/min/max/nested) with auto-repair, one corrective retry that re-shows the schema, loop breaker, `disable_parallel_tool_use`. Counters at `GET /stats`.

Supported: streaming + non-streaming, multi tool_use per turn, tool_choice, stop_sequences, max_tokens,
Anthropic error types (401 auth / 429 rate_limit / 529 overloaded), request-id header, concurrency limiter, waitress server.
Not possible with Gemini web: real token counts (estimated), images, prompt caching, extended-thinking blocks.
