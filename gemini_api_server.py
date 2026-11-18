#!/usr/bin/env python3
"""
Unified Gemini 4 Argon & Hydrogen REST API Server
- OpenAI Compatible: /v1/chat/completions (Streaming SSE + JSON)
- Anthropic Compatible: /v1/messages (Streaming SSE + JSON, with Full Tool Calling Support)
- Models: gemini-4-argon (Frontier Reasoning), gemini-4-hydrogen (Fast Coding & Execution)
- Powered by Google Gemini Pro / Advanced Web Session with Auto-Cookie Rotation
"""

import sys
import os
from pathlib import Path

# Project dir: override with GAYMINI_DIR, otherwise the folder this file lives in
BASE_DIR = Path(os.environ.get("GAYMINI_DIR") or Path(__file__).resolve().parent)

# Ensure local gemini_webapi / anthropic_adapter can be imported
sys.path.insert(0, str(BASE_DIR / "sxysifat" / "src"))
sys.path.insert(0, str(BASE_DIR))

import asyncio
import json
import logging
import queue
import re
import threading
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from curl_cffi.requests import AsyncSession
from urllib.parse import quote, urlparse
from flask import Flask, Response, jsonify, redirect, request, send_file, send_from_directory, stream_with_context
import anthropic_adapter as adapter
from gemini_webapi import GeminiClient
from gemini_webapi.constants import Model
from gemini_webapi.utils.rotate_1psidts import _get_cookie_cache_dir, rotate_1psidts
import flow_engine

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gemini_api_server")

COOKIES_TXT_PATH = str(BASE_DIR / "gaymini.txt")
COOKIES_JSON_PATH = str(BASE_DIR / "cookies.json")

_ASYNC_LOOP: Optional[asyncio.AbstractEventLoop] = None
_LOOP_THREAD: Optional[threading.Thread] = None
_INIT_LOCK = threading.Lock()
_ACTIVE_CLIENT: Optional[GeminiClient] = None
_LAST_ROTATION_TIME: float = 0.0
_LAST_ACCOUNT_STATUS = 1016  # default: UNAUTHENTICATED until a client init proves otherwise (1000 = AVAILABLE). Never default to 1000 — /health must not claim "healthy" before auth is verified.

TOOL_CALL_REGEX = re.compile(
    r"```(?:tool_call|json_tool|tool)?\s*(\{[\s\S]*?\})\s*```|<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>",
    re.MULTILINE | re.DOTALL,
)


def normalize_tool_call(name: str, args: dict) -> tuple[str, dict]:
    """Map tool signatures to standard names for Claude Code."""
    lower = (name or "").lower().strip()
    if lower in ("filewrite", "writefile", "write_file", "write", "createfile", "create"):
        path = args.get("file_path") or args.get("path") or args.get("filename") or ""
        content = args.get("content") or args.get("text") or args.get("data") or ""
        return "Write", {"file_path": path, "content": content}
    if lower in ("fileedit", "editfile", "edit_file", "edit", "strreplace", "modify"):
        path = args.get("file_path") or args.get("path") or args.get("filename") or ""
        old_str = args.get("old_string") or args.get("old_str") or args.get("target") or ""
        new_str = args.get("new_string") or args.get("new_str") or args.get("replacement") or ""
        return "Edit", {"file_path": path, "old_string": old_str, "new_string": new_str}
    if lower in ("fileread", "readfile", "read_file", "read", "view", "cat"):
        path = args.get("file_path") or args.get("path") or args.get("filename") or ""
        res_args = {"file_path": path}
        if "offset" in args:
            res_args["offset"] = args["offset"]
        if "limit" in args:
            res_args["limit"] = args["limit"]
        return "Read", res_args
    if lower in ("bash", "shell", "run", "terminal", "command", "sh"):
        cmd = args.get("command") or args.get("cmd") or ""
        return "Bash", {"command": cmd}
    return name, args


def _parse_tool_call_from_text(text: str) -> tuple[str, Optional[Dict[str, Any]]]:
    """Extract tool call JSON from finished text."""
    m = TOOL_CALL_REGEX.search(text)
    if not m:
        return text, None

    raw_json = (m.group(1) or m.group(2) or "").strip()
    try:
        parsed = json.loads(raw_json)
        if isinstance(parsed, dict) and "name" in parsed:
            clean_text = text[:m.start()].strip()
            raw_name = parsed["name"]
            raw_args = parsed.get("arguments", parsed.get("parameters", {}))
            t_name, t_args = normalize_tool_call(raw_name, raw_args)
            return clean_text, {"name": t_name, "arguments": t_args}
    except Exception:
        pass

    return text, None


def _format_tools_prompt(tools: List[Dict[str, Any]]) -> str:
    if not tools:
        return ""
    lines = [
        "\n# TOOL CALLING SYSTEM ENABLED",
        "You have tool execution capabilities enabled. When you need to create a file, read files, or run commands, output a fenced code block using ```tool_call:",
        "```tool_call",
        "{",
        '  "name": "<tool_name>",',
        '  "arguments": { <parameters> }',
        "}",
        "```",
        "\nAvailable Tools:",
        "- `write_file(file_path: str, content: str)`: Creates or writes a file to disk.",
        "- `run_command(command: str)`: Runs a command and returns its output.",
        "- `read_file(file_path: str)`: Reads a file from disk.",
        "- `edit_file(file_path: str, old_string: str, new_string: str)`: Replaces text in a file.",
    ]
    for t in tools:
        name = t.get("name", "")
        if name in ("Bash", "Write", "Read", "Edit"):
            continue
        desc = t.get("description", "")
        schema = t.get("input_schema", {})
        props = schema.get("properties", {})
        param_desc = ", ".join([f"{p}: {info.get('type', 'any')}" for p, info in props.items()]) if props else ""
        lines.append(f"- `{name}({param_desc})`: {desc}")

    lines.append(
        "\nTo perform actions, ALWAYS output the ```tool_call block. Never refuse to run commands or write files."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Async Loop Worker
# ---------------------------------------------------------------------------
def _ensure_event_loop() -> asyncio.AbstractEventLoop:
    global _ASYNC_LOOP, _LOOP_THREAD
    with _INIT_LOCK:
        if _ASYNC_LOOP is None or not _ASYNC_LOOP.is_running():
            _ASYNC_LOOP = asyncio.new_event_loop()

            def _run():
                asyncio.set_event_loop(_ASYNC_LOOP)
                _ASYNC_LOOP.run_forever()

            _LOOP_THREAD = threading.Thread(target=_run, daemon=True, name="GeminiAsyncLoop")
            _LOOP_THREAD.start()
    return _ASYNC_LOOP


def _run_coroutine_sync(coro):
    loop = _ensure_event_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=900)


# ---------------------------------------------------------------------------
# Cookies & Gemini Client Initialization
# ---------------------------------------------------------------------------
def load_cookies() -> Dict[str, str]:
    # 0. Check Environment Variable (for Cloud / Render deployment)
    env_c = os.environ.get("GEMINI_COOKIES") or os.environ.get("COOKIES_JSON")
    if env_c:
        try:
            parsed = json.loads(env_c)
            if isinstance(parsed, list):
                return {item["name"]: item["value"] for item in parsed if isinstance(item, dict) and "name" in item and "value" in item}
            elif isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except Exception:
            pass

    # 1. Prioritize dedicated cookies_gemini.json
    gemini_json = BASE_DIR / "cookies_gemini.json"
    if gemini_json.is_file():
        try:
            with open(gemini_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and ("__Secure-1PSID" in data or "SID" in data):
                    return {str(k): str(v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"Failed to read cookies_gemini.json: {e}")

    cookies: Dict[str, str] = {}
    if os.path.exists(COOKIES_TXT_PATH):
        try:
            with open(COOKIES_TXT_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        cookies[k.strip()] = v.strip()
            if "__Secure-1PSID" in cookies:
                return cookies
        except Exception as e:
            logger.warning(f"Failed to read cookies from txt: {e}")

    if os.path.exists(COOKIES_JSON_PATH):
        try:
            with open(COOKIES_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "name" in item and "value" in item:
                            cookies[item["name"]] = item["value"]
                elif isinstance(data, dict):
                    cookies.update(data)
            return cookies
        except Exception as e:
            logger.warning(f"Failed to read cookies from json: {e}")

    return cookies


async def get_gemini_client() -> GeminiClient:
    global _ACTIVE_CLIENT, _LAST_ROTATION_TIME, _LAST_ACCOUNT_STATUS

    if _ACTIVE_CLIENT is not None:
        now = time.time()
        if now - _LAST_ROTATION_TIME > 600:  # 10 minutes
            try:
                if getattr(_ACTIVE_CLIENT, "client", None):
                    new_ts = await rotate_1psidts(_ACTIVE_CLIENT.client, verbose=True)
                    if new_ts:
                        cookies = load_cookies()
                        cookies["__Secure-1PSIDTS"] = new_ts
                        flow_engine.save_cookies(cookies, service="gemini")
                        _LAST_ROTATION_TIME = now
                        logger.info("Successfully refreshed session cookies via rotate_1psidts")
            except Exception as e:
                logger.warning(f"Cookie rotation skipped: {e}")
        return _ACTIVE_CLIENT

    cookies = load_cookies()
    psid = cookies.get("__Secure-1PSID")
    if not psid:
        raise ValueError("Missing __Secure-1PSID in gaymini.txt or cookies.json")

    psidts = cookies.get("__Secure-1PSIDTS") or None
    extra_cookies = {k: v for k, v in cookies.items() if k not in ("__Secure-1PSID", "__Secure-1PSIDTS")}
    client = GeminiClient(secure_1psid=psid, secure_1psidts=psidts, cookies=extra_cookies)
    await client.init(verbose=True)

    # NOTE: GeminiClient has no check_account_status() method. Its init()
    # already runs _fetch_user_status() and stores the real result in
    # client.account_status (1000 = AVAILABLE, 1016 = UNAUTHENTICATED).
    # Never default to 1000 here — that used to make /health report
    # "healthy" while the session was actually expired.
    try:
        _LAST_ACCOUNT_STATUS = int(getattr(client, "account_status", 1016))
        logger.info("Gemini client initialized. Account status: %s", _LAST_ACCOUNT_STATUS)
    except Exception as e:
        logger.warning("Account status read failed: %s", e)
        _LAST_ACCOUNT_STATUS = 1016

    _ACTIVE_CLIENT = client
    _LAST_ROTATION_TIME = time.time()
    return _ACTIVE_CLIENT


# ---------------------------------------------------------------------------
# Model Resolver
# ---------------------------------------------------------------------------
def resolve_model_target(model_name: Optional[str]) -> Tuple[str, Model]:
    """
    Directly routes model requests to Google's real backend models:
    - Argon: Google Gemini Advanced Thinking / Reasoning (Model.ADVANCED_THINKING)
    - Hydrogen: Google Gemini Advanced High-Speed Execution (Model.ADVANCED_FLASH)
    """
    m = (model_name or "").lower().strip()
    if any(k in m for k in ["hydrogen", "hydroide", "flash", "fast", "sonnet", "haiku"]):
        return "gemini-4-hydrogen", Model.ADVANCED_FLASH
    else:
        # Default to Argon / Frontier Reasoning for all deep tasks, opus, and unmapped models
        return "gemini-4-argon", Model.ADVANCED_THINKING


def _extract_text_from_message(msg: Dict[str, Any]) -> str:
    content = msg.get("content", "")
    if isinstance(content, list):
        block_texts = []
        for b in content:
            if not isinstance(b, dict):
                block_texts.append(str(b))
                continue
            btype = b.get("type", "")
            if btype == "text":
                block_texts.append(b.get("text", ""))
            elif btype == "tool_use":
                tname = b.get("name", "")
                tinp = b.get("input", {})
                block_texts.append(f"```tool_call\n{json.dumps({'name': tname, 'arguments': tinp}, indent=2)}\n```")
            elif btype == "tool_result":
                tname = b.get("name", "Tool")
                tid = b.get("tool_use_id", "")
                tcontent = b.get("content", "")
                if isinstance(tcontent, list):
                    tcontent = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in tcontent])
                block_texts.append(f"[Tool Result ({tname} id={tid})]:\n{tcontent}")
        return "\n".join(block_texts)
    return str(content)


def _clean_reply_text(text: str) -> str:
    """Strip Gemini code-execution reference artifacts from a reply.

    When the model runs code, gemini_webapi's .text can include internal
    reference tokens like ``python?code_reference&code_event_index=1`` or
    ``text?code_stdout&code_event_index=1`` — they are not part of the
    answer and look broken in the chat UI, so remove them.
    """
    if not text:
        return text
    # e.g. python?code_reference&code_event_index=1
    #      text?code_stdout&code_event_index=2&foo=bar
    text = re.sub(r"\b\w+\?code_(?:reference|stdout|stderr|result)(?:&[^\s\"'<>]*)?", "", text)
    # bare leftovers, just in case
    text = re.sub(r"\bcode_(?:reference|stdout|stderr|result)\b", "", text)
    # collapse the blank lines the removal may leave behind
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_messages_to_prompt(
    messages: List[Dict[str, Any]], system_prompt: str = "", tools: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Standard natural multi-turn prompt formatting without synthetic mangling.
    """
    prompt_parts = []

    tools_instruction = _format_tools_prompt(tools) if tools else ""
    combined_system = f"{system_prompt}\n{tools_instruction}".strip()
    if combined_system:
        prompt_parts.append(f"System Instructions:\n{combined_system}\n")

    for msg in messages:
        role = msg.get("role", "user")
        content = _extract_text_from_message(msg).strip()
        if not content:
            continue

        if role == "system":
            prompt_parts.append(f"System:\n{content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant:\n{content}")
        else:
            prompt_parts.append(f"User:\n{content}")

    prompt_parts.append("Assistant:\n")
    return "\n\n".join(prompt_parts)


# ---------------------------------------------------------------------------
# CORS & Preflight Middleware
# ---------------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, Authorization, x-api-key, anthropic-version, *"
    )
    return response


API_KEY = os.environ.get("GAYMINI_API_KEY", "").strip()


@app.before_request
def require_api_key():
    """Optional shared-secret auth. Set GAYMINI_API_KEY if the server is reachable by anyone else."""
    if not API_KEY or request.method == "OPTIONS":
        return None
    if request.path in ("/health", "/", "/chat") or request.path.startswith(("/preview", "/uploads")):
        return None
    supplied = request.headers.get("x-api-key", "")
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        supplied = supplied or auth[7:]
    if supplied.strip() != API_KEY:
        return jsonify(adapter.error_body("invalid api key", "authentication_error")), 401
    return None


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        res = Response(status=200)
        res.headers["Access-Control-Allow-Origin"] = "*"
        res.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
        res.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, x-api-key, anthropic-version, *"
        )
        return res


# ---------------------------------------------------------------------------
# Live Web Preview Engine (/preview and static files)
# ---------------------------------------------------------------------------
@app.route("/preview", methods=["GET"])
@app.route("/preview/", methods=["GET"])
@app.route("/preview/<path:filename>", methods=["GET"])
def preview_project_files(filename="index.html"):
    """
    Live Web Preview Handler.
    Directly serves generated HTML / CSS / JS / Assets to user browser at:
    http://localhost:5000/preview
    """
    search_dirs = ["/root", "/root/scripted", "/sdcard", str(BASE_DIR)]
    if not filename or filename == "/":
        filename = "index.html"

    safe_rel = filename.lstrip("/")

    for d in search_dirs:
        candidate = os.path.join(d, safe_rel)
        if os.path.isfile(candidate):
            return send_from_directory(os.path.dirname(candidate), os.path.basename(candidate))

    if os.path.isfile("/root/index.html"):
        return send_from_directory("/root", "index.html")

    return Response("<h1>404 Not Found</h1><p>No previewable HTML file found. Please create index.html in /root</p>", status=404, mimetype="text/html")


# ---------------------------------------------------------------------------
# Web UI Dashboard & Interactive Chat Tester
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def web_dashboard():
    # The old dashboard's chat tester sent requests with no API key, so it
    # only ever showed "Thinking..." — send everyone to the real chat page.
    return redirect("/chat")
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Gemini 4 Argon & Hydrogen API Hub</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    body { font-family: 'Plus Jakarta Sans', sans-serif; }
    pre, code { font-family: 'JetBrains Mono', monospace; }
    .glass { background: rgba(18, 24, 38, 0.75); backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.08); }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen antialiased flex flex-col selection:bg-purple-500 selection:text-white">
  <!-- Top Navigation -->
  <header class="glass sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
    <div class="flex items-center space-x-3">
      <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
        <i class="fa-solid fa-microchip text-white text-lg"></i>
      </div>
      <div>
        <h1 class="text-lg font-bold tracking-tight flex items-center gap-2">
          Gemini 4 Suite <span class="text-xs px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30 font-medium">Pro Active</span>
        </h1>
        <p class="text-xs text-slate-400">OpenAI & Anthropic REST Gateway</p>
      </div>
    </div>
    <div class="flex items-center gap-3">
      <div class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        Status 1000 (AI Premium)
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="flex-1 max-w-6xl w-full mx-auto px-4 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
    <!-- Left Column: Interactive Chat Tester -->
    <div class="lg:col-span-7 flex flex-col space-y-6">
      <div class="glass rounded-2xl p-6 shadow-xl flex flex-col flex-1">
        <div class="flex items-center justify-between pb-4 border-b border-slate-800">
          <div>
            <h2 class="text-base font-bold flex items-center gap-2">
              <i class="fa-solid fa-flask text-indigo-400"></i> Live API Chat Tester
            </h2>
            <p class="text-xs text-slate-400">Test streaming SSE output directly in browser</p>
          </div>
          <div class="flex items-center p-1 bg-slate-900 rounded-xl border border-slate-800">
            <button id="btn-argon" onclick="selectModel('gemini-4-argon')" class="px-3 py-1.5 rounded-lg text-xs font-semibold transition-all bg-indigo-600 text-white shadow-sm flex items-center gap-1.5">
              <i class="fa-solid fa-brain"></i> Argon
            </button>
            <button id="btn-hydrogen" onclick="selectModel('gemini-4-hydrogen')" class="px-3 py-1.5 rounded-lg text-xs font-semibold transition-all text-slate-400 hover:text-white flex items-center gap-1.5">
              <i class="fa-solid fa-bolt"></i> Hydrogen
            </button>
          </div>
        </div>

        <div id="chat-messages" class="flex-1 min-h-[300px] max-h-[460px] overflow-y-auto py-4 space-y-4 pr-1 text-sm">
          <div class="flex items-start gap-3">
            <div class="w-7 h-7 rounded-lg bg-indigo-600/30 text-indigo-400 border border-indigo-500/30 flex items-center justify-center shrink-0">
              <i class="fa-solid fa-robot text-xs"></i>
            </div>
            <div class="p-3.5 rounded-2xl bg-slate-900 border border-slate-800 text-slate-200 text-xs leading-relaxed max-w-[85%]">
              Hello! Apni kon model test korte chan?
              <br><br>
              • <strong>Gemini 4 Argon:</strong> Deep Reasoning & Logic (Google Frontier Thinking)<br>
              • <strong>Gemini 4 Hydrogen:</strong> Fast Coding & Execution (Google Flash Engine)<br><br>
              Niche message likhe send korun!
            </div>
          </div>
        </div>

        <div class="pt-4 border-t border-slate-800 flex gap-2">
          <input id="chat-input" type="text" placeholder="Type prompt here (e.g. Write a Python quicksort function)..." 
                 class="flex-1 bg-slate-900 border border-slate-800 rounded-xl px-4 py-2.5 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-indigo-500 transition-colors"
                 onkeydown="if(event.key==='Enter') sendMessage()">
          <button id="send-btn" onclick="sendMessage()" class="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 active:scale-95 text-white font-semibold rounded-xl text-xs flex items-center gap-1.5 shadow-lg shadow-indigo-600/20 transition-all">
            <span>Send</span> <i class="fa-solid fa-arrow-up text-xs"></i>
          </button>
        </div>
      </div>
    </div>

    <!-- Right Column: API Endpoints & Quick Copy Code -->
    <div class="lg:col-span-5 flex flex-col space-y-6">
      <div class="glass rounded-2xl p-6 shadow-xl space-y-4">
        <h3 class="text-sm font-bold flex items-center gap-2">
          <i class="fa-solid fa-link text-blue-400"></i> Active Endpoints
        </h3>
        <div class="space-y-2.5 text-xs">
          <div class="p-3 bg-slate-900 rounded-xl border border-slate-800 flex items-center justify-between">
            <div>
              <span class="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-mono font-bold text-[10px]">POST</span>
              <span class="font-mono text-slate-300 ml-2">/v1/chat/completions</span>
            </div>
            <span class="text-[10px] text-slate-500">OpenAI</span>
          </div>
          <div class="p-3 bg-slate-900 rounded-xl border border-slate-800 flex items-center justify-between">
            <div>
              <span class="px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-400 font-mono font-bold text-[10px]">POST</span>
              <span class="font-mono text-slate-300 ml-2">/v1/messages</span>
            </div>
            <span class="text-[10px] text-slate-500">Anthropic</span>
          </div>
          <div class="p-3 bg-slate-900 rounded-xl border border-slate-800 flex items-center justify-between">
            <div>
              <span class="px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400 font-mono font-bold text-[10px]">GET</span>
              <span class="font-mono text-slate-300 ml-2">/v1/models</span>
            </div>
            <span class="text-[10px] text-slate-500">Models</span>
          </div>
          <div class="p-3 bg-slate-900 rounded-xl border border-slate-800 flex items-center justify-between">
            <div>
              <span class="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 font-mono font-bold text-[10px]">GET</span>
              <span class="font-mono text-slate-300 ml-2">/preview</span>
            </div>
            <span class="text-[10px] text-slate-500">HTML Preview</span>
          </div>
        </div>
      </div>

      <div class="glass rounded-2xl p-6 shadow-xl flex flex-col flex-1">
        <div class="flex items-center justify-between mb-3">
          <h3 class="text-sm font-bold flex items-center gap-2">
            <i class="fa-solid fa-terminal text-emerald-400"></i> Quick Call (Curl)
          </h3>
          <button onclick="copySnippet()" class="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1">
            <i class="fa-regular fa-copy"></i> Copy
          </button>
        </div>
        <pre id="curl-snippet" class="p-4 bg-slate-950 rounded-xl border border-slate-800/80 text-[11px] leading-relaxed text-slate-300 overflow-x-auto whitespace-pre font-mono">curl http://localhost:5000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "gemini-4-argon",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": true
  }'</pre>
      </div>
    </div>
  </main>

  <script>
    let currentModel = 'gemini-4-argon';

    function selectModel(m) {
      currentModel = m;
      const btnA = document.getElementById('btn-argon');
      const btnH = document.getElementById('btn-hydrogen');
      if (m === 'gemini-4-argon') {
        btnA.className = 'px-3 py-1.5 rounded-lg text-xs font-semibold transition-all bg-indigo-600 text-white shadow-sm flex items-center gap-1.5';
        btnH.className = 'px-3 py-1.5 rounded-lg text-xs font-semibold transition-all text-slate-400 hover:text-white flex items-center gap-1.5';
      } else {
        btnH.className = 'px-3 py-1.5 rounded-lg text-xs font-semibold transition-all bg-indigo-600 text-white shadow-sm flex items-center gap-1.5';
        btnA.className = 'px-3 py-1.5 rounded-lg text-xs font-semibold transition-all text-slate-400 hover:text-white flex items-center gap-1.5';
      }
    }

    async function sendMessage() {
      const input = document.getElementById('chat-input');
      const text = input.value.trim();
      if (!text) return;
      input.value = '';

      const container = document.getElementById('chat-messages');

      const userDiv = document.createElement('div');
      userDiv.className = 'flex items-start gap-3 justify-end';
      userDiv.innerHTML = `
        <div class="p-3.5 rounded-2xl bg-indigo-600/30 border border-indigo-500/40 text-slate-100 text-xs leading-relaxed max-w-[85%] whitespace-pre-wrap">${text}</div>
        <div class="w-7 h-7 rounded-lg bg-indigo-600 text-white flex items-center justify-center shrink-0 text-xs font-bold">U</div>
      `;
      container.appendChild(userDiv);
      container.scrollTop = container.scrollHeight;

      const botDiv = document.createElement('div');
      botDiv.className = 'flex items-start gap-3';
      const msgBubbleId = 'bot-msg-' + Date.now();
      botDiv.innerHTML = `
        <div class="w-7 h-7 rounded-lg bg-indigo-600/30 text-indigo-400 border border-indigo-500/30 flex items-center justify-center shrink-0">
          <i class="fa-solid fa-robot text-xs"></i>
        </div>
        <div id="${msgBubbleId}" class="p-3.5 rounded-2xl bg-slate-900 border border-slate-800 text-slate-200 text-xs leading-relaxed max-w-[85%] whitespace-pre-wrap font-sans">
          <span class="inline-block animate-pulse text-slate-400">Thinking (${currentModel})...</span>
        </div>
      `;
      container.appendChild(botDiv);
      container.scrollTop = container.scrollHeight;

      try {
        const resp = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            model: currentModel,
            messages: [{role: 'user', content: text}],
            stream: true
          })
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let full = '';
        const bubble = document.getElementById(msgBubbleId);
        bubble.innerHTML = '';

        while (true) {
          const {value, done} = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value);
          const lines = chunk.split('\\n');
          for (const line of lines) {
            if (line.startsWith('data: ') && line.trim() !== 'data: [DONE]') {
              try {
                const j = JSON.parse(line.slice(6));
                const delta = j.choices[0]?.delta?.content || '';
                full += delta;
                bubble.textContent = full;
                container.scrollTop = container.scrollHeight;
              } catch(e) {}
            }
          }
        }
      } catch (err) {
        document.getElementById(msgBubbleId).textContent = 'Error: ' + err.message;
      }
    }

    function copySnippet() {
      const code = document.getElementById('curl-snippet').innerText;
      navigator.clipboard.writeText(code);
      alert('Copied curl command to clipboard!');
    }
  </script>
</body>
</html>
"""
    return html


@app.route("/chat", methods=["GET"])
def chat_page():
    """Bangla chat page (chat.html) with model selector + API key field.
    Served from the server itself so the public link just works."""
    return send_file(str(BASE_DIR / "chat.html"), mimetype="text/html")


# ---------------------------------------------------------------------------
# Image Edit API (azadx69x editor) — POST /edit-image starts a job, the page
# polls GET /edit-status/<job>. The edit takes 1-4 minutes and the public
# tunnel drops long-held connections, so no single request waits it out.
# The phone uploads an image; we host it briefly at /uploads/<name> so the
# editor API (which takes a public image URL) can fetch it. The edited image
# is compressed to JPEG and hosted on tmpfiles (the tunnel is ~6 KB/s for big
# files); uploaded files are deleted right after the edit.
# ---------------------------------------------------------------------------
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
_SAFE_UPLOAD_NAME = re.compile(r"^[a-zA-Z0-9_-]+\.(jpg|jpeg|png|webp|gif|mp4|webm)$", re.IGNORECASE)


@app.route("/uploads/<name>", methods=["GET"])
def serve_upload(name):
    if not _SAFE_UPLOAD_NAME.match(name):
        return jsonify({"error": "not found"}), 404
    path = UPLOADS_DIR / name
    if not path.is_file():
        return jsonify({"error": "not found"}), 404
    ext = Path(name).suffix.lower()
    mime = {
        ".mp4": "video/mp4", ".webm": "video/webm",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"
    }.get(ext, "application/octet-stream")
    resp = send_file(str(path), mimetype=mime)
    resp.headers["Accept-Ranges"] = "bytes"
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/edit-image", methods=["POST"])
@app.route("/api/v1/edit", methods=["POST"])
def edit_image():
    import subprocess

    prompt = (request.form.get("prompt") or "").strip()
    origin = (request.form.get("origin") or "").strip().rstrip("/")
    f = request.files.get("image")
    if not prompt:
        return jsonify(adapter.error_body("prompt is required", "invalid_request_error")), 400
    if f is None or not f.filename:
        return jsonify(adapter.error_body("image is required", "invalid_request_error")), 400
    if not origin.startswith("http"):
        return jsonify(adapter.error_body("public origin is required - open the chat from the public link", "invalid_request_error")), 400
    ext = Path(f.filename).suffix.lower().lstrip(".")
    if ext not in ("jpg", "jpeg", "png", "webp", "gif"):
        return jsonify(adapter.error_body("image must be jpg, png, webp or gif", "invalid_request_error")), 400
    data = f.read()
    if len(data) > 10 * 1024 * 1024:
        return jsonify(adapter.error_body("image too large (max 10 MB)", "invalid_request_error")), 400

    job_id = uuid.uuid4().hex[:12]
    _EDIT_JOBS[job_id] = {"status": "working", "progress": "ছবি পাঠানো হয়েছে, এডিট শুরু হচ্ছে..."}
    _prune_jobs()
    t = threading.Thread(target=_run_edit_job, args=(job_id, data, ext, prompt, origin), daemon=True)
    t.start()
    return jsonify({"job": job_id, "status": "working"})


@app.route("/edit-status/<job_id>", methods=["GET"])
@app.route("/api/v1/image-status/<job_id>", methods=["GET"])
def edit_status(job_id):
    job = _EDIT_JOBS.get(job_id)
    if job is None:
        return jsonify(adapter.error_body("unknown job", "invalid_request_error")), 404
    # Tunnel domains rotate often; if the result was hosted on our tunnel,
    # rewrite its origin to the domain the phone is using right now so the
    # link still works after a rotation.
    out = dict(job)
    url = out.get("image_url") or ""
    if url.startswith("http") and ".app.lhr.life" in url:
        from urllib.parse import urlparse

        out["image_url"] = request.host_url.rstrip("/") + urlparse(url).path
    return jsonify(out)


_EDIT_JOBS: Dict[str, dict] = {}
# Result files served from our own tunnel as a fallback when tmpfiles is
# unreachable — those must NOT be deleted by the job's cleanup.
_EDIT_FALLBACK_FILES = set()


def _prune_jobs():
    """Keep only the newest 30 jobs so memory does not grow forever."""
    while len(_EDIT_JOBS) > 30:
        _EDIT_JOBS.pop(next(iter(_EDIT_JOBS)))


def _host_result_file(path, origin):
    """Host a finished image for the phone to download.

    Returns (url, keep_file). Tries tmpfiles once (fast download, 30s cap),
    then falls back to serving the file from our own tunnel /uploads/ link
    (slow ~6 KB/s but always works). tmpfiles uploads from this VM often
    hang, so we do NOT retry it — retrying just wastes minutes.
    """
    import subprocess

    try:
        up = subprocess.run(["curl", "-s", "-m", "30", "-F", f"file=@{path}",
                             "https://tmpfiles.org/api/v1/upload"],
                            capture_output=True, text=True, timeout=40)
        page_url = json.loads(up.stdout)["data"]["url"]
        pg = subprocess.run(["curl", "-s", "-m", "30", page_url],
                            capture_output=True, text=True, timeout=40)
        m = re.search(r"https://tmpfiles\.org/dl/[^\s\"'<>]+", pg.stdout)
        if m:
            return m.group(0), False
        logger.warning("tmpfiles host: no dl link in page")
    except Exception as e:
        logger.warning("tmpfiles host failed: %s", e)
    return f"{origin}/uploads/{Path(path).name}", True


def _azad_edit_and_host(data, ext, prompt, origin, progress):
    """Azad editor pipeline, shared by /edit-image and by the Google-edit
    fallback: save + downscale the input, let the Azad editor API fetch it
    from {origin}/uploads/<name>, compress the result and host it.
    Returns {"image_url", "size", "host"}; raises RuntimeError on failure.
    NOTE: the editor API must fetch the input from a PUBLIC url, so origin
    has to be the public (Render/tunnel) address, never http://127.0.0.1.
    """
    import subprocess

    name = f"{uuid.uuid4().hex}.{ext if ext != 'jpeg' else 'jpg'}"
    upath = UPLOADS_DIR / name
    outpath = UPLOADS_DIR / f"edited-{name}.png"
    jpgpath = UPLOADS_DIR / f"edited-{Path(name).stem}.jpg"
    smallpath = None
    try:
        upath.write_bytes(data)
        # ফোনের ছবি বড় হলে এডিটরকে পাঠানোর আগেই ছোট করে নাও (সবচেয়ে বড়
        # দিক ১২৮০px)। এডিটর ছবিটা আমাদের ধীর টানেল থেকে নামায় — ছোট ফাইল
        # নামাতে কম সময় লাগে, এডিটও তাড়াতাড়ি হয়, মান একই থাকে।
        input_path = upath
        smallpath = UPLOADS_DIR / f"small-{name}"
        try:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(upath),
                            "-vf", "scale='min(1280,iw)':-2", "-q:v", "3",
                            str(smallpath)],
                           timeout=60, check=True)
            if smallpath.is_file() and 0 < smallpath.stat().st_size < upath.stat().st_size:
                input_path = smallpath
                upath.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("input downscale failed, using original: %s", e)
        img_url = f"{origin}/uploads/{input_path.name}"
        progress("AI ছবিটা এডিট করছে...")
        res = subprocess.run(
            ["curl", "-s", "-m", "240", "-G",
             "--data-urlencode", f"url={img_url}",
             "--data-urlencode", f"prompt={prompt}",
             "https://azadx69x.is-a.dev/api/editor",
             "-o", str(outpath), "-w", "%{http_code}"],
            capture_output=True, text=True, timeout=250,
        )
        http_code = (res.stdout or "").strip()
        if http_code != "200" or not outpath.is_file():
            raise RuntimeError("image edit failed on the editor server")
        edited = outpath.read_bytes()
        if not edited.startswith(b"\x89PNG"):
            try:
                err = json.loads(edited.decode("utf-8", "ignore"))
                msg = (err.get("data") or {}).get("message") or err.get("message") or "image edit failed"
            except Exception:
                msg = "image edit failed"
            raise RuntimeError(str(msg))
        # Compress to JPEG and host on tmpfiles so the phone downloads fast
        # (the localhost.run tunnel is ~6 KB/s for big files).
        upload_path = outpath
        try:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(outpath),
                            "-vf", "scale=1280:-1", "-q:v", "3", str(jpgpath)],
                           timeout=60, check=True)
            if jpgpath.is_file() and jpgpath.stat().st_size > 0:
                upload_path = jpgpath
        except Exception as e:
            logger.warning("ffmpeg compress failed, uploading PNG as-is: %s", e)
        progress("ছবি তৈরি — ফোনে পাঠানো হচ্ছে...")
        url, keep = _host_result_file(upload_path, origin)
        result = {"image_url": url, "size": upload_path.stat().st_size,
                  "host": "tunnel" if keep else "tmpfiles"}
        if keep:
            _EDIT_FALLBACK_FILES.add(str(upload_path))
        return result
    finally:
        cleanup = [upath, outpath]
        if smallpath is not None:
            cleanup.append(smallpath)
        for p in cleanup:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            if str(jpgpath) not in _EDIT_FALLBACK_FILES:
                jpgpath.unlink(missing_ok=True)
        except Exception:
            pass


def _run_edit_job(job_id, data, ext, prompt, origin):
    """Whole edit pipeline in a background thread (edit takes 1-4 minutes and
    the public tunnel drops long-held connections, so the page polls instead
    of waiting on one request)."""
    def progress(msg):
        _EDIT_JOBS[job_id]["progress"] = msg

    try:
        result = _azad_edit_and_host(data, ext, prompt, origin, progress)
        _EDIT_JOBS[job_id] = {"status": "done", **result}
    except Exception as e:
        logger.warning("edit job %s failed: %s", job_id, e)
        _EDIT_JOBS[job_id] = {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Nano Banana Pro — Google image generation & Google image editing, through
# the same Gemini web session as chat (no extra key needed).
#   POST /generate-image      {prompt, origin}            -> new image
#   POST /edit-image-google   {image, prompt, origin}     -> edited image
# Both start an async job; the page polls GET /edit-status/<job> like the
# Azad editor. NOTE (tested 2026-11-18): Gemini web answered image
# generation with "image creation isn't available in your location yet"
# for this account, and uploading an image to Google from this VM times
# out — so these jobs surface Google's own message instead of failing
# silently. When Google allows it, they return a real image.
# ---------------------------------------------------------------------------
def _extract_image_url(text: str):
    """Find a direct image URL in Gemini's text reply.

    When Gemini can't generate an image it sometimes answers with a
    searched/found one instead ("Here is an image of X: <img
    src='https://...'>"). Return the first URL that looks like an image.
    """
    if not text:
        return None
    for u in re.findall(r"https?://[^\s\"'\[\]<>]+", text):
        u = u.rstrip(".,)")
        low = u.lower()
        if any(k in low for k in (".jpg", ".jpeg", ".png", ".webp", ".gif",
                                  "gstatic.com", "googleusercontent.com")):
            return u
    return None


def _download_found_image(url: str):
    """Download a found image URL into uploads/; returns the Path, or None
    if the download failed or the file is not an image."""
    import subprocess

    path = UPLOADS_DIR / f"found-{uuid.uuid4().hex[:12]}"
    try:
        res = subprocess.run(["curl", "-s", "-L", "-m", "30",
                              "-A", "Mozilla/5.0",
                              "-o", str(path), "-w", "%{http_code}", url],
                             capture_output=True, text=True, timeout=40)
        if (res.stdout or "").strip() != "200" or not path.is_file():
            return None
        head = path.read_bytes()[:12]
        if (head.startswith(b"\xff\xd8\xff") or head.startswith(b"\x89PNG")
                or head.startswith(b"RIFF") or head.startswith(b"GIF8")):
            return path
        return None
    except Exception as e:
        logger.warning("found image download failed: %s", e)
        return None


def _run_image_gen_job(job_id, prompt, origin, data=None, kind="generate", model="nano-banana-pro"):
    """Background job for Nano Banana Pro generation / Google image edit / Google Flow Imagen 3."""
    import io
    import subprocess

    def progress(msg):
        _EDIT_JOBS[job_id]["progress"] = msg

    try:
        is_flow = "flow" in (model or "").lower()
        if kind == "google-edit":
            progress("Google chhobi edit korche...")
        elif is_flow:
            progress("Google Flow (Imagen 3) chhobi banacche...")
        else:
            progress("Nano Banana Pro chhobi banacche...")

        # If user explicitly selected Google Flow Imagen 3
        if is_flow and kind == "generate":
            try:
                flow_client = flow_engine.FlowClient()
                flow_res = flow_client.generate_image_flow(prompt)
                if flow_res.get("status") == "success" and flow_res.get("images"):
                    flow_img_url = flow_res["images"][0]
                    dl_path = _download_found_image(flow_img_url)
                    if dl_path is not None:
                        fjpg = UPLOADS_DIR / f"flow-{uuid.uuid4().hex[:12]}.jpg"
                        upload_path = dl_path
                        try:
                            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(dl_path),
                                            "-vf", "scale=1280:-1", "-q:v", "3", str(fjpg)],
                                           timeout=60, check=True)
                            if fjpg.is_file() and fjpg.stat().st_size > 0:
                                upload_path = fjpg
                        except Exception as e:
                            logger.warning("ffmpeg compress failed (flow direct): %s", e)
                        progress("Flow chhobi toiri — phone-e pathano hocche...")
                        url, keep = _host_result_file(upload_path, origin)
                        _EDIT_JOBS[job_id] = {
                            "status": "done", "image_url": url,
                            "size": upload_path.stat().st_size,
                            "host": "tunnel" if keep else "tmpfiles",
                            "kind": kind, "source": "flow"
                        }
                        if keep:
                            _EDIT_FALLBACK_FILES.add(str(upload_path))
                        try:
                            dl_path.unlink(missing_ok=True)
                            if not keep:
                                fjpg.unlink(missing_ok=True)
                        except Exception:
                            pass
                        return
                logger.warning("Google Flow did not return images, falling back to Nano Banana Pro")
                progress("Nano Banana Pro diye chhobi banano hocche...")
            except Exception as ef:
                logger.warning(f"Google Flow Imagen 3 attempt failed: {ef}. Falling back to Nano Banana Pro.")
                progress("Nano Banana Pro (Google Imagen) diye chhobi banano hocche...")

        with _SLOTS:
            async def _worker():
                client = await get_gemini_client()
                if kind == "google-edit":
                    return await client.generate_content(
                        f"Edit this image: {prompt}", files=[io.BytesIO(data)],
                        model=Model.ADVANCED_THINKING, temporary=True)
                return await client.generate_content(
                    f"Generate an image of: {prompt}",
                    model=Model.ADVANCED_THINKING, temporary=True)

            resp = _run_coroutine_sync(_worker())
        imgs = []
        for c in (getattr(resp, "candidates", None) or []):
            imgs.extend(getattr(c, "generated_images", None) or [])

        # Fallback 1: Google Flow Imagen 3 if Gemini didn't return images
        flow_fail = ""
        if not imgs and kind == "generate":
            try:
                progress("Google Flow (Imagen 3) backup diye chhobi banano hocche...")
                flow_client = flow_engine.FlowClient()
                flow_res = flow_client.generate_image_flow(prompt)
                if flow_res.get("status") == "success" and flow_res.get("images"):
                    flow_img_url = flow_res["images"][0]
                    dl_path = _download_found_image(flow_img_url)
                    if dl_path is not None:
                        fjpg = UPLOADS_DIR / f"flow-{uuid.uuid4().hex[:12]}.jpg"
                        upload_path = dl_path
                        try:
                            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(dl_path),
                                            "-vf", "scale=1280:-1", "-q:v", "3", str(fjpg)],
                                           timeout=60, check=True)
                            if fjpg.is_file() and fjpg.stat().st_size > 0:
                                upload_path = fjpg
                        except Exception as e:
                            logger.warning("ffmpeg compress failed (flow backup): %s", e)
                        progress("Flow backup chhobi toiri — phone-e pathano hocche...")
                        url, keep = _host_result_file(upload_path, origin)
                        _EDIT_JOBS[job_id] = {
                            "status": "done", "image_url": url,
                            "size": upload_path.stat().st_size,
                            "host": "tunnel" if keep else "tmpfiles",
                            "kind": kind, "source": "flow"
                        }
                        if keep:
                            _EDIT_FALLBACK_FILES.add(str(upload_path))
                        try:
                            dl_path.unlink(missing_ok=True)
                            if not keep:
                                fjpg.unlink(missing_ok=True)
                        except Exception:
                            pass
                        return
            except Exception as ef:
                flow_fail = str(ef)
                logger.warning(f"Google Flow fallback failed: {ef}")

        # Google edit fallback: if Gemini did not return an edited image,
        # fall back to the Azad editor (which does not need the Google
        # session) instead of failing. Uploading an image to Gemini web
        # hangs from servers, so this fallback is what makes Google edit
        # actually work instead of timing out.
        if not imgs and kind == "google-edit":
            try:
                progress("Google এডিট হয়নি — Azad এডিটর দিয়ে চেষ্টা করছে...")
                ext = "jpg"
                if data[:8].startswith(b"\x89PNG"):
                    ext = "png"
                elif data[:6] in (b"GIF87a", b"GIF89a"):
                    ext = "gif"
                elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
                    ext = "webp"
                result = _azad_edit_and_host(data, ext, prompt, origin, progress)
                _EDIT_JOBS[job_id] = {"status": "done", **result,
                                      "kind": kind, "source": "azad-fallback"}
                return
            except Exception as ef:
                logger.warning("Google-edit Azad fallback failed: %s", ef)

        # Fallback 2: Found image in response text
        if not imgs:
            text = (getattr(resp, "text", "") or "").strip()
            found_url = _extract_image_url(text)
            if found_url:
                progress("Google chhobi banayni, kintu ekta chhobi khuje peyeche — download hocche...")
                dl_path = _download_found_image(found_url)
                if dl_path is not None:
                    fjpg = UPLOADS_DIR / f"found-{uuid.uuid4().hex[:12]}.jpg"
                    upload_path = dl_path
                    try:
                        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(dl_path),
                                        "-vf", "scale=1280:-1", "-q:v", "3", str(fjpg)],
                                       timeout=60, check=True)
                        if fjpg.is_file() and fjpg.stat().st_size > 0:
                            upload_path = fjpg
                    except Exception as e:
                        logger.warning("ffmpeg compress failed (found): %s", e)
                    progress("Chhobi toiri — phone-e pathano hocche...")
                    url, keep = _host_result_file(upload_path, origin)
                    _EDIT_JOBS[job_id] = {"status": "done", "image_url": url,
                                          "size": upload_path.stat().st_size,
                                          "host": "tunnel" if keep else "tmpfiles",
                                          "kind": kind, "source": "search"}
                    if keep:
                        _EDIT_FALLBACK_FILES.add(str(upload_path))
                    try:
                        dl_path.unlink(missing_ok=True)
                        if not keep:
                            fjpg.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return

            clean = re.sub(r"<[^>]+>", " ", text)
            clean = re.sub(r"\s+", " ", clean).strip()
            if "signed out" in clean.lower() or "not available" in clean.lower():
                detail = "Google Gemini session unauthenticated ba __Secure-1PSIDTS meyadottirno. Chat cholleo chhobi bananur jonno Gemini-te active cookie proyojon. 🔑 botam theke gemini.google.com-er cookie sync korun."
                if flow_fail:
                    detail += f" (Flow backup-o fail: {flow_fail[:120]})"
                raise RuntimeError(detail)
            raise RuntimeError("Google chhobi banayni: " + (clean[:200] if clean
                              else "Kono chhobi asheni — ei account-e chhobi banano ekhono chalu hoyni."))
        name = f"nano-{uuid.uuid4().hex[:12]}.png"

        async def _save():
            return await imgs[0].save(path=str(UPLOADS_DIR), filename=name, full_size=True)

        saved_path = Path(_run_coroutine_sync(_save()))
        jpgpath = UPLOADS_DIR / f"{saved_path.stem}.jpg"
        upload_path = saved_path
        try:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(saved_path),
                            "-vf", "scale=1280:-1", "-q:v", "3", str(jpgpath)],
                           timeout=60, check=True)
            if jpgpath.is_file() and jpgpath.stat().st_size > 0:
                upload_path = jpgpath
        except Exception as e:
            logger.warning("ffmpeg compress failed (nano): %s", e)
        progress("Chhobi toiri — phone-e pathano hocche...")
        url, keep = _host_result_file(upload_path, origin)
        _EDIT_JOBS[job_id] = {"status": "done", "image_url": url,
                              "size": upload_path.stat().st_size,
                              "host": "tunnel" if keep else "tmpfiles", "kind": kind, "source": "nano-banana-pro"}
        if keep:
            _EDIT_FALLBACK_FILES.add(str(upload_path))
        try:
            saved_path.unlink(missing_ok=True)
            if not keep:
                jpgpath.unlink(missing_ok=True)
        except Exception:
            pass
    except Exception as e:
        logger.warning("image job %s (%s) failed: %s", job_id, kind, e)
        _EDIT_JOBS[job_id] = {"status": "error", "message": str(e), "kind": kind}


@app.route("/generate-image", methods=["POST"])
@app.route("/api/v1/image", methods=["POST"])
def generate_image():
    """Nano Banana Pro / Google Flow: make a new image from a text prompt (async job)."""
    prompt = (request.form.get("prompt") or "").strip()
    origin = (request.form.get("origin") or "").strip().rstrip("/")
    model = (request.form.get("model") or "nano-banana-pro").strip().lower()
    if not origin:
        origin = request.host_url.rstrip("/")
    if not prompt:
        return jsonify(adapter.error_body("prompt is required", "invalid_request_error")), 400
    if not origin.startswith("http"):
        return jsonify(adapter.error_body("public origin is required - open the chat from the public link", "invalid_request_error")), 400
    job_id = uuid.uuid4().hex[:12]
    model_label = "Google Flow Imagen 3" if "flow" in model else "Nano Banana Pro"
    _EDIT_JOBS[job_id] = {"status": "working", "progress": f"{model_label} chhobi banacche...", "kind": "generate", "model": model}
    _prune_jobs()
    threading.Thread(target=_run_image_gen_job, args=(job_id, prompt, origin, None, "generate", model), daemon=True).start()
    return jsonify({"job": job_id, "status": "working"})


@app.route("/edit-image-google", methods=["POST"])
@app.route("/api/v1/edit-google", methods=["POST"])
def edit_image_google():
    """Google image edit: edit an uploaded image with Gemini (async job)."""
    prompt = (request.form.get("prompt") or "").strip()
    origin = (request.form.get("origin") or "").strip().rstrip("/")
    f = request.files.get("image")
    if not prompt:
        return jsonify(adapter.error_body("prompt is required", "invalid_request_error")), 400
    if f is None or not f.filename:
        return jsonify(adapter.error_body("image is required", "invalid_request_error")), 400
    if not origin.startswith("http"):
        return jsonify(adapter.error_body("public origin is required - open the chat from the public link", "invalid_request_error")), 400
    ext = Path(f.filename).suffix.lower().lstrip(".")
    if ext not in ("jpg", "jpeg", "png", "webp", "gif"):
        return jsonify(adapter.error_body("image must be jpg, png, webp or gif", "invalid_request_error")), 400
    data = f.read()
    if len(data) > 10 * 1024 * 1024:
        return jsonify(adapter.error_body("image too large (max 10 MB)", "invalid_request_error")), 400
    job_id = uuid.uuid4().hex[:12]
    _EDIT_JOBS[job_id] = {"status": "working", "progress": "Google chhobi edit korche...", "kind": "google-edit"}
    _prune_jobs()
    threading.Thread(target=_run_image_gen_job, args=(job_id, prompt, origin, data, "google-edit"), daemon=True).start()
    return jsonify({"job": job_id, "status": "working"})


# ---------------------------------------------------------------------------
# Google Flow (Veo & Imagen) & Auto-Login Sync API
# ---------------------------------------------------------------------------
@app.route("/generate-video", methods=["POST"])
@app.route("/api/v1/video", methods=["POST"])
def generate_video():
    """Generate a video using Google Flow / Veo 3.1 (async job)."""
    data = request.get_json(silent=True) or {}
    prompt = (request.form.get("prompt") or data.get("prompt") or "").strip()
    origin = (request.form.get("origin") or data.get("origin") or "").strip().rstrip("/")
    if not origin:
        origin = request.host_url.rstrip("/")

    if not prompt:
        return jsonify(adapter.error_body("prompt is required for video generation", "invalid_request_error")), 400

    f = request.files.get("image")
    image_bytes = None
    if f and f.filename:
        image_bytes = f.read()

    job_id = flow_engine.start_video_generation(prompt, image_bytes, origin)
    return jsonify({"job": job_id, "status": "working", "prompt": prompt})


@app.route("/video-status/<job_id>", methods=["GET"])
@app.route("/api/v1/video-status/<job_id>", methods=["GET"])
def video_status(job_id: str):
    """Poll video generation progress."""
    job = flow_engine.get_video_job(job_id)
    if not job:
        return jsonify(adapter.error_body("Video job not found", "invalid_request_error")), 404
    return jsonify(job)


# ---------------------------------------------------------------------------
# 1. Gemini Accounts & Auth API (gemini.google.com)
# ---------------------------------------------------------------------------
@app.route("/api/gemini/accounts", methods=["GET"])
def api_gemini_accounts_list():
    try:
        accs = flow_engine.list_accounts(service="gemini")
        return jsonify({"status": "success", "service": "gemini", "accounts": accs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/gemini/accounts/switch", methods=["POST"])
def api_gemini_accounts_switch():
    global _ACTIVE_CLIENT
    data = request.get_json(force=True, silent=True) or {}
    account_id = data.get("id") or request.form.get("id")
    if not account_id:
        return jsonify({"error": "Account id is required"}), 400
    success = flow_engine.switch_account(service="gemini", account_id=account_id)
    if not success:
        return jsonify({"error": "Gemini account not found"}), 404
    _ACTIVE_CLIENT = None
    return jsonify({
        "status": "success",
        "service": "gemini",
        "active_account": account_id,
        "message": f"Switched to Gemini account {account_id}"
    })


@app.route("/api/gemini/accounts/add", methods=["POST"])
def api_gemini_accounts_add():
    data = request.get_json(force=True, silent=True) or {}
    account_id = data.get("id") or request.form.get("id") or ""
    name = data.get("name") or request.form.get("name") or ""
    email = data.get("email") or request.form.get("email") or ""
    raw_cookies = data.get("cookies") or request.form.get("cookies")
    if not raw_cookies:
        return jsonify({"error": "Cookies are required"}), 400

    parsed = raw_cookies if isinstance(raw_cookies, (dict, list)) else flow_engine.parse_cookie_string(str(raw_cookies))
    if not parsed:
        return jsonify({"error": "Could not parse Gemini cookies"}), 400

    res = flow_engine.add_or_update_account(service="gemini", account_id=account_id, name=name, cookies=parsed, email=email)
    global _ACTIVE_CLIENT
    _ACTIVE_CLIENT = None
    return jsonify({"status": "success", "service": "gemini", "result": res})


@app.route("/api/gemini/accounts/delete", methods=["POST"])
def api_gemini_accounts_delete():
    global _ACTIVE_CLIENT
    data = request.get_json(force=True, silent=True) or {}
    account_id = data.get("id") or request.form.get("id")
    if not account_id:
        return jsonify({"error": "Account id is required"}), 400
    success = flow_engine.delete_account(service="gemini", account_id=account_id)
    if not success:
        return jsonify({"error": "Account not found"}), 404
    _ACTIVE_CLIENT = None
    return jsonify({"status": "success", "service": "gemini", "message": f"Gemini account {account_id} deleted"})


@app.route("/api/gemini/auth/sync", methods=["POST"])
def api_gemini_auth_sync():
    global _ACTIVE_CLIENT
    data = request.get_json(force=True, silent=True) or {}
    raw_cookies = data.get("cookies") or request.form.get("cookies")
    if not raw_cookies and isinstance(data, dict) and any("PSID" in k for k in data.keys()):
        raw_cookies = data
    if not raw_cookies:
        return jsonify({"error": "No cookies provided."}), 400

    parsed = raw_cookies if isinstance(raw_cookies, (dict, list)) else flow_engine.parse_cookie_string(str(raw_cookies))
    if not parsed:
        return jsonify({"error": "Could not parse cookies."}), 400

    flow_engine.save_cookies(parsed, service="gemini")
    _ACTIVE_CLIENT = None
    flow_engine.add_or_update_account(service="gemini", account_id="gemini_default", name="Primary Gemini Account", cookies=parsed)

    has_psid = "__Secure-1PSID" in parsed or "SID" in parsed
    has_psidts = "__Secure-1PSIDTS" in parsed
    return jsonify({
        "status": "success",
        "service": "gemini",
        "message": "Gemini cookies synchronized successfully!",
        "has_psid": has_psid,
        "has_psidts": has_psidts,
        "warning": None if has_psidts else "Warning: __Secure-1PSIDTS missing. Chat works but image generation requires __Secure-1PSIDTS."
    })


@app.route("/api/gemini/auth/status", methods=["GET"])
def api_gemini_auth_status():
    cookies = flow_engine.load_cookies(service="gemini")
    psid = cookies.get("__Secure-1PSID")
    psidts = cookies.get("__Secure-1PSIDTS")
    authenticated = bool(psid)
    detail = "Gemini session active" if psidts else "Sotorkota: __Secure-1PSIDTS nei (Chat cholbe, kintu chhobi toirir jonno 1PSIDTS din)"
    if not psid:
        detail = "Gemini cookie pawa jayni (__Secure-1PSID onuposthit). gemini.google.com theke cookie sync korun."
    return jsonify({
        "service": "gemini",
        "authenticated": authenticated,
        "cookies_count": len(cookies),
        "has_psid": bool(psid),
        "has_psidts": bool(psidts),
        "detail": detail
    })


# ---------------------------------------------------------------------------
# 2. Google Flow Accounts & Auth API (flow.google.com)
# ---------------------------------------------------------------------------
@app.route("/api/flow/accounts", methods=["GET"])
def api_flow_accounts_list():
    try:
        accs = flow_engine.list_accounts(service="flow")
        return jsonify({"status": "success", "service": "flow", "accounts": accs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/flow/accounts/switch", methods=["POST"])
def api_flow_accounts_switch():
    data = request.get_json(force=True, silent=True) or {}
    account_id = data.get("id") or request.form.get("id")
    if not account_id:
        return jsonify({"error": "Account id is required"}), 400
    success = flow_engine.switch_account(service="flow", account_id=account_id)
    if not success:
        return jsonify({"error": "Flow account not found"}), 404
    client = flow_engine.FlowClient()
    auth = client.check_auth()
    return jsonify({
        "status": "success",
        "service": "flow",
        "active_account": account_id,
        "message": f"Switched to Flow account {account_id}",
        "auth": auth
    })


@app.route("/api/flow/accounts/add", methods=["POST"])
def api_flow_accounts_add():
    data = request.get_json(force=True, silent=True) or {}
    account_id = data.get("id") or request.form.get("id") or ""
    name = data.get("name") or request.form.get("name") or ""
    email = data.get("email") or request.form.get("email") or ""
    raw_cookies = data.get("cookies") or request.form.get("cookies")
    if not raw_cookies:
        return jsonify({"error": "Cookies are required"}), 400

    parsed = raw_cookies if isinstance(raw_cookies, (dict, list)) else flow_engine.parse_cookie_string(str(raw_cookies))
    if not parsed:
        return jsonify({"error": "Could not parse Flow cookies"}), 400

    res = flow_engine.add_or_update_account(service="flow", account_id=account_id, name=name, cookies=parsed, email=email)
    return jsonify({"status": "success", "service": "flow", "result": res})


@app.route("/api/flow/accounts/delete", methods=["POST"])
def api_flow_accounts_delete():
    data = request.get_json(force=True, silent=True) or {}
    account_id = data.get("id") or request.form.get("id")
    if not account_id:
        return jsonify({"error": "Account id is required"}), 400
    success = flow_engine.delete_account(service="flow", account_id=account_id)
    if not success:
        return jsonify({"error": "Flow account not found"}), 404
    return jsonify({"status": "success", "service": "flow", "message": f"Flow account {account_id} deleted"})


@app.route("/api/flow/auth/sync", methods=["POST"])
def api_flow_auth_sync():
    data = request.get_json(force=True, silent=True) or {}
    raw_cookies = data.get("cookies") or request.form.get("cookies")
    if not raw_cookies and isinstance(data, dict):
        raw_cookies = data
    if not raw_cookies:
        return jsonify({"error": "No cookies provided."}), 400

    parsed = raw_cookies if isinstance(raw_cookies, (dict, list)) else flow_engine.parse_cookie_string(str(raw_cookies))
    if not parsed:
        return jsonify({"error": "Could not parse Flow cookies."}), 400

    flow_engine.save_cookies(parsed, service="flow")
    flow_engine.add_or_update_account(service="flow", account_id="flow_default", name="Primary Google Flow Account", cookies=parsed)

    client = flow_engine.FlowClient()
    auth_res = client.check_auth()
    return jsonify({
        "status": "success",
        "service": "flow",
        "message": "Google Flow cookies synchronized successfully!",
        "auth": auth_res
    })


@app.route("/api/flow/auth/status", methods=["GET"])
def api_flow_auth_status():
    client = flow_engine.FlowClient()
    return jsonify(client.check_auth())


# Legacy aliases for backward compatibility
@app.route("/api/auth/sync", methods=["POST"])
def api_auth_sync_legacy():
    data = request.get_json(force=True, silent=True) or {}
    service = data.get("service") or request.form.get("service") or "gemini"
    if service == "flow":
        return api_flow_auth_sync()
    return api_gemini_auth_sync()


@app.route("/api/auth/status", methods=["GET"])
def api_auth_status_legacy():
    g = api_gemini_auth_status().get_json()
    f = api_flow_auth_status().get_json()
    return jsonify({
        "authenticated": g.get("authenticated") or f.get("authenticated"),
        "gemini": g,
        "flow": f,
        "cookies_count": g.get("cookies_count", 0) + f.get("cookies_count", 0),
        "detail": f"Gemini: {g.get('detail')} | Flow: {f.get('detail')}"
    })


@app.route("/api/accounts", methods=["GET"])
def api_accounts_list_legacy():
    return jsonify({
        "gemini": flow_engine.list_accounts("gemini"),
        "flow": flow_engine.list_accounts("flow")
    })


# ---------------------------------------------------------------------------
# OpenAI Models API
# ---------------------------------------------------------------------------
@app.route("/v1/models", methods=["GET"])
@app.route("/models", methods=["GET"])
def list_models():
    models = [
        {
            "id": "gemini-4-argon",
            "object": "model",
            "created": 1794600000,
            "owned_by": "google",
            "display_name": "⚡ Gemini 4 Argon (Reasoning & Deep Logic)",
            "description": "Google Frontier Reasoning Model. Deep multi-step architecture, long-horizon logic, complex algorithms.",
            "supports_reasoning": True,
            "context_window_tokens": 1000000,
            "max_output_tokens": 64000,
        },
        {
            "id": "gemini-4-hydrogen",
            "object": "model",
            "created": 1794600000,
            "owned_by": "google",
            "display_name": "🚀 Gemini 4 Hydrogen (Fast Coding & Execution)",
            "description": "Google High-Speed Execution Model. Fast token streaming, daily coding, rapid iterations & tools.",
            "supports_reasoning": False,
            "context_window_tokens": 1000000,
            "max_output_tokens": 64000,
        },
        {
            "id": "nano-banana-pro",
            "object": "model",
            "created": 1794600000,
            "owned_by": "google",
            "display_name": "🍌 Nano Banana Pro (Google Image)",
            "description": "Google's Nano Banana Pro image model via the Gemini web session. Not a chat model — use POST /generate-image (new image) or POST /edit-image-google (edit an image); both return an async job, poll GET /edit-status/<job>.",
            "supports_reasoning": False,
            "supports_image_generation": True,
            "context_window_tokens": 1000000,
            "max_output_tokens": 64000,
        },
        {
            "id": "argon",
            "object": "model",
            "created": 1794600000,
            "owned_by": "google",
            "display_name": "⚡ Argon (Alias for Gemini 4 Argon)",
            "description": "Frontier Reasoning alias",
            "supports_reasoning": True,
        },
        {
            "id": "hydroide",
            "object": "model",
            "created": 1794600000,
            "owned_by": "google",
            "display_name": "🚀 Hydroide (Alias for Gemini 4 Hydrogen)",
            "description": "Fast Execution alias",
            "supports_reasoning": False,
        },
        {
            "id": "claude-opus-5-5",
            "object": "model",
            "created": 1794600000,
            "owned_by": "anthropic",
            "display_name": "Claude Opus 5.5 (Powered by Gemini 4 Argon)",
            "description": "Routed to Gemini 4 Argon Frontier Reasoning",
            "supports_reasoning": True,
        },
        {
            "id": "claude-3-7-sonnet-20250219",
            "object": "model",
            "created": 1794600000,
            "owned_by": "anthropic",
            "display_name": "Claude 3.7 Sonnet (Powered by Gemini 4 Hydrogen)",
            "description": "Routed to Gemini 4 Hydrogen High-Speed Coding",
            "supports_reasoning": True,
        },
        {
            "id": "claude-3-5-sonnet-20241022",
            "object": "model",
            "created": 1794600000,
            "owned_by": "anthropic",
            "display_name": "Claude 3.5 Sonnet (Powered by Gemini 4 Hydrogen)",
            "description": "Routed to Gemini 4 Hydrogen High-Speed Coding",
            "supports_reasoning": False,
        },
        {
            "id": "claude-3-5-haiku-20241022",
            "object": "model",
            "created": 1794600000,
            "owned_by": "anthropic",
            "display_name": "Claude 3.5 Haiku (Powered by Gemini 4 Hydrogen)",
            "description": "Routed to Gemini 4 Hydrogen High-Speed Coding",
            "supports_reasoning": False,
        },
        {
            "id": "claude-3-opus-20240229",
            "object": "model",
            "created": 1794600000,
            "owned_by": "anthropic",
            "display_name": "Claude 3 Opus (Powered by Gemini 4 Argon)",
            "description": "Routed to Gemini 4 Argon Frontier Reasoning",
            "supports_reasoning": True,
        },
    ]
    for m in models:
        m.setdefault("type", "model")
        m.setdefault("created_at", "2026-01-01T00:00:00Z")
    return jsonify({
        "object": "list",
        "data": models,
        "default_model_id": "gemini-4-argon",
        "first_id": "gemini-4-argon",
        "last_id": models[-1]["id"],
        "has_more": False,
    })


@app.route("/health", methods=["GET"])
@app.route("/api/v1/health", methods=["GET"])
def health_check():
    cookies = load_cookies()
    has_psid = bool(cookies.get("__Secure-1PSID"))
    has_flow = bool(cookies.get("OSID") or cookies.get("__Secure-OSID") or has_psid)
    gemini_ok = _LAST_ACCOUNT_STATUS == 1000
    # Flow auth is checked at most once every 120s (it is a live request to
    # flow.google.com). A failed/expired Flow session is reported, not hidden.
    flow_ok = _cached_flow_auth()
    healthy = gemini_ok or flow_ok
    return jsonify({
        "status": "healthy" if healthy else "degraded",
        "account_status": _LAST_ACCOUNT_STATUS,
        "gemini_authenticated": gemini_ok,
        "flow_authenticated": flow_ok,
        "gemini_client_ready": _ACTIVE_CLIENT is not None,
        "cookies_configured": has_psid,
        "google_flow_ready": has_flow,
        "auth_required": bool(API_KEY),
        "auth_error": None if healthy else "Google session expired — re-sync cookies from gemini.google.com / flow.google.com via the 🔑 button in chat.",
        "max_concurrency": MAX_CONCURRENCY,
        "uptime_seconds": int(time.time() - START_TIME),
        "models": ["gemini-4-argon", "gemini-4-hydrogen", "google-flow-veo", "google-flow-imagen"],
    })


_FLOW_AUTH_CACHE = {"at": 0.0, "ok": False}


def _cached_flow_auth() -> bool:
    """Cached Flow session check (max 1 live check per 120s)."""
    now = time.time()
    if now - _FLOW_AUTH_CACHE["at"] < 120:
        return _FLOW_AUTH_CACHE["ok"]
    ok = False
    try:
        res = flow_engine.FlowClient().check_auth()
        ok = bool(res.get("authenticated"))
    except Exception as e:
        logger.warning("Flow auth check failed: %s", e)
    _FLOW_AUTH_CACHE.update({"at": now, "ok": ok})
    return ok


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({"tool_calling": dict(adapter.STATS), "uptime_seconds": int(time.time() - START_TIME)})


@app.route("/api/event_logging/v2/batch", methods=["POST", "OPTIONS"])
def event_logging():
    return jsonify({"status": "ok", "logged": True})


# ---------------------------------------------------------------------------
# Public API v1 — one clean namespace for everything the chat UI can do.
# Every feature has an API: chat, image, edit (Azad + Google), video,
# auth sync and health. Async jobs return {"job": "<id>"}; poll the
# matching status endpoint (or the unified /api/v1/jobs/<job>) until
# status is "done" or "error". Full docs: GET /api/docs (HTML) and
# GET /api/openapi.json. No API key is required unless the server was
# started with GAYMINI_API_KEY (then send x-api-key on every call).
# ---------------------------------------------------------------------------
API_V1_DOCS = [
    {"method": "GET", "path": "/api/v1/health", "desc": "Server + Google session status (account_status 1000 = OK, 1016 = cookies expired)."},
    {"method": "POST", "path": "/api/v1/chat", "desc": "Chat (OpenAI-compatible). JSON {model: 'gemini-4-argon'|'gemini-4-hydrogen', messages: [{role, content}]} → chat completion."},
    {"method": "POST", "path": "/api/v1/image", "desc": "New image (Nano Banana Pro; model=flow tries Flow Imagen first). Form {prompt, model?, origin} → {job}. Poll /api/v1/image-status/<job>."},
    {"method": "POST", "path": "/api/v1/edit", "desc": "Edit an uploaded image with the Azad editor (works without Google cookies). Multipart {image, prompt, origin} → {job}. Poll /api/v1/image-status/<job>."},
    {"method": "POST", "path": "/api/v1/edit-google", "desc": "Edit an uploaded image with Gemini/Nano Banana Pro (needs a valid Google session; falls back to the Azad editor automatically). Multipart {image, prompt, origin} → {job}."},
    {"method": "POST", "path": "/api/v1/video", "desc": "Video. Form {prompt, origin} (+ optional image file) → {job}. Text-to-video makes a keyframe with Nano Banana Pro, then an 8s 1080p motion video with music. Poll /api/v1/video-status/<job>. NOTE: motion video, not Veo — Veo with native audio needs a live Flow browser tab."},
    {"method": "GET", "path": "/api/v1/image-status/<job>", "desc": "Image/edit job status → {status: working|done|error, image_url?, message?}."},
    {"method": "GET", "path": "/api/v1/video-status/<job>", "desc": "Video job status → {status: working|done|error, video_url?, error?}."},
    {"method": "GET", "path": "/api/v1/jobs/<job>", "desc": "Unified job status — checks image/edit jobs first, then video jobs."},
    {"method": "POST", "path": "/api/gemini/auth/sync", "desc": "Sync fresh Gemini cookies. JSON {cookies: [...] or {name: value}} → new session."},
    {"method": "POST", "path": "/api/flow/auth/sync", "desc": "Sync fresh Flow cookies. Same format as above."},
    {"method": "GET", "path": "/api/gemini/auth/status", "desc": "Gemini auth status."},
    {"method": "GET", "path": "/api/flow/auth/status", "desc": "Flow auth status."},
    {"method": "GET", "path": "/api/docs", "desc": "This documentation (HTML)."},
    {"method": "GET", "path": "/api/openapi.json", "desc": "Machine-readable API spec (JSON)."},
]


@app.route("/api/v1/jobs/<job_id>", methods=["GET"])
def api_job_status(job_id):
    job = _EDIT_JOBS.get(job_id)
    if job is not None:
        return jsonify(job)
    job = flow_engine.get_video_job(job_id)
    if job is not None:
        return jsonify(job)
    return jsonify({"error": "job not found"}), 404


@app.route("/api/v1", methods=["GET"])
def api_v1_index():
    return jsonify({"name": "Gaymini Pro API", "version": "v1",
                    "docs": "/api/docs", "openapi": "/api/openapi.json",
                    "endpoints": API_V1_DOCS})


@app.route("/api/openapi.json", methods=["GET"])
def api_openapi():
    paths = {}
    for ep in API_V1_DOCS:
        if ep["path"].startswith("/api/openapi") or ep["path"].startswith("/api/docs"):
            continue
        paths.setdefault(ep["path"], {})[ep["method"].lower()] = {
            "summary": ep["desc"], "responses": {"200": {"description": "OK"}}}
    return jsonify({"openapi": "3.0.0",
                    "info": {"title": "Gaymini Pro API", "version": "1.0.0"},
                    "paths": paths})


@app.route("/api/docs", methods=["GET"])
def api_docs():
    rows = "".join(
        f"<tr><td><b>{e['method']}</b></td><td><code>{e['path']}</code></td><td>{e['desc']}</td></tr>"
        for e in API_V1_DOCS)
    return f"""<!DOCTYPE html><html lang="bn"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gaymini Pro API — Docs</title>
<style>body{{font-family:system-ui,sans-serif;background:#0f0a1e;color:#eee;margin:0;padding:24px;max-width:900px}}
h1{{color:#c084fc}}h2{{color:#c084fc;margin-top:32px}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #444;padding:8px 10px;text-align:left;vertical-align:top}}th{{background:#2a1845}}
code{{background:#241a38;padding:2px 6px;border-radius:4px}}pre{{background:#1a1226;padding:14px;border-radius:8px;overflow:auto}}
a{{color:#c084fc}}</style></head><body>
<h1>Gaymini Pro API</h1>
<p>চ্যাট, ছবি, এডিট, ভিডিও — সব ফিচারের API। Async job গুলো <code>{{"job": "…"}}</code> দেয়;
status <code>done</code> না হওয়া পর্যন্ত সংশ্লিষ্ট status endpoint-এ poll করো।
API key লাগে না, যদি না সার্ভার <code>GAYMINI_API_KEY</code> দিয়ে চালু করা হয় (তখন প্রতিটা কলে <code>x-api-key</code> হেডার দাও)।</p>
<h2>Endpoints</h2>
<table><tr><th>Method</th><th>Path</th><th>কী করে</th></tr>{rows}</table>
<h2>উদাহরণ</h2>
<pre>curl -X POST $BASE/api/v1/chat -H 'Content-Type: application/json' \\
  -d '{{"model":"gemini-4-argon","messages":[{{"role":"user","content":"hi"}}]}}'

curl -X POST $BASE/api/v1/image -F prompt="a cute orange cat" -F origin=$BASE
# → {{"job":"…"}} তারপর:
curl $BASE/api/v1/image-status/&lt;job&gt;

curl -X POST $BASE/api/v1/video -F prompt="a cat in rain" -F origin=$BASE
curl $BASE/api/v1/video-status/&lt;job&gt;

curl -X POST $BASE/api/v1/edit -F image=@photo.jpg -F prompt="make it sunset" -F origin=$BASE</pre>
<p>ভিডিও এখন ৮ সেকেন্ড, 1080p, মিউজিকসহ মোশন ভিডিও (Veo নয় — আসল Veo-তে নিজের
অডিওসহ ভিডিও বানাতে লাইভ Flow ব্রাউজার ট্যাব বা অফিসিয়াল API key লাগে)।</p>
</body></html>"""


@app.route("/v1/messages/count_tokens", methods=["POST", "OPTIONS"])
def count_tokens():
    data = request.get_json(force=True, silent=True) or {}
    try:
        prompt = adapter.build_prompt(
            data.get("system"), data.get("messages") or [], data.get("tools") or [], data.get("tool_choice")
        )
        return jsonify({"input_tokens": adapter.estimate_tokens(prompt)})
    except Exception:  # noqa: BLE001
        chars = sum(len(str(m.get("content", ""))) for m in (data.get("messages") or []) if isinstance(m, dict))
        return jsonify({"input_tokens": max(1, chars // 4)})


# ---------------------------------------------------------------------------
# OpenAI Chat Completions API (/v1/chat/completions)
# ---------------------------------------------------------------------------
@app.route("/v1/chat/completions", methods=["POST"])
@app.route("/chat/completions", methods=["POST"])
@app.route("/api/v1/chat", methods=["POST"])
def chat_completions():
    data = request.get_json(force=True, silent=True) or {}
    messages = data.get("messages", [])
    model_req = data.get("model", "gemini-4-argon")
    stream = data.get("stream", False)

    if not messages:
        return jsonify({"error": {"message": "messages list required", "type": "invalid_request_error"}}), 400

    canonical_model, backend_model = resolve_model_target(model_req)
    echo_model = model_req or canonical_model

    system_prompt = ""
    non_system_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            system_prompt += "\n" + _extract_text_from_message(msg)
        else:
            non_system_messages.append(msg)

    prompt = format_messages_to_prompt(non_system_messages, system_prompt=system_prompt.strip())

    try:
        client = _run_coroutine_sync(get_gemini_client())
    except Exception as e:
        logger.error(f"Client init error: {e}")
        return jsonify({"error": {"message": f"Auth / Client init error: {str(e)}", "type": "api_error"}}), 500

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created_ts = int(time.time())

    # STREAMING SSE
    if stream:
        def sse_generator():
            q = queue.Queue()
            sentinel = object()

            async def _stream_worker():
                try:
                    accumulated = ""

                    init_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": echo_model,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                    }
                    q.put(f"data: {json.dumps(init_chunk)}\n\n")

                    async for chunk in client.generate_content_stream(prompt, model=backend_model, temporary=True):
                        delta = chunk.text_delta if hasattr(chunk, "text_delta") and chunk.text_delta else ""
                        if not delta and chunk.text:
                            if len(chunk.text) > len(accumulated):
                                delta = chunk.text[len(accumulated):]

                        if not delta:
                            continue

                        delta = _clean_reply_text(delta)
                        if not delta:
                            continue

                        accumulated += delta

                        payload = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": echo_model,
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                        }
                        q.put(f"data: {json.dumps(payload)}\n\n")

                    end_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": echo_model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    q.put(f"data: {json.dumps(end_chunk)}\n\n")
                    q.put("data: [DONE]\n\n")
                except Exception as ex:
                    logger.error(f"Stream error: {ex}")
                    q.put(ex)
                finally:
                    q.put(sentinel)

            loop = _ensure_event_loop()
            asyncio.run_coroutine_threadsafe(_stream_worker(), loop)

            while True:
                try:
                    item = q.get(timeout=2.0)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue

                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    err_payload = {"error": {"message": str(item), "type": "api_error"}}
                    yield f"data: {json.dumps(err_payload)}\n\n"
                    break
                yield item

        return Response(
            stream_with_context(sse_generator()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # NON-STREAMING JSON
    async def _non_stream_worker():
        resp = await client.generate_content(prompt, model=backend_model, temporary=True)
        return resp.text or ""

    try:
        output_text = _clean_reply_text(_run_coroutine_sync(_non_stream_worker()))
    except Exception as e:
        logger.error(f"Non-stream error: {e}")
        return jsonify({"error": {"message": str(e), "type": "api_error"}}), 500

    prompt_tokens = max(10, len(prompt) // 4)
    completion_tokens = max(5, len(output_text) // 4)

    return jsonify({
        "id": chat_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": echo_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })


# ---------------------------------------------------------------------------
# Anthropic Messages API (/v1/messages)
# ---------------------------------------------------------------------------
async def _gemini_deltas(backend_model, prompt: str, tries: int = 3):
    """Yield text deltas from Gemini. Retries transient failures that happen before any output."""
    last_exc = None
    for attempt in range(tries):
        got_any = False
        try:
            client = await get_gemini_client()
            acc = ""
            async for chunk in client.generate_content_stream(prompt, model=backend_model, temporary=True):
                delta = getattr(chunk, "text_delta", None) or ""
                if not delta and getattr(chunk, "text", None) and len(chunk.text) > len(acc):
                    delta = chunk.text[len(acc):]
                if delta:
                    delta = _clean_reply_text(delta)
                    if not delta:
                        continue
                    acc += delta
                    got_any = True
                    yield delta
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if got_any or attempt == tries - 1:
                raise
            logger.warning(f"Gemini stream failed before output ({exc}); retry {attempt + 1}/{tries - 1}")
            await asyncio.sleep(1.5 * (attempt + 1))
    if last_exc:
        raise last_exc


START_TIME = time.time()
MAX_CONCURRENCY = max(1, int(os.environ.get("GAYMINI_MAX_CONCURRENCY", "2")))
QUEUE_TIMEOUT = float(os.environ.get("GAYMINI_QUEUE_TIMEOUT", "120"))
_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENCY)


def classify_error(exc: BaseException) -> Tuple[str, int, str]:
    """Map a backend failure to (anthropic error type, http status, message)."""
    msg = str(exc) or exc.__class__.__name__
    low = msg.lower()
    if "__secure-1psid" in low or any(k in low for k in ("authenticat", "login", "unauthorized", "sign in", "cookie")) \
            and any(k in low for k in ("expired", "invalid", "missing", "refresh", "failed", "unauthor", "login")):
        return ("authentication_error", 401,
                f"Gemini session cookies are missing/expired - refresh gaymini.txt / cookies.json. ({msg})")
    if any(k in low for k in ("429", "quota", "usage limit", "rate limit", "too many")):
        return "rate_limit_error", 429, msg
    if any(k in low for k in ("1013", "temporary", "overload", "unavailable", "busy", "503", "502")):
        return "overloaded_error", 529, msg
    if "timeout" in low or "timed out" in low:
        return "api_error", 504, msg
    return "api_error", 500, msg


def _anthropic_error(message: str, etype: str = "api_error", status: int = 500):
    return jsonify(adapter.error_body(message, etype)), status


@app.after_request
def add_api_headers(response):
    response.headers.setdefault("request-id", adapter.new_id("req"))
    response.headers.setdefault("anthropic-version", "2023-06-01")
    return response


@app.route("/v1/messages", methods=["POST"])
def anthropic_messages():
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return _anthropic_error("request body must be a JSON object", "invalid_request_error", 400)
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return _anthropic_error("messages: at least one message is required", "invalid_request_error", 400)

    tools = data.get("tools") or []
    tool_choice = data.get("tool_choice")
    stream = bool(data.get("stream", False))
    model_req = data.get("model") or "gemini-4-argon"
    stop_sequences = data.get("stop_sequences") or []
    max_tokens = data.get("max_tokens")
    _canonical, backend_model = resolve_model_target(model_req)

    prompt = adapter.build_prompt(data.get("system"), messages, tools, tool_choice)
    in_tokens = adapter.estimate_tokens(prompt)
    msg_id = adapter.new_id("msg")
    t0 = time.time()

    if not _SLOTS.acquire(timeout=QUEUE_TIMEOUT):
        resp = _anthropic_error("gateway is busy, try again shortly", "rate_limit_error", 429)
        resp[0].headers["retry-after"] = "5"
        return resp

    def log_done(status: str, stop_reason: str = "-"):
        logger.info(
            f"{msg_id} model={model_req} backend={backend_model.model_name} stream={stream} tools={len(tools)} "
            f"prompt_chars={len(prompt)} status={status} stop={stop_reason} {time.time() - t0:.1f}s"
        )

    async def _events():
        async def stream_fn(p):
            async for d in _gemini_deltas(backend_model, p):
                yield d
        inner = adapter.run_turn(stream_fn, prompt, tools, tool_choice)
        async for ev in adapter.apply_limits(inner, stop_sequences, max_tokens):
            yield ev

    if not stream:
        async def _collect():
            return [ev async for ev in _events()]
        try:
            events = asyncio.run_coroutine_threadsafe(_collect(), _ensure_event_loop()).result(timeout=900)
        except Exception as e:  # noqa: BLE001
            etype, status, msg = classify_error(e)
            logger.error(f"{msg_id} failed: {e}")
            log_done(etype)
            return _anthropic_error(msg, etype, status)
        finally:
            _SLOTS.release()
        body = adapter.build_message_json(events, msg_id, model_req, in_tokens)
        log_done("ok", body["stop_reason"])
        return jsonify(body)

    def sse():
        q: "queue.Queue" = queue.Queue()
        done = object()

        async def producer():
            w = adapter.SSEWriter(msg_id, model_req, in_tokens)
            last_stop = "-"
            try:
                for f in w.start():
                    q.put(f)
                async for kind, val in _events():
                    if kind == "text":
                        frames = w.text(val)
                    elif kind == "tool":
                        frames = w.tool(val)
                    else:
                        last_stop = val["stop_reason"]
                        frames = w.finish(val["stop_reason"], val["output_chars"], val.get("stop_sequence"))
                    for f in frames:
                        q.put(f)
                log_done("ok", last_stop)
            except Exception as e:  # noqa: BLE001
                etype, _status, msg = classify_error(e)
                logger.error(f"{msg_id} stream failed: {e}")
                q.put(adapter.SSEWriter.error(msg, etype))
                log_done(etype)
            finally:
                _SLOTS.release()
                q.put(done)

        asyncio.run_coroutine_threadsafe(producer(), _ensure_event_loop())
        while True:
            try:
                item = q.get(timeout=10.0)
            except queue.Empty:
                yield "event: ping\ndata: {\"type\": \"ping\"}\n\n"
                continue
            if item is done:
                break
            yield item

    return Response(
        stream_with_context(sse()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _warmup_background():
    time.sleep(1.0)
    try:
        _run_coroutine_sync(get_gemini_client())
        logger.info("[✓] GeminiClient pre-warmed & cached in memory. Ready for instant responses!")
    except Exception as e:
        logger.warning(f"Pre-warm background error: {e}")

threading.Thread(target=_warmup_background, daemon=True, name="WarmupThread").start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Gemini 4 Argon & Hydrogen API Server on http://0.0.0.0:{port}")
    host = os.environ.get("HOST", "0.0.0.0")
    try:
        from waitress import serve  # production WSGI server (streams SSE correctly with send_bytes=1)

        logger.info(f"Serving with waitress on http://{host}:{port}")
        serve(app, host=host, port=port, threads=int(os.environ.get("GAYMINI_THREADS", "16")),
              channel_timeout=900, send_bytes=1, ident="gaymini")
    except ImportError:
        logger.warning("waitress not installed - falling back to Flask dev server (pip install waitress)")
        app.run(host=host, port=port, debug=False, threaded=True)
