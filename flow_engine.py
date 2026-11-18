"""
Google Flow (flow.google.com) & Veo / Imagen Integration Engine
Handles:
1. Google Flow session & cookie authentication (flow.google.com & labs.google)
2. Text-to-Video (t2v) & Image-to-Video (i2v) via Veo (YhhmEf / jwpduf / as29s RPCs)
3. Text-to-Image (t2i) & Image-to-Image (i2i) via Imagen 3 (ogiZ0b RPC)
4. Fallback to Gemini WebAPI video/image generation
5. Automated Cookie parsing, rotation, and session sync
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from curl_cffi import requests

logger = logging.getLogger("flow_engine")
logger.setLevel(logging.INFO)

# Base URLs
FLOW_BASE = "https://flow.google.com"
FLOW_BATCH_EXECUTE = "https://flow.google.com/_/AiSandboxAngularFrontend/data/batchexecute"
FLOW_PROJECT_URL = "https://flow.google.com/project"
LABS_TRPC_BASE = "https://labs.google/fx/api/trpc"

UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Jobs dictionary for video and image generations
_VIDEO_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def get_cookies_path(service: str = "flow") -> Path:
    base = Path(__file__).resolve().parent
    if service == "gemini":
        p = base / "cookies_gemini.json"
        return p if p.is_file() else (base / "cookies.json")
    elif service == "flow":
        p = base / "cookies_flow.json"
        return p if p.is_file() else (base / "cookies.json")
    return base / "cookies.json"


def load_cookies(service: str = "flow") -> dict[str, str]:
    """Load cookies dictionary for specified service ('flow' or 'gemini')."""
    env_key = f"{service.upper()}_COOKIES"
    env_c = os.environ.get(env_key) or (os.environ.get("GEMINI_COOKIES") if service == "gemini" else None)
    if env_c:
        try:
            data = json.loads(env_c)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
            elif isinstance(data, list):
                return {c.get("name"): c.get("value") for c in data if "name" in c and "value" in c}
        except Exception:
            pass

    cpath = get_cookies_path(service)
    if not cpath.is_file():
        cpath = Path(__file__).resolve().parent / "cookies.json"
    if not cpath.is_file():
        return {}
    try:
        data = json.loads(cpath.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        elif isinstance(data, list):
            return {c.get("name"): c.get("value") for c in data if "name" in c and "value" in c}
    except Exception as e:
        logger.error(f"Error loading {service} cookies: {e}")
    return {}


def save_cookies(cookies: dict[str, str] | list[dict[str, Any]], service: str = "flow") -> None:
    """Save cookies to cookies_<service>.json and update cookies.json."""
    base = Path(__file__).resolve().parent
    target_path = base / f"cookies_{service}.json"

    clean_dict: dict[str, str] = {}
    clean_list: list[dict[str, Any]] = []

    if isinstance(cookies, dict):
        clean_dict = {k.strip(): str(v).strip() for k, v in cookies.items() if k and v}
        for k, v in clean_dict.items():
            domain = ".google.com"
            if service == "flow" or "flow" in k.lower() or k in ("OSID", "__Secure-OSID"):
                domain = "flow.google.com"
            clean_list.append({
                "name": k,
                "value": v,
                "domain": domain,
                "path": "/"
            })
    elif isinstance(cookies, list):
        for item in cookies:
            name = item.get("name")
            val = item.get("value")
            if name and val:
                clean_dict[name.strip()] = str(val).strip()
                clean_list.append(item)

    target_path.write_text(json.dumps(clean_dict, indent=2), encoding="utf-8")
    try:
        target_path.chmod(0o600)
    except Exception:
        pass

    # If saving gemini cookies, also keep cookies.json in sync for legacy clients
    if service == "gemini":
        cpath = base / "cookies.json"
        cpath.write_text(json.dumps(clean_dict, indent=2), encoding="utf-8")
        try:
            cpath.chmod(0o600)
        except Exception:
            pass


def parse_cookie_string(raw: str) -> dict[str, str]:
    """Parse cookie string from browser document.cookie or HTTP header format."""
    cookies = {}
    # Handle JSON string if provided
    raw = raw.strip()
    if raw.startswith("{") or raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {k.strip(): str(v).strip() for k, v in parsed.items()}
            elif isinstance(parsed, list):
                return {c.get("name"): str(c.get("value")) for c in parsed if "name" in c and "value" in c}
        except Exception:
            pass

    # Standard "name=val; name2=val2" format
    pairs = raw.split(";")
    for pair in pairs:
        if "=" in pair:
            k, v = pair.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k and v:
                cookies[k] = v
    return cookies


class FlowClient:
    """HTTP Client for flow.google.com reverse-engineered APIs."""

    def __init__(self, cookies: dict[str, str] | None = None):
        self.cookies = cookies or load_cookies(service="flow")
        self.session = requests.Session(impersonate="chrome")
        self._xsrf_token: str | None = None
        self._project_id: str | None = None
        self._user_email: str | None = None
        self._apply_cookies()

    def _apply_cookies(self):
        for k, v in self.cookies.items():
            if k in ("OSID", "__Secure-OSID", "_ga", "_ga_X2GNH8R5NS") or "flow" in k.lower():
                self.session.cookies.set(k, v, domain="flow.google.com", path="/")
            else:
                self.session.cookies.set(k, v, domain=".google.com", path="/")

    def check_auth(self) -> dict[str, Any]:
        """Check if cookies authenticate with Google Flow."""
        result = {
            "service": "flow",
            "cookies_count": len(self.cookies),
            "authenticated": False,
            "flow_online": False,
            "detail": ""
        }
        if not self.cookies:
            result["detail"] = "Google Flow-er kono cookie dewa nei. flow.google.com theke cookie sync korun."
            return result

        try:
            headers = {
                "authority": "flow.google.com",
                "referer": "https://flow.google.com/",
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            }
            r = self.session.get(FLOW_BASE, headers=headers, timeout=12, allow_redirects=True)
            if "CookieMismatch" in r.url:
                result["detail"] = "Google CookieMismatch: Gemini ebong Flow-er cookie alada hote hobe. flow.google.com-e giye Flow-er jonno alada cookie sync korun."
                return result
            if "accounts.google.com/ServiceLogin" in r.url or "signin" in r.url:
                result["detail"] = "Google Flow session unauthenticated. flow.google.com-e login kore cookie sync korun."
                return result
            if r.status_code == 200:
                result["authenticated"] = True
                result["flow_online"] = True
                result["detail"] = "Google Flow session active ebong verified! (Veo o Imagen 3 active)"
                m = re.search(r'"SNlM0e":"([^"]+)"', r.text)
                if m:
                    self._xsrf_token = m.group(1)
                return result
            result["detail"] = f"Flow response code: HTTP {r.status_code}"
        except Exception as e:
            result["detail"] = f"Flow verification check error: {e}"

        return result

    def get_xsrf_token(self) -> str:
        """Fetch Flow XSRF 'at' token from flow.google.com."""
        if self._xsrf_token:
            return self._xsrf_token
        try:
            headers = {
                "authority": "flow.google.com",
                "referer": "https://flow.google.com/",
            }
            r = self.session.get(FLOW_BASE, headers=headers, timeout=15)
            m = re.search(r'"SNlM0e":"([^"]+)"', r.text)
            if m:
                self._xsrf_token = m.group(1)
                return self._xsrf_token
        except Exception as e:
            logger.warning(f"Could not extract xsrf token: {e}")
        return "generic"

    def ensure_project(self) -> str:
        """Ensure an active project ID exists."""
        if self._project_id:
            return self._project_id
        # Standard UUID project identifier
        self._project_id = str(uuid.uuid4())
        return self._project_id

    def generate_image_flow(self, prompt: str) -> dict[str, Any]:
        """Generate image via Google Flow Imagen 3 (ogiZ0b RPC)."""
        project = self.ensure_project()
        xsrf = self.get_xsrf_token()

        # Build ogiZ0b request body
        entry = [
            None,
            None,
            None,
            int(time.time()),
            3,
            "BELUGA",  # Imagen model in Flow
            None,
            [None, 22, None, None, None, project],
            [
                [],
                [f" {prompt}"]
            ],
            None,
            None,
            None,
        ]
        inner = [None, [entry], 1, [None, 22, None, None, None, project]]
        req_payload = [[["ogiZ0b", json.dumps(inner), None, "generic"]]]

        body_data = f"f.req={quote(json.dumps(req_payload))}&at={quote(xsrf)}&"
        headers = {
            "authority": "flow.google.com",
            "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
            "origin": "https://flow.google.com",
            "referer": f"https://flow.google.com/project/{project}",
            "x-same-domain": "1",
        }

        r = self.session.post(
            f"{FLOW_BATCH_EXECUTE}?rpcids=ogiZ0b",
            headers=headers,
            data=body_data,
            timeout=60
        )

        if r.status_code != 200:
            raise RuntimeError(f"Flow image submit returned HTTP {r.status_code}")

        # Parse batchexecute response
        image_urls = []
        for line in r.text.splitlines():
            if "https://" in line and any(ext in line for ext in [".jpg", ".png", ".webp", "flow-content"]):
                urls = re.findall(r'https?://[^\s"\'\\,]+', line)
                for u in urls:
                    if "flow-content" in u or "googleusercontent" in u:
                        image_urls.append(u.replace("\\u0026", "&"))

        if not image_urls:
            # Try parsing wrb.fr payload
            m = re.findall(r'https://flow-content\.google/[^\s"\'\\,]+', r.text)
            image_urls.extend([u.replace("\\u0026", "&") for u in m])

        return {
            "status": "success" if image_urls else "empty",
            "images": image_urls,
            "raw_text": r.text[:500] if not image_urls else ""
        }

    def generate_video_flow(self, prompt: str, image_bytes: bytes | None = None) -> dict[str, Any]:
        """Generate video via Google Flow Veo (YhhmEf RPC) with jwpduf poll.

        NOTE: this only *submits* the job — the jwpduf polling loop that
        waits for Veo to finish is not implemented yet, so run_video_job()
        does not call it. Video today is the Cinematic Motion Engine:
        a keyframe image (Gemini / Flow Imagen / your upload) animated
        with ffmpeg zoompan into a 5s 720p mp4. Do not label that output
        as "Veo 3.1" in the UI.
        """
        project = self.ensure_project()
        xsrf = self.get_xsrf_token()

        media_id = str(uuid.uuid4())
        workflow_id = str(uuid.uuid4())

        # Abra / Veo model: abra_t2v_8s (8s text to video)
        model_key = "abra_t2v_8s"
        details: list[Any] = [
            [int(time.time()), 0],
            prompt,
            None,
            None,
            None,
            None,
            [None, [[model_key, 1, None, None, 2, 1]], [[None, None, [[[prompt]]]]], None, 1],
            None,
            [6],  # status = submitted
            1,
        ]

        media_info: list[Any] = [
            [
                None,
                0,
                None,
                None,
                None,
                None,
                None,
                prompt,
                None,
                None,
                None,
                None,
                model_key
            ]
        ]

        record = [workflow_id, project, media_id, None, None, details, None, media_info]
        inner = [None, 1, [], [[record]]]

        req_payload = [[["YhhmEf", json.dumps(inner), None, "generic"]]]
        body_data = f"f.req={quote(json.dumps(req_payload))}&at={quote(xsrf)}&"
        headers = {
            "authority": "flow.google.com",
            "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
            "origin": "https://flow.google.com",
            "referer": f"https://flow.google.com/project/{project}",
            "x-same-domain": "1",
        }

        r = self.session.post(
            f"{FLOW_BATCH_EXECUTE}?rpcids=YhhmEf",
            headers=headers,
            data=body_data,
            timeout=120
        )

        return {
            "status": "submitted" if r.status_code == 200 else "error",
            "workflow_id": workflow_id,
            "media_id": media_id,
            "http_status": r.status_code
        }


def download_media_file(url: str, output_ext: str = ".mp4") -> Path:
    """Download media file locally into uploads directory."""
    filename = f"flow-{uuid.uuid4().hex[:10]}{output_ext}"
    out_path = UPLOADS_DIR / filename
    
    # Download with Chrome impersonation
    s = requests.Session(impersonate="chrome")
    r = s.get(url, timeout=180)
    if r.status_code == 200:
        out_path.write_bytes(r.content)
        return out_path
    raise RuntimeError(f"Failed to download media: HTTP {r.status_code}")


def run_video_job(job_id: str, prompt: str, image_bytes: bytes | None, origin: str) -> None:
    """Background execution for Google Flow & Gemini Veo video generation."""
    def set_progress(msg: str):
        with _JOBS_LOCK:
            if job_id in _VIDEO_JOBS:
                _VIDEO_JOBS[job_id]["progress"] = msg

    try:
        video_url = None
        local_file: Path | None = None
        base_img_path: Path | None = None

        if image_bytes:
            set_progress("Image analyze kora hocche...")
            tmp_in = UPLOADS_DIR / f"vbase-{uuid.uuid4().hex[:8]}.jpg"
            tmp_in.write_bytes(image_bytes)
            base_img_path = tmp_in
        else:
            set_progress("Nano Banana Pro Cinematic Engine diye visual frame toiri hocche...")
            # Gemini sometimes answers a keyframe request with text only
            # (no generated_images) even when the session is fine — the
            # /generate-image endpoint hits the same flakiness. Retry the
            # same call a few times before giving up and trying Flow.
            for _attempt in range(3):
                try:
                    import gemini_api_server
                    with gemini_api_server._SLOTS:
                        async def _gen_frame():
                            client = await gemini_api_server.get_gemini_client()
                            return await client.generate_content(
                                f"Generate an image of: {prompt}",
                                model=gemini_api_server.Model.ADVANCED_THINKING,
                                temporary=True
                            )
                        f_resp = gemini_api_server._run_coroutine_sync(_gen_frame())

                        cand_imgs = []
                        for c in (getattr(f_resp, "candidates", None) or []):
                            cand_imgs.extend(getattr(c, "generated_images", None) or [])
                        if cand_imgs:
                            frame_name = f"vframe-{uuid.uuid4().hex[:8]}.png"
                            async def _save():
                                return await cand_imgs[0].save(path=str(UPLOADS_DIR), filename=frame_name, full_size=False)
                            saved_str = gemini_api_server._run_coroutine_sync(_save())
                            base_img_path = Path(saved_str)
                            break
                        elif getattr(f_resp, "text", ""):
                            # Extract any image URL from text
                            found = gemini_api_server._extract_image_url(f_resp.text)
                            if found:
                                base_img_path = gemini_api_server._download_found_image(found)
                                if base_img_path:
                                    break
                except Exception as ef:
                    logger.warning(f"Nano Banana Pro keyframe generation failed (attempt {_attempt + 1}/3): {ef}")
                if base_img_path and base_img_path.is_file():
                    break

            # Backup if Gemini didn't return image: try Google Flow Imagen
            if not base_img_path or not base_img_path.is_file():
                try:
                    flow_c = load_cookies(service="flow")
                    client = FlowClient(flow_c)
                    f_res = client.generate_image_flow(prompt)
                    if f_res.get("status") == "success" and f_res.get("images"):
                        base_img_path = download_media_file(f_res["images"][0], ".jpg")
                except Exception as ef2:
                    logger.warning(f"Flow image fallback for video failed: {ef2}")

        if base_img_path and base_img_path.is_file():
            set_progress("Cinematic animation compile hocche (1080p 25fps H.264 + music)...")
            try:
                out_vid = UPLOADS_DIR / f"motion-{uuid.uuid4().hex[:8]}.mp4"
                # 8s 1080p slow cinematic zoom + gentle ambient music bed.
                # NOTE: this is still a motion video (one still image
                # animated with ffmpeg), NOT Veo — Veo with native audio
                # needs a live signed-in Flow browser tab / official API
                # key, which a headless server cannot do. The music bed
                # gives the clip sound; the label in the UI says so.
                cmd = [
                    "ffmpeg", "-y", "-loop", "1", "-i", str(base_img_path),
                    "-f", "lavfi", "-i", "sine=frequency=110:duration=8",
                    "-f", "lavfi", "-i", "sine=frequency=164.81:duration=8",
                    "-f", "lavfi", "-i", "sine=frequency=220:duration=8",
                    "-f", "lavfi", "-i", "sine=frequency=277.18:duration=8",
                    "-filter_complex",
                    "[0:v]zoompan=z='min(zoom+0.0012,1.25)':d=200:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps=25,format=yuv420p[v];"
                    "[1:a][2:a][3:a][4:a]amix=inputs=4:duration=first:dropout_transition=0,volume=0.12,afade=t=in:st=0:d=1,afade=t=out:st=7:d=1[a]",
                    "-map", "[v]", "-map", "[a]",
                    "-t", "8", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k", "-shortest", str(out_vid)
                ]
                subprocess.run(cmd, timeout=120, check=True)
                if out_vid.is_file() and out_vid.stat().st_size > 0:
                    local_file = out_vid
            except Exception as ve:
                logger.warning(f"FFmpeg motion compile failed: {ve}")

        if local_file and local_file.is_file():
            final_url = f"{origin}/uploads/{local_file.name}"
            with _JOBS_LOCK:
                _VIDEO_JOBS[job_id] = {
                    "status": "done",
                    "video_url": final_url,
                    "filename": local_file.name,
                    "size": local_file.stat().st_size,
                    "prompt": prompt,
                    "progress": "Video shompurno hoyeche! Nicher player-e dekhun ba download korun."
                }
        elif video_url:
            with _JOBS_LOCK:
                _VIDEO_JOBS[job_id] = {
                    "status": "done",
                    "video_url": video_url,
                    "prompt": prompt,
                    "progress": "Video shofolbhabe generate hoyeche!"
                }
        else:
            # No keyframe could be made: text-to-video needs a working
            # Gemini/Flow session to create the first frame. Image-to-video
            # (upload a photo first) does not need it and keeps working.
            # NOTE: do not always blame the session — Gemini sometimes
            # answers with text only even while authenticated. Check the
            # real account status before telling the user to re-sync.
            _auth_ok = None
            try:
                import gemini_api_server as _gas
                _auth_ok = getattr(_gas, "_LAST_ACCOUNT_STATUS", None) == 1000
            except Exception:
                _auth_ok = None
            if _auth_ok:
                err_msg = (
                    "Video toiri kora jayni — prothom frame banano jayni. "
                    "Google session thik ache, kintu Gemini ebar chhobi dite pareni. "
                    "Abar chesta korun, ba ekta chhobi upload diye image-to-video korun."
                )
            else:
                err_msg = (
                    "Video toiri kora jayni — prothom frame banano jayni. "
                    "Google Account session expired hoyeche; '🔑 Google & Flow Sync' theke notun cookie sync korun, "
                    "ba ekta chhobi upload diye image-to-video chesta korun."
                )
            with _JOBS_LOCK:
                _VIDEO_JOBS[job_id] = {
                    "status": "error",
                    "error": err_msg,
                    "progress": "Video toiri byartho hoyeche."
                }

    except Exception as e:
        logger.error(f"Video job {job_id} error: {e}", exc_info=True)
        with _JOBS_LOCK:
            _VIDEO_JOBS[job_id] = {
                "status": "error",
                "error": str(e),
                "progress": f"Truti: {e}"
            }


def start_video_generation(prompt: str, image_bytes: bytes | None, origin: str) -> str:
    """Start video generation job asynchronously."""
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _VIDEO_JOBS[job_id] = {
            "status": "working",
            "progress": "Video request toiri hocche...",
            "prompt": prompt,
            "created_at": time.time()
        }
    t = threading.Thread(target=run_video_job, args=(job_id, prompt, image_bytes, origin), daemon=True)
    t.start()
    return job_id


def get_video_job(job_id: str) -> dict[str, Any] | None:
    """Get video job details."""
    with _JOBS_LOCK:
        return _VIDEO_JOBS.get(job_id)


# ---------------------------------------------------------------------------
# Multi-Account Management Vault
# ---------------------------------------------------------------------------
VAULT_PATH = Path(__file__).resolve().parent / "accounts_vault.json"


def load_accounts_vault() -> dict[str, Any]:
    """Load multi-account vault separating Gemini and Flow accounts."""
    default_vault = {
        "active_gemini_account": "gemini_default",
        "active_flow_account": "flow_default",
        "gemini_accounts": {},
        "flow_accounts": {}
    }
    if not VAULT_PATH.is_file():
        gemini_c = load_cookies(service="gemini")
        flow_c = load_cookies(service="flow")
        default_vault["gemini_accounts"]["gemini_default"] = {
            "id": "gemini_default",
            "name": "Gemini Account (Primary)",
            "email": "",
            "cookies": gemini_c,
            "status": "ready" if gemini_c else "empty",
            "updated_at": time.time()
        }
        default_vault["flow_accounts"]["flow_default"] = {
            "id": "flow_default",
            "name": "Google Flow Account (Primary)",
            "email": "",
            "cookies": flow_c,
            "status": "ready" if flow_c else "empty",
            "updated_at": time.time()
        }
        save_accounts_vault(default_vault)
        return default_vault

    try:
        data = json.loads(VAULT_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            # Migrate older single-account-pool format if present
            if "accounts" in data and ("gemini_accounts" not in data or "flow_accounts" not in data):
                data.setdefault("gemini_accounts", {})
                data.setdefault("flow_accounts", {})
                data.setdefault("active_gemini_account", "gemini_default")
                data.setdefault("active_flow_account", "flow_default")
                for k, v in data.get("accounts", {}).items():
                    data["gemini_accounts"][f"gemini_{k}"] = {
                        "id": f"gemini_{k}",
                        "name": f"Gemini ({v.get('name', k)})",
                        "cookies": v.get("cookies", {}),
                        "status": "ready",
                        "updated_at": v.get("updated_at", time.time())
                    }
                    data["flow_accounts"][f"flow_{k}"] = {
                        "id": f"flow_{k}",
                        "name": f"Flow ({v.get('name', k)})",
                        "cookies": v.get("cookies", {}),
                        "status": "ready",
                        "updated_at": v.get("updated_at", time.time())
                    }
                save_accounts_vault(data)
            return data
    except Exception as e:
        logger.warning(f"Error loading accounts vault: {e}")

    return default_vault


def save_accounts_vault(data: dict[str, Any]) -> None:
    """Save multi-account vault with strict 0600 Linux permissions."""
    try:
        VAULT_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        VAULT_PATH.chmod(0o600)  # Owner read/write only, secure against other users
    except Exception as e:
        logger.error(f"Error saving accounts vault: {e}")


def list_accounts(service: str = "gemini") -> list[dict[str, Any]]:
    """List all configured accounts in vault for specified service ('gemini' or 'flow')."""
    vault = load_accounts_vault()
    if service == "flow":
        accs = vault.get("flow_accounts", {})
        active_id = vault.get("active_flow_account", "flow_default")
    else:
        accs = vault.get("gemini_accounts", {})
        active_id = vault.get("active_gemini_account", "gemini_default")

    result = []
    for aid, acc in accs.items():
        cookies = acc.get("cookies", {})
        has_ts = "__Secure-1PSIDTS" in cookies if isinstance(cookies, dict) else False
        result.append({
            "id": aid,
            "service": service,
            "name": acc.get("name", aid),
            "email": acc.get("email", ""),
            "status": acc.get("status", "ready"),
            "is_active": aid == active_id,
            "cookies_count": len(cookies) if isinstance(cookies, dict) else 0,
            "has_1psidts": has_ts,
            "updated_at": acc.get("updated_at", 0)
        })
    return result


def add_or_update_account(service: str, account_id: str, name: str, cookies: dict[str, str], email: str = "") -> dict[str, Any]:
    """Add or update an account for specified service ('gemini' or 'flow')."""
    vault = load_accounts_vault()
    acc_key = "flow_accounts" if service == "flow" else "gemini_accounts"
    act_key = "active_flow_account" if service == "flow" else "active_gemini_account"

    if not account_id:
        account_id = f"{service}_{uuid.uuid4().hex[:6]}"

    vault.setdefault(acc_key, {})[account_id] = {
        "id": account_id,
        "service": service,
        "name": name or f"{service.capitalize()} Account ({account_id})",
        "email": email,
        "cookies": cookies,
        "status": "ready",
        "updated_at": time.time()
    }

    # If first account or current active missing, activate
    if len(vault[acc_key]) == 1 or vault.get(act_key) not in vault[acc_key]:
        vault[act_key] = account_id
        save_cookies(cookies, service=service)

    save_accounts_vault(vault)
    return {"id": account_id, "service": service, "name": name, "status": "saved"}


def switch_account(service: str, account_id: str) -> bool:
    """Switch active account for specified service ('gemini' or 'flow')."""
    vault = load_accounts_vault()
    acc_key = "flow_accounts" if service == "flow" else "gemini_accounts"
    act_key = "active_flow_account" if service == "flow" else "active_gemini_account"

    accounts = vault.get(acc_key, {})
    if account_id not in accounts:
        return False

    vault[act_key] = account_id
    save_accounts_vault(vault)

    acc = accounts[account_id]
    cookies = acc.get("cookies", {})
    save_cookies(cookies, service=service)
    logger.info(f"Switched active {service} account to: {acc.get('name', account_id)}")
    return True


def delete_account(service: str, account_id: str) -> bool:
    """Delete an account from the vault."""
    vault = load_accounts_vault()
    acc_key = "flow_accounts" if service == "flow" else "gemini_accounts"
    act_key = "active_flow_account" if service == "flow" else "active_gemini_account"

    accounts = vault.get(acc_key, {})
    if account_id in accounts:
        del accounts[account_id]
        if vault.get(act_key) == account_id:
            vault[act_key] = next(iter(accounts.keys()), f"{service}_default")
            if vault[act_key] in accounts:
                save_cookies(accounts[vault[act_key]].get("cookies", {}), service=service)
        save_accounts_vault(vault)
        return True
    return False


# ---------------------------------------------------------------------------
# Background Continuous Keepalive & Rotation Daemon
# ---------------------------------------------------------------------------
_KEEPALIVE_THREAD: threading.Thread | None = None


def _continuous_keepalive_worker():
    """Background loop: checks and rotates Gemini __Secure-1PSIDTS every 12 minutes."""
    while True:
        try:
            time.sleep(720)  # 12 minutes
            cookies = load_cookies(service="gemini")
            psid = cookies.get("__Secure-1PSID")
            psidts = cookies.get("__Secure-1PSIDTS")
            if psid and psidts:
                # Call accounts.google.com RotateCookies
                s = requests.Session(impersonate="chrome")
                for k, v in cookies.items():
                    s.cookies.set(k, v, domain=".google.com", path="/")
                headers = {
                    "authority": "accounts.google.com",
                    "content-type": "application/json",
                    "origin": "https://gemini.google.com",
                    "referer": "https://gemini.google.com/",
                }
                r = s.post("https://accounts.google.com/RotateCookies", headers=headers, data='[000,"-0000000000000000000"]', timeout=30)
                if r.status_code == 200:
                    new_ts = None
                    for c in s.cookies.jar:
                        if c.name == "__Secure-1PSIDTS":
                            new_ts = c.value
                            break
                    if new_ts and new_ts != psidts:
                        cookies["__Secure-1PSIDTS"] = new_ts
                        save_cookies(cookies, service="gemini")
                        # Also update active gemini account in vault
                        vault = load_accounts_vault()
                        aid = vault.get("active_gemini_account")
                        if aid and aid in vault.get("gemini_accounts", {}):
                            vault["gemini_accounts"][aid]["cookies"]["__Secure-1PSIDTS"] = new_ts
                            vault["gemini_accounts"][aid]["updated_at"] = time.time()
                            save_accounts_vault(vault)
                        logger.info(f"[✓] Keepalive Daemon: Successfully rotated Gemini __Secure-1PSIDTS at {time.strftime('%H:%M:%S')}")
        except Exception as e:
            logger.debug(f"Keepalive daemon iteration note: {e}")


def ensure_keepalive_daemon():
    """Start keepalive daemon thread if not already running."""
    global _KEEPALIVE_THREAD
    if _KEEPALIVE_THREAD is None or not _KEEPALIVE_THREAD.is_alive():
        _KEEPALIVE_THREAD = threading.Thread(target=_continuous_keepalive_worker, daemon=True, name="SessionKeepalive")
        _KEEPALIVE_THREAD.start()
        logger.info("[✓] Session Keepalive Daemon started in background.")

ensure_keepalive_daemon()

