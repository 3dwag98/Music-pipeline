"""ACE-Step 1.5 REST client, tuned for a 6 GB card.

The pipeline does not need ACE-Step - the built-in engine renders lofi with no
model at all.  This is the optional path for when you want a neural generator's
sound and you have the VRAM budget for it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .util import AUDIO_EXTS, die, log

#: What to put in ACE-Step's .env on a GTX 1660 Ti (6 GB, Turing TU116).
LOW_VRAM_ENV = {
    "ACESTEP_CONFIG_PATH": "acestep-v15-turbo",
    "ACESTEP_INIT_LLM": "false",           # the caption LLM will not fit alongside the DiT
    "ACESTEP_OFFLOAD_TO_CPU": "true",
    "ACESTEP_OFFLOAD_DIT_TO_CPU": "true",
    "ACESTEP_TORCH_DTYPE": "float32",      # see the note in doctor.py about 16-series fp16
}

#: Request defaults that keep a 6 GB card inside its budget.
LOW_VRAM_REQUEST = {
    "batch_size": 1,          # the server default is 2 and will OOM at 6 GB
    "thinking": False,
    "use_cot_caption": False,
    "use_cot_language": False,
    "use_format": False,
    "inference_steps": 8,     # the turbo checkpoint is built for very few steps
}


class AceStepClient:
    STATUS_WORDS = {"queued": 0, "running": 0, "pending": 0, "processing": 0,
                    "succeeded": 1, "success": 1, "done": 1, "completed": 1,
                    "failed": 2, "error": 2, "cancelled": 2}

    def __init__(self, base_url, api_key=None, timeout=60):
        try:
            import requests
        except ImportError:
            die("the ACE-Step backend needs requests: pip install requests")
        self.requests = requests
        self.base = str(base_url).rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout = timeout

    # -------------------------------------------------------------- plumbing --
    def _unwrap(self, resp):
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
        body = resp.json()
        if isinstance(body, dict) and "data" in body:
            if body.get("error"):
                raise RuntimeError(str(body["error"]))
            return body["data"]
        return body

    def health(self, fatal=True):
        try:
            r = self.requests.get(f"{self.base}/health", headers=self.headers, timeout=10)
            return self._unwrap(r)
        except Exception as exc:
            if fatal:
                die(f"cannot reach ACE-Step at {self.base} ({exc}).\n"
                    f"       Start it first (start_api_server.bat in the ACE-Step folder),\n"
                    f"       or use the built-in engine instead:  python pipeline.py lofi")
            return None

    def submit(self, payload, reference=None, source=None):
        files = {}
        try:
            # These have to stay open across the POST, so a `with` block is not
            # available; the `finally` below closes them on every path.
            if reference:
                files["reference_audio"] = open(reference, "rb")  # noqa: SIM115
            if source:
                files["src_audio"] = open(source, "rb")  # noqa: SIM115
            if files:
                form = {k: (str(v).lower() if isinstance(v, bool) else str(v))
                        for k, v in payload.items()}
                r = self.requests.post(f"{self.base}/release_task", data=form, files=files,
                                       headers=self.headers, timeout=180)
            else:
                r = self.requests.post(f"{self.base}/release_task", json=payload,
                                       headers=self.headers, timeout=self.timeout)
        finally:
            for fh in files.values():
                fh.close()
        data = self._unwrap(r)
        task_id = data.get("task_id") or data.get("job_id") if isinstance(data, dict) else None
        if not task_id:
            raise RuntimeError(f"no task_id in response: {str(data)[:300]}")
        return task_id

    def wait(self, task_id, poll=3.0, timeout=1800):
        start = time.time()
        last_note = 0.0
        while True:
            r = self.requests.post(f"{self.base}/query_result",
                                   json={"task_id_list": [task_id]},
                                   headers=self.headers, timeout=30)
            items = self._unwrap(r)
            if isinstance(items, dict):
                items = items.get("items") or [items]
            item = next((i for i in items if i.get("task_id") == task_id),
                        items[0] if items else {})
            status = item.get("status", 0)
            if isinstance(status, str):
                status = self.STATUS_WORDS.get(status.lower(), 0)
            if status == 1:
                return self._files_from(item)
            if status == 2:
                raise RuntimeError(f"generation failed: "
                                   f"{str(item.get('result') or item.get('error'))[:400]}")
            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(f"task {task_id} still running after {timeout}s")
            if elapsed - last_note >= 30:
                log(f"    ...still working ({int(elapsed)}s)")
                last_note = elapsed
            time.sleep(poll)

    @staticmethod
    def _files_from(item):
        result = item.get("result")
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                result = []
        if isinstance(result, dict):
            result = result.get("audio_paths") or [result]
        files = []
        for entry in result or []:
            if isinstance(entry, str):
                files.append(entry)
            elif isinstance(entry, dict):
                ref = entry.get("file") or entry.get("audio_path") or entry.get("path")
                if ref:
                    files.append(ref)
        if not files:
            raise RuntimeError(f"task finished but returned no audio: {str(item)[:300]}")
        return files

    def download(self, file_ref, dest_stem):
        dest_stem = Path(dest_stem)
        if file_ref.startswith("http"):
            url, params = file_ref, None
        elif file_ref.startswith("/"):
            url, params = self.base + file_ref, None
        else:
            url, params = f"{self.base}/v1/audio", {"path": file_ref}
        server_path = params["path"] if params else \
            parse_qs(urlparse(url).query).get("path", [""])[0]
        ext = Path(server_path).suffix.lower()
        dest = dest_stem.with_suffix(ext if ext in AUDIO_EXTS else ".wav")
        with self.requests.get(url, params=params, headers=self.headers,
                               stream=True, timeout=600) as r:
            if r.status_code >= 400:
                raise RuntimeError(f"download failed: HTTP {r.status_code}")
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(1 << 16):
                    fh.write(chunk)
        return dest


def is_oom(exc) -> bool:
    text = str(exc).lower()
    return any(s in text for s in ("out of memory", "cuda oom", "outofmemory",
                                   "allocate", "cublas_status_alloc_failed"))
