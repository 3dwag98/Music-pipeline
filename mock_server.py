#!/usr/bin/env python3
"""
Fake ACE-Step API for testing the pipeline without a GPU or model download.
It speaks the same endpoints (/health, /release_task, /query_result, /v1/audio)
but returns simple synthesized chords instead of real music.

    python mock_server.py            # listens on http://127.0.0.1:8001
    python pipeline.py all --count 3 --duration 20 --minutes 1
"""

import json
import tempfile
import threading
import time
import uuid
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import numpy as np
import soundfile as sf

OUT = Path(tempfile.mkdtemp(prefix="mock_acestep_"))
TASKS = {}
SR = 48000


def synth(duration, bpm, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * SR)) / SR
    roots = rng.choice([220.0, 246.9, 261.6, 293.7, 329.6], size=4)
    bar = 4 * 60 / bpm
    audio = np.zeros_like(t)
    for i, root in enumerate(roots):
        mask = ((t // bar) % 4) == i
        for ratio in (1.0, 1.26, 1.5):
            audio += mask * 0.12 * np.sin(2 * np.pi * root * ratio * t)
    beat = 60 / bpm
    audio += 0.3 * np.sin(2 * np.pi * 55 * t) * np.exp(-30 * (t % beat))
    audio += 0.01 * rng.standard_normal(len(t))
    stereo = np.stack([audio, np.roll(audio, 200)], axis=1) * 0.5
    silence = np.zeros((SR // 2, 2))  # half a second of silence at both ends, like real outputs
    return np.concatenate([silence, stereo, silence]).astype("float32")


def finish(task_id, payload):
    time.sleep(1.5)
    duration = float(payload.get("audio_duration") or payload.get("duration") or 30)
    bpm = int(float(payload.get("bpm") or 80))
    seed = int(float(payload.get("seed") or 0)) % (2**32)
    path = OUT / f"{task_id}.wav"
    sf.write(str(path), synth(duration, bpm, seed), SR)
    TASKS[task_id]["status"] = 1
    TASKS[task_id]["result"] = json.dumps([{
        "file": f"/v1/audio?path={quote(str(path))}", "status": 1,
        "prompt": payload.get("prompt", ""), "seed_value": str(seed),
        "metas": {"bpm": bpm, "duration": duration, "keyscale": payload.get("key_scale", "")},
    }])


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps({"data": obj, "code": code, "error": None,
                           "timestamp": int(time.time() * 1000), "extra": None}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        ctype = self.headers.get("Content-Type", "")
        if ctype.startswith("multipart/form-data"):
            msg = BytesParser(policy=email_policy).parsebytes(
                b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + raw)
            fields, files = {}, []
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if part.get_filename():
                    files.append(name)
                else:
                    fields[name] = part.get_content()
            fields["_files"] = files
            return fields
        return json.loads(raw or b"{}")

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/health":
            return self._send(200, {"status": "ok", "service": "ACE-Step API (mock)", "version": "1.0"})
        if url.path == "/v1/audio":
            path = Path(parse_qs(url.query).get("path", [""])[0])
            if not path.is_file():
                return self._send(404, {"detail": "not found"})
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._send(404, {"detail": "not found"})

    def do_POST(self):
        url = urlparse(self.path)
        payload = self._body()
        if url.path == "/release_task":
            task_id = str(uuid.uuid4())
            TASKS[task_id] = {"task_id": task_id, "status": 0, "result": ""}
            files = payload.get("_files", [])
            print(f"task {task_id[:8]} type={payload.get('task_type')} batch={payload.get('batch_size')} "
                  f"thinking={payload.get('thinking')} files={files} bpm={payload.get('bpm')} "
                  f"key={payload.get('key_scale')!r}", flush=True)
            threading.Thread(target=finish, args=(task_id, payload), daemon=True).start()
            return self._send(200, {"task_id": task_id, "status": "queued", "queue_position": 1})
        if url.path == "/query_result":
            ids = payload.get("task_id_list", [])
            if isinstance(ids, str):
                ids = json.loads(ids)
            return self._send(200, [TASKS[i] for i in ids if i in TASKS])
        self._send(404, {"detail": "not found"})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"Mock ACE-Step API on http://127.0.0.1:8001  (audio in {OUT})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8001), Handler).serve_forever()
