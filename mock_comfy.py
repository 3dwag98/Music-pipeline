#!/usr/bin/env python3
"""Fake ComfyUI server, so the ComfyUI path can be rehearsed without a GPU.

Speaks the same endpoints the real thing does (/system_stats, /object_info,
/prompt, /history, /view) and returns a placeholder image or a few test chords
instead of real output.

    python mock_comfy.py                       # http://127.0.0.1:8188
    python pipeline.py art --prompt "test" --comfy-url http://127.0.0.1:8188
"""

import json
import struct
import tempfile
import threading
import time
import uuid
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

OUT = Path(tempfile.mkdtemp(prefix="mock_comfy_"))
HISTORY = {}
PORT = 8188
SR = 44100


def make_png(width=832, height=480, seed=0):
    """A small gradient PNG, written by hand so the mock needs no imaging deps."""
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            row += bytes(((x * 255) // max(1, width - 1),
                          (y * 255) // max(1, height - 1),
                          (seed * 37 + x + y) % 256))
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def make_wav(seconds=20.0, seed=0):
    import math
    n = int(seconds * SR)
    frames = bytearray()
    roots = [220.0, 261.6, 293.7, 329.6]
    for i in range(n):
        t = i / SR
        root = roots[int(t // 4) % len(roots)]
        value = 0.0
        for ratio in (1.0, 1.26, 1.5):
            value += 0.12 * math.sin(2 * math.pi * root * ratio * t)
        value += 0.25 * math.sin(2 * math.pi * 55 * t) * math.exp(-8 * (t % 0.75))
        sample = max(-1.0, min(1.0, value))
        frames += struct.pack("<hh", int(sample * 30000), int(sample * 30000))
    data = bytes(frames)
    head = (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 2, SR, SR * 4, 4, 16)
            + b"data" + struct.pack("<I", len(data)))
    return head + data


def finish(prompt_id, graph):
    time.sleep(1.0)
    wants_audio = any("Audio" in str(node.get("class_type", "")) for node in graph.values())
    outputs, node_id = {}, next(iter(graph), "1")
    seed = 0
    for node in graph.values():
        seed = node.get("inputs", {}).get("seed", seed)
    if wants_audio:
        seconds = 20.0
        for node in graph.values():
            seconds = node.get("inputs", {}).get("seconds", seconds)
        name = f"{prompt_id[:8]}.wav"
        (OUT / name).write_bytes(make_wav(min(float(seconds), 30.0), int(seed)))
        outputs[node_id] = {"audio": [{"filename": name, "subfolder": "", "type": "output"}]}
    else:
        width = height = None
        for node in graph.values():
            width = node.get("inputs", {}).get("width", width)
            height = node.get("inputs", {}).get("height", height)
        name = f"{prompt_id[:8]}.png"
        (OUT / name).write_bytes(make_png(int(width or 832), int(height or 480), int(seed)))
        outputs[node_id] = {"images": [{"filename": name, "subfolder": "", "type": "output"}]}
    HISTORY[prompt_id] = {"outputs": outputs,
                          "status": {"status_str": "success", "completed": True,
                                     "messages": []}}


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/system_stats":
            return self._json(200, {
                "system": {"comfyui_version": "mock", "python_version": "3.11"},
                "devices": [{"name": "Mock GPU", "type": "cuda",
                             "vram_total": 6 * 1024 ** 3, "vram_free": 5 * 1024 ** 3}]})
        if url.path == "/object_info":
            return self._json(200, {
                "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [
                    ["v1-5-pruned-emaonly.safetensors", "ace_step_v1_3.5b.safetensors"]]}}},
                "KSampler": {}, "CLIPTextEncode": {}, "SaveImage": {}, "SaveAudio": {},
                "EmptyLatentImage": {}, "VAEDecode": {}, "VAEDecodeAudio": {},
                "TextEncodeAceStepAudio": {}, "EmptyAceStepLatentAudio": {},
                "ModelSamplingSD3": {}})
        if url.path.startswith("/history/"):
            prompt_id = url.path.rsplit("/", 1)[-1]
            entry = HISTORY.get(prompt_id)
            return self._json(200, {prompt_id: entry} if entry else {})
        if url.path == "/history":
            return self._json(200, HISTORY)
        if url.path == "/view":
            query = parse_qs(url.query)
            path = OUT / query.get("filename", [""])[0]
            if not path.is_file():
                return self._json(404, {"error": "not found"})
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        url = urlparse(self.path)
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0)
        body = json.loads(raw or b"{}")
        if url.path == "/prompt":
            graph = body.get("prompt") or {}
            if not graph:
                return self._json(400, {"error": {"type": "no_prompt",
                                                  "message": "empty workflow"}})
            missing = {nid: node for nid, node in graph.items()
                       if "class_type" not in node}
            if missing:
                return self._json(400, {"error": {"type": "invalid_prompt",
                                                  "message": "node without class_type"},
                                        "node_errors": {nid: {"errors": [
                                            {"message": "missing class_type"}]}
                                            for nid in missing}})
            prompt_id = str(uuid.uuid4())
            print(f"prompt {prompt_id[:8]}  {len(graph)} nodes  "
                  f"{sorted({n.get('class_type') for n in graph.values()})}", flush=True)
            threading.Thread(target=finish, args=(prompt_id, graph), daemon=True).start()
            return self._json(200, {"prompt_id": prompt_id, "number": 1, "node_errors": {}})
        self._json(404, {"error": "not found"})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"Mock ComfyUI on http://127.0.0.1:{PORT}  (outputs in {OUT})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
