"""ComfyUI client: drive a local ComfyUI server from the pipeline.

ComfyUI is optional.  It is useful here for two things:

  art       the looping clip or still image that goes under the audio - the one
            step the pipeline could not do for you before
  audio     ACE-Step (and anything else ComfyUI can load) as a generation
            backend, with ComfyUI's memory management instead of the standalone
            server's, which matters a lot on a 6 GB card

The important design decision: workflows are **data, not code**.  Node names
and required checkpoints change between ComfyUI versions and between machines,
so hardcoding a graph would break on somebody else's install.  Instead the
pipeline drives any API-format workflow JSON - including one you exported
yourself from ComfyUI with *Workflow -> Export (API)* - and patches values into
it by node title, class type or node id.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from .util import AUDIO_EXTS, die, log, warn

#: ComfyUI flags that matter on a GTX 1660 Ti (6 GB, Turing TU116).
#: --force-fp32 is the same story as ACE-Step: the 16-series half-precision
#: path produces black images and silent audio, not speed.
LOW_VRAM_FLAGS = [
    "--lowvram",
    "--force-fp32",
    "--fp32-vae",
    "--use-split-cross-attention",
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".webm", ".gif", ".mkv"}


class ComfyClient:
    """Minimal ComfyUI HTTP client (POST /prompt, GET /history, GET /view)."""

    def __init__(self, base_url="http://127.0.0.1:8188", timeout=30, client_id=None):
        try:
            import requests
        except ImportError:
            die("the ComfyUI backend needs requests: pip install requests")
        self.requests = requests
        self.base = str(base_url).rstrip("/")
        self.timeout = timeout
        self.client_id = client_id or str(uuid.uuid4())

    # ------------------------------------------------------------- plumbing --
    def _get(self, path, **kwargs):
        r = self.requests.get(f"{self.base}{path}", timeout=kwargs.pop("timeout", self.timeout),
                              **kwargs)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code} on {path}: {r.text[:300]}")
        return r

    def health(self, fatal=True):
        """ComfyUI has no /health; /system_stats is the usual liveness probe."""
        try:
            return self._get("/system_stats").json()
        except Exception as exc:
            if fatal:
                die(f"cannot reach ComfyUI at {self.base} ({exc}).\n"
                    f"       Start it with:  python main.py {' '.join(LOW_VRAM_FLAGS)}\n"
                    f"       or point elsewhere with --comfy-url")
            return None

    def describe(self):
        """A short human summary of what the server is and how much VRAM it has."""
        stats = self.health(fatal=False)
        if not stats:
            return "unreachable"
        devices = stats.get("devices") or []
        if devices:
            dev = devices[0]
            total = dev.get("vram_total", 0) / 1e9
            free = dev.get("vram_free", 0) / 1e9
            return f"{dev.get('name', 'device')} ({free:.1f}/{total:.1f} GB VRAM free)"
        return "connected"

    def object_info(self):
        try:
            return self._get("/object_info", timeout=60).json()
        except Exception:
            return {}

    def available(self, node_class):
        return node_class in self.object_info()

    def checkpoints(self):
        """Model files ComfyUI can actually see - used to give a useful error."""
        info = self.object_info().get("CheckpointLoaderSimple", {})
        try:
            return list(info["input"]["required"]["ckpt_name"][0])
        except (KeyError, IndexError, TypeError):
            return []

    # ------------------------------------------------------------ execution --
    def submit(self, graph):
        payload = {"prompt": graph, "client_id": self.client_id}
        r = self.requests.post(f"{self.base}/prompt", json=payload, timeout=self.timeout)
        if r.status_code >= 400:
            detail = r.text[:800]
            try:
                body = r.json()
                node_errors = body.get("node_errors") or {}
                if node_errors:
                    lines = []
                    for node_id, err in node_errors.items():
                        for item in err.get("errors", []):
                            lines.append(f"node {node_id} ({err.get('class_type', '?')}): "
                                         f"{item.get('message')} {item.get('details', '')}")
                    detail = "\n       ".join(lines) or detail
                elif body.get("error"):
                    err = body["error"]
                    detail = f"{err.get('type')}: {err.get('message')} {err.get('details', '')}"
            except ValueError:
                pass
            raise RuntimeError(f"ComfyUI rejected the workflow:\n       {detail}")
        data = r.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"no prompt_id in response: {str(data)[:300]}")
        return prompt_id

    def wait(self, prompt_id, timeout=1800, poll=1.5):
        """Poll /history until the prompt finishes.  Returns its outputs dict."""
        start = time.time()
        last_note = 0.0
        while True:
            try:
                history = self._get(f"/history/{prompt_id}").json()
            except Exception as exc:                 # a restart mid-run
                raise RuntimeError(f"lost contact with ComfyUI: {exc}") from exc
            entry = history.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error" or status.get("completed") is False:
                    raise RuntimeError(f"workflow failed: {_error_text(status)}")
                outputs = entry.get("outputs")
                if outputs:
                    return outputs
                if status.get("completed"):
                    raise RuntimeError("workflow finished but produced no outputs - "
                                       "does it have a Save node?")
            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(f"ComfyUI prompt {prompt_id} still running after {timeout}s")
            if elapsed - last_note >= 30:
                log(f"    ...still working ({int(elapsed)}s)")
                last_note = elapsed
            time.sleep(poll)

    def download(self, item, dest):
        """Fetch one output file described by {filename, subfolder, type}."""
        params = {"filename": item.get("filename", ""),
                  "subfolder": item.get("subfolder", ""),
                  "type": item.get("type", "output")}
        dest = Path(dest)
        suffix = Path(params["filename"]).suffix
        if suffix and not dest.suffix:
            dest = dest.with_suffix(suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self.requests.get(f"{self.base}/view", params=params, stream=True,
                               timeout=600) as r:
            if r.status_code >= 400:
                raise RuntimeError(f"could not download {params['filename']}: "
                                   f"HTTP {r.status_code}")
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(1 << 16):
                    fh.write(chunk)
        return dest

    def run(self, graph, dest_stem, want=None, timeout=1800):
        """Submit, wait, and download every produced file.  Returns paths."""
        prompt_id = self.submit(graph)
        outputs = self.wait(prompt_id, timeout=timeout)
        items = collect_outputs(outputs, want)
        if not items:
            raise RuntimeError(f"workflow produced no {want or 'output'} files")
        paths = []
        for i, item in enumerate(items):
            stem = Path(dest_stem)
            if len(items) > 1:
                stem = stem.with_name(f"{stem.name}_{i + 1}")
            paths.append(self.download(item, stem))
        return paths


def _error_text(status):
    messages = status.get("messages") or []
    for kind, payload in messages:
        if kind in ("execution_error", "execution_interrupted") and isinstance(payload, dict):
            return (f"{payload.get('node_type', '?')}: "
                    f"{payload.get('exception_message') or payload.get('exception_type', '')}")
    return status.get("status_str", "unknown error")


#: Which output key each kind of result lands under in /history.
OUTPUT_KEYS = {
    "audio": ("audio",),
    "image": ("images",),
    "video": ("gifs", "videos", "images"),
    None: ("audio", "images", "gifs", "videos"),
}


def collect_outputs(outputs, want=None):
    """Flatten ComfyUI's {node_id: {images:[...], audio:[...]}} into a list."""
    keys = OUTPUT_KEYS.get(want, OUTPUT_KEYS[None])
    items = []
    for node_output in outputs.values():
        for key in keys:
            for item in node_output.get(key, []) or []:
                if isinstance(item, dict) and item.get("filename"):
                    items.append(item)
    if want == "video":
        items = [i for i in items if Path(i["filename"]).suffix.lower() in VIDEO_EXTS] or items
    if want == "audio":
        items = [i for i in items if Path(i["filename"]).suffix.lower() in AUDIO_EXTS] or items
    return items


# --------------------------------------------------------------- workflows ---

WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "workflows"


def load_workflow(path):
    """Load an API-format workflow.  Accepts a name in workflows/ or a path."""
    candidate = Path(path)
    if not candidate.exists():
        for guess in (WORKFLOW_DIR / path, WORKFLOW_DIR / f"{path}.json"):
            if guess.exists():
                candidate = guess
                break
        else:
            available = ", ".join(sorted(p.stem for p in WORKFLOW_DIR.glob("*.json")))
            die(f"workflow not found: {path}\n       built in: {available or '(none)'}")
    try:
        graph = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        die(f"{candidate} is not valid JSON: {exc}")
    if not isinstance(graph, dict) or not graph:
        die(f"{candidate} is empty")
    if "nodes" in graph and "links" in graph:
        die(f"{candidate} is a ComfyUI *editor* workflow, not an API one.\n"
            f"       In ComfyUI: Workflow -> Export (API), and use that file.")
    for node_id, node in graph.items():
        if not isinstance(node, dict) or "class_type" not in node:
            die(f"{candidate}: node {node_id!r} has no class_type - "
                f"is this an API-format export?")
    return graph


def node_title(node):
    return (node.get("_meta") or {}).get("title", "")


def find_nodes(graph, selector):
    """Resolve a selector to node ids: a node id, a _meta title, or a class_type.

    Matching is case-insensitive for titles and class types so that
    `--set positive.text=...` works against a node titled "Positive".
    """
    selector = str(selector)
    if selector in graph:
        return [selector]
    low = selector.lower()
    by_title = [nid for nid, node in graph.items() if node_title(node).lower() == low]
    if by_title:
        return by_title
    by_class = [nid for nid, node in graph.items()
                if str(node.get("class_type", "")).lower() == low]
    return by_class


def set_input(graph, selector, field, value, required=False, create=False):
    """Patch `field` on every node matching `selector`.  Returns how many changed.

    A field the node does not already have is skipped rather than created:
    inventing an input silently does nothing in ComfyUI (the node ignores it)
    while making the caller think the value was applied.
    """
    nodes = find_nodes(graph, selector)
    if not nodes:
        if required:
            die(f"no node matching '{selector}' in the workflow")
        return 0
    changed = 0
    for nid in nodes:
        inputs = graph[nid].setdefault("inputs", {})
        if field not in inputs and not create:
            if required:
                die(f"node '{selector}' has no input '{field}' "
                    f"(it has: {', '.join(sorted(inputs)) or 'none'})")
            continue
        if field in inputs and isinstance(inputs[field], list):
            # this input is wired to another node's output; overwriting it would
            # silently disconnect the graph
            warn(f"skipping {selector}.{field}: it is connected to another node")
            continue
        inputs[field] = value
        changed += 1
    return changed


def set_where(graph, field, value, classes=None):
    """Patch `field` on every node that already has it (optionally class-filtered)."""
    changed = 0
    for node in graph.values():
        if classes and node.get("class_type") not in classes:
            continue
        inputs = node.get("inputs", {})
        if field in inputs and not isinstance(inputs[field], list):
            inputs[field] = value
            changed += 1
    return changed


#: Which input carries the prompt, per node class.  An image encoder calls it
#: `text`; ACE-Step's audio encoder calls it `tags`.  Patching the wrong one is
#: silent - the node just ignores the extra key - so the class decides.
TEXT_FIELDS = {
    "CLIPTextEncode": "text",
    "CLIPTextEncodeSDXL": "text_g",
    "CLIPTextEncodeSDXLRefiner": "text",
    "TextEncodeAceStepAudio": "tags",
    "T5TextEncode": "text",
}
#: Fallbacks for a class we do not know, in preference order.
TEXT_FALLBACKS = ("text", "tags", "prompt", "string", "text_g")
SEED_FIELDS = ("seed", "noise_seed")


def text_field_of(node):
    """The input that carries this node's prompt, or None if it has none."""
    inputs = node.get("inputs") or {}
    field = TEXT_FIELDS.get(node.get("class_type"))
    if field and field in inputs and not isinstance(inputs[field], list):
        return field
    for candidate in TEXT_FALLBACKS:
        if candidate in inputs and isinstance(inputs[candidate], str):
            return candidate
    return None


def set_text(graph, selector, value):
    """Set the prompt on matching nodes, using each node's own text field."""
    changed = 0
    for nid in find_nodes(graph, selector):
        field = text_field_of(graph[nid])
        if field:
            graph[nid]["inputs"][field] = value
            changed += 1
    return changed


def apply_settings(graph, prompt=None, negative=None, seed=None, steps=None, cfg=None,
                   width=None, height=None, seconds=None, lyrics=None, filename=None,
                   overrides=None):
    """Patch the usual knobs into a workflow, by title first then by class.

    Title wins so that a workflow you exported yourself can opt in simply by
    naming its nodes POSITIVE / NEGATIVE.
    """
    applied = {}

    def note(key, count):
        if count:
            applied[key] = count

    if prompt is not None:
        count = set_text(graph, "POSITIVE", prompt)
        if not count:
            # no titled node: use the first text-carrying node in graph order
            for nid in sorted(graph, key=_as_int):
                if node_title(graph[nid]).upper() == "NEGATIVE":
                    continue
                field = text_field_of(graph[nid])
                if field:
                    graph[nid]["inputs"][field] = prompt
                    count = 1
                    break
        note("prompt", count)
    if negative is not None:
        note("negative", set_text(graph, "NEGATIVE", negative))
    if lyrics is not None:
        # only the positive conditioning takes lyrics; putting them on the
        # negative node asks the model to avoid singing the words you wanted
        count = set_input(graph, "POSITIVE", "lyrics", lyrics)
        if not count:
            count = set_where(graph, "lyrics", lyrics,
                              classes={"TextEncodeAceStepAudio"})
        note("lyrics", count)
    if seed is not None:
        total = 0
        for field in SEED_FIELDS:
            total += set_where(graph, field, int(seed))
        note("seed", total)
    if steps is not None:
        note("steps", set_where(graph, "steps", int(steps)))
    if cfg is not None:
        note("cfg", set_where(graph, "cfg", float(cfg)))
    if width is not None:
        note("width", set_where(graph, "width", int(width)))
    if height is not None:
        note("height", set_where(graph, "height", int(height)))
    if seconds is not None:
        note("seconds", set_where(graph, "seconds", float(seconds)))
    if filename is not None:
        note("filename_prefix", set_where(graph, "filename_prefix", str(filename)))

    for raw in overrides or []:
        if "=" not in raw:
            die(f"--set needs NODE.FIELD=VALUE, got: {raw}")
        target, _, value = raw.partition("=")
        if "." not in target:
            die(f"--set needs NODE.FIELD=VALUE, got: {raw}")
        selector, _, field = target.rpartition(".")
        set_input(graph, selector, field, _coerce(value), required=True, create=False)
        applied[target] = 1
    return applied


def _as_int(node_id):
    try:
        return int(node_id)
    except (TypeError, ValueError):
        return 1 << 30


def _coerce(text):
    """Turn a --set value into int/float/bool where it obviously is one."""
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def describe_workflow(graph):
    """One line per node - so `--show-workflow` tells you what you can patch."""
    lines = []
    for nid in sorted(graph, key=_as_int):
        node = graph[nid]
        title = node_title(node)
        fields = [k for k, v in (node.get("inputs") or {}).items() if not isinstance(v, list)]
        lines.append(f"  {nid:>4}  {node.get('class_type', '?'):<28} "
                     f"{('[' + title + ']') if title else '':<20} "
                     f"{', '.join(sorted(fields))}")
    return lines
