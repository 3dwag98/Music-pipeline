# Workflows

API-format ComfyUI workflows the pipeline can drive. They are **data, not code** —
the pipeline patches values into whatever graph you point it at.

## Using your own

The shipped files are starting points. Node names and checkpoint filenames differ
between ComfyUI versions and installs, so the reliable path is to build the
workflow you want *in ComfyUI*, confirm it runs there, then:

**Workflow → Export (API)** → save the JSON here → `--workflow yourfile.json`

Note it must be the **API** export. The regular "Save" format (with `nodes` and
`links` arrays) is a different shape and the pipeline will tell you so.

## How values get patched in

`--prompt`, `--seed`, `--steps`, `--width`, `--height`, `--seconds` are matched
this way, in order:

1. A node whose `_meta.title` matches (`POSITIVE`, `NEGATIVE`, …). Title your
   nodes and everything else follows — rename a node in ComfyUI by
   double-clicking its header.
2. Otherwise, by class type (`TextEncodeAceStepAudio`, `KSampler`, …).
3. Otherwise, any node that already has an input of that name (`seed`, `steps`).

Anything else: `--set NODE.FIELD=VALUE`, where `NODE` is a title, a class type or
a node id.

```
python pipeline.py art --set SAMPLER.cfg=6.5 --set CHECKPOINT.ckpt_name=my_model.safetensors
python pipeline.py art --show-workflow        # lists every node and patchable field
```

Inputs wired to another node are never overwritten — that would silently
disconnect the graph — and you get a warning if you try.

## Shipped files

| File | What it needs |
|---|---|
| `art.json` | any SD1.5 checkpoint (default `v1-5-pruned-emaonly.safetensors`) |
| `acestep.json` | the ACE-Step checkpoint in `ComfyUI/models/checkpoints/` |

If the checkpoint name doesn't match yours, the pipeline lists what ComfyUI can
actually see rather than failing with a bare error.
