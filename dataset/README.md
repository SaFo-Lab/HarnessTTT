# dataset/

`self_adapt_harness/` is the frozen evaluator snapshot the pipeline runs against (git-ignored payload, 1.5 GB, 889 files).
Copied on 2026-09-24 from the upstream dataset snapshot (byte-identical, `rsync -rlc`).
`MANIFEST_eft.json` sha256 and `DATA_PROVENANCE.json` sha256 are recorded in `config/dependencies.lock.json` and checked by
`scripts/setup/preflight.sh`.

Contents (see `self_adapt_harness/README.md` for the upstream protocol):

- `MANIFEST_eft.json` -- the frozen task protocol (math tasks, ADRS tasks loaded by `inner.tasks.eft_task`, SimpleTES
  cross-references).  `inner.tasks.eft_task.load_tasks()` yields 15 tasks from this snapshot.
- `raw/` -- task sources copied from immutable upstream commits: evolution-fine-tuning / SkyDiscover
  `aac2e79be773715ab35b7945f5d9028e46675f02`, SimpleTES `b7e0367b5ab19554fb40859e8069e091bb8c4ca2`.
- `heldout/` -- held-out task material.
- `runtime/` -- import shims (`skydiscover_min`, `simpletes_min`) and `ahc_native/` (the docker/time shim the AHC evaluators
  expect; `config/env.sh` exports `SAH_AHC_NATIVE_BIN` from here and the round template mounts `.../time` into the container).
- `DATA_PROVENANCE.json` -- URL, byte count and sha256 of the downloaded EPLB / LLM-SQL payloads.

The code finds this directory through `SAH_DATASET_ROOT`, which `config/env.sh` sets to `dataset/self_adapt_harness`
(the in-code default of `src/inner/tasks/eft_task.py` is the same relative path).

One symlink points outside the tree: `raw/simpletes-47d3413/datasets/zapbench/whole_brain_forecasting/zapbench_cache`
(5.9 GB cache of the optional zapbench task, not used by the math tasks).  Recreate or drop it when relocating.

To refresh the task-text registry after changing the snapshot: `python3 scripts/tools/build_task_text_registry.py`.
