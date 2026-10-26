# dataset/

`self_adapt_harness/` is the task snapshot the pipeline runs against: task texts, seed programs, evaluators and the runtime
shims.  It is a git-ignored payload built from pinned public sources:

```bash
python3 scripts/setup/setup_dataset.py                        # core tier: the 11-task protocol (~350 MB, needs git + network)
python3 scripts/setup/setup_dataset.py --tiers core,gpu       # + GPU MODE kernel tasks (trimul, trimul_lb, mla_decode)
python3 scripts/setup/setup_dataset.py --tiers core,gpu,extra # + SimpleTES astro / bio / ZAPBench sources
python3 scripts/setup/setup_dataset.py --verify               # re-check an existing snapshot against config/dataset/CHECKSUMS.json
python3 scripts/tools/build_task_text_registry.py --check     # the task texts match the pinned registry
```

The recipe lives in `config/dataset/`: `sources.json` (repositories, commits, subtrees, downloads), `MANIFEST_eft.json`
(the task protocol installed into the snapshot), `overlay/` (locally authored files: evaluator shims, the AHC compatibility
commands, the kernel tasks' initial programs and leaderboard-protocol evaluator), `CHECKSUMS.json` (sha256 of every file a
build produces) and `README.snapshot.md` (installed as the snapshot's README; it documents the layout).

The code finds the snapshot through `SAH_DATASET_ROOT` (`config/env.sh` defaults it to `dataset/self_adapt_harness` and keeps
the scripts' `SAH_DATA_ROOT` equal to it; set either in `config/local.env` to relocate the snapshot, and point the builder's
`--dest` at the same place).  `scripts/setup/preflight.sh` checks the manifest hash
recorded in `config/dependencies.lock.json`.

The `extra` tier copies sources only; the astrodynamics ephemerides are downloaded by the builder, while the scRNA splits and
the 6 GB ZAPBench cache are produced by those tasks' own `setup.sh` (see `config/dataset/sources.json`, `task_setup`).
