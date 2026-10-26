# Contributing

Thanks for your interest in HarnessTTT.

## Before opening a pull request

```bash
pip install -e ".[dev]"
pytest scripts/tests                     # contract tests; the whole evolve loop runs against a fake Slurm, no GPUs needed
```

The suite runs on a fresh clone without the task snapshot or the NexAU runtime: the fake-Slurm loop uses a one-task fixture
snapshot (`scripts/tests/site_env.py`), and the few tests that need NexAU (Python 3.12) skip when it is absent.  With the real
snapshot built (`scripts/setup/setup_dataset.py`) the task-registry tests run against it as well.

## What is welcome

- **New tasks**: a task folder (seed program with `# EVOLVE-BLOCK-START/END`, `evaluator.py` returning `combined_score` and
  `validity`, `config.yaml` with the task text) plus a record in `config/dataset/MANIFEST_eft.json` and, if the sources are
  public, an entry in `config/dataset/sources.json`.  Run `python3 scripts/tools/build_task_text_registry.py` afterwards and
  give the task an alias in `src/inner/tasks/aliases.py` (`TASK_IDS`, `DISPLAY`).
- **Harness components** under `src/inner/harness` (skills, tools, middleware) and proposer-side improvements under
  `src/outer` (gates, rewards, ratchet).
- **Portability fixes** for other Slurm sites (everything site-specific belongs in `config/local.env`).

## Conventions

- No site paths, account names or credentials in the tree (`scripts/tests/test_harnessttt_layout.py` checks `src/`).
- Pinned external code is described in `config/dependencies.lock.json`; update the lock together with the patch.
- Keep incident narratives out of templates and workers; a short neutral comment is enough.
