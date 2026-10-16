# AHC native evaluator compatibility layer

The frozen SimpleTES evaluators invoke
`yimjk/ale-bench:cpp20-202301` through Docker. Pyxis/enroot compute containers
do not expose Docker or Podman. This directory supplies a deliberately narrow
`docker run` compatibility command that accepts only the exact SimpleTES AHC
invocation and runs the pinned `docker_runner.py` in the current Slurm
container.

Verified environment:

- compute architecture: `x86_64`
- official `ahc039_tester` and `ahc058_tester`: x86-64 and dynamically loadable
- image compiler: Ubuntu `g++ 11.4.0`
- upstream requested compiler: `g++-12`
- upstream runner always links GMP; the container image has no GMP linker
  libraries, so the wrapper drops `-lgmpxx -lgmp` for the frozen seeds, which
  contain no GMP includes or calls
- Docker/Podman: unavailable
- `timeout` and `prlimit`: available
- the image `prlimit` accepts only integral CPU seconds, while the upstream
  runner passes values such as `3.1`; the adjacent wrapper rounds only that
  limit upward (`3.1 -> 4`) and leaves the independent wall timeout unchanged
- the image has no `/usr/bin/time`, while the upstream runner calls that
  absolute path and then runs `sync`, which masks the resulting launch failure;
  the round template bind-mounts the adjacent executable `time` shim read-only at
  `/usr/bin/time`
- the `time` shim accepts only the runner's `-f FORMAT -o PROFILE COMMAND...`
  shape, executes the candidate without a shell, emits the JSON resource
  profile consumed by the frozen runner, and propagates the candidate status
- the upstream bare `sync` would flush every mount, including the shared
  parallel filesystem, and storage latency would fall inside the candidate wall timeout;
  the adjacent `sync` shim accepts no arguments and runs
  `/usr/bin/sync -f "$TMPDIR"`, limiting the flush to node-local `/tmp`, where
  every case output and timing profile is created

The evaluator and runner source files remain byte-identical to SimpleTES
commit `b7e0367b5ab19554fb40859e8069e091bb8c4ca2`. Compiler/linker substitutions
are isolated in the adjacent `g++-12` wrapper; process-launch substitutions are
isolated in the adjacent `prlimit`, `time`, and `sync` wrappers plus the
explicit read-only Pyxis mount. Baseline score/acceptance validation is a required gate
before this layer is used for experiments.

As required by SimpleTES `datasets/ahc/setup.sh`, both copied tester binaries
are marked executable after snapshot creation; their file bytes are unchanged.
