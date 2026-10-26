#!/usr/bin/env python3
"""Build (or verify) the HarnessTTT task snapshot ``dataset/self_adapt_harness`` from its pinned public sources.

    python3 scripts/setup/setup_dataset.py                         # core tier: the 11-task protocol (math, Hadamard, AHC, ADRS), ~350 MB
    python3 scripts/setup/setup_dataset.py --tiers core,gpu        # + GPU MODE kernel tasks (trimul, trimul_lb, mla_decode)
    python3 scripts/setup/setup_dataset.py --tiers core,gpu,extra  # + SimpleTES astro / bio / ZAPBench sources (see --tiers help)
    python3 scripts/setup/setup_dataset.py --verify                # check an existing snapshot against config/dataset/CHECKSUMS.json

The recipe is config/dataset/sources.json: every repository is fetched at one commit (git >= 2.20, network), the listed
subtrees are copied, the locally authored files under config/dataset/overlay/ are laid over them (evaluator shims, the
kernel tasks' initial programs and the leaderboard-protocol evaluator), the data files are downloaded and checked
(sha256), and the manifest config/dataset/MANIFEST_eft.json is installed.  The result is verified file by file against
config/dataset/CHECKSUMS.json.  Destination: --dest, else $SAH_DATASET_ROOT / $SAH_DATA_ROOT, else
<repo>/dataset/self_adapt_harness.  The upstream checkouts (~215 MB) stay under <repo>/.deps/dataset_sources (--work) so a
rebuild or an added tier reuses them; --clean-work removes them.  A snapshot that already verifies is left untouched.
Only the standard library is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(os.environ.get("HARNESSTTT_ROOT") or Path(__file__).resolve().parents[2])
CFG = ROOT / "config" / "dataset"
TIERS = ("core", "gpu", "extra")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store", ".venv", ".git")


def log(msg: str) -> None:
    print(f"[setup_dataset {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: List[str], **kw) -> subprocess.CompletedProcess:
    out = subprocess.run(cmd, text=True, capture_output=True, **kw)
    if out.returncode != 0:
        raise SystemExit(f"command failed ({out.returncode}): {' '.join(cmd)}\n{(out.stderr or out.stdout).strip()[-800:]}")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------------------------------------------------
def fetch_repo(name: str, url: str, commit: str, work: Path) -> Path:
    """A checkout of ``url`` at ``commit`` under ``work/name`` (reused when already there)."""
    repo = work / name
    if (repo / ".git").is_dir():
        probe = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, capture_output=True)
        if probe.returncode == 0 and probe.stdout.strip() == commit:
            log(f"{name}: reusing checkout {repo}")
            return repo
        log(f"{name}: removing an incomplete or stale checkout at {repo}")
        shutil.rmtree(repo)
    repo.mkdir(parents=True)
    run(["git", "init", "-q", str(repo)])
    run(["git", "-C", str(repo), "remote", "add", "origin", url])
    log(f"{name}: fetching {commit[:10]} from {url}")
    shallow = subprocess.run(["git", "-C", str(repo), "fetch", "-q", "--depth", "1", "origin", commit], text=True, capture_output=True)
    if shallow.returncode != 0:   # GitHub serves single commits by hash; other hosts may refuse -> full history
        log(f"{name}: shallow fetch of the commit refused ({shallow.stderr.strip()[-120:]}), fetching the full history")
        run(["git", "-C", str(repo), "fetch", "-q", "origin"])
    run(["git", "-C", str(repo), "-c", "advice.detachedHead=false", "checkout", "-q", commit])
    return repo


def copy_rules(repo: Path, rules: Iterable[List[str]], dest: Path) -> List[str]:
    copied: List[str] = []
    for src_rel, dst_rel in rules:
        src, dst = repo / src_rel, dest / dst_rel
        if not src.exists():
            raise SystemExit(f"{repo.name}: {src_rel} is not in the repository at this commit")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=IGNORE, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        copied.append(dst_rel)
    return copied


def apply_overlay(spec: Dict, tiers: List[str], dest: Path) -> int:
    overlay = ROOT / spec["overlay"]["dir"]
    tier_of = spec["overlay"]["tiers"]
    n = 0
    for path in sorted(p for p in overlay.rglob("*") if p.is_file()):
        rel = path.relative_to(overlay).as_posix()
        tier = next((t for prefix, t in tier_of.items() if rel.startswith(prefix)), "core")
        if tier not in tiers:
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        n += 1
    return n


def download(url: str, target: Path, sha256: Optional[str], expected_bytes: Optional[int]) -> Dict:
    if target.is_file() and (sha256 is None or sha256_of(target) == sha256):
        log(f"download: {target.name} present and verified")
        return {"path": None, "sha256": sha256 or sha256_of(target), "bytes": target.stat().st_size, "url": url}
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    log(f"download: {url} -> {target}")
    req = urllib.request.Request(url, headers={"User-Agent": "harnessttt-setup-dataset/1.0"})
    for attempt in range(1, 4):   # a stalled connection is retried from the start (the files are < 250 MB)
        h = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
                for chunk in iter(lambda: resp.read(1 << 20), b""):
                    out.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
            break
        except (OSError, urllib.error.URLError) as exc:
            if attempt == 3:
                tmp.unlink(missing_ok=True)
                raise SystemExit(f"download {url} failed after 3 attempts: {exc}")
            log(f"download: attempt {attempt} failed ({exc}); retrying")
    digest = h.hexdigest()
    if sha256 is not None and digest != sha256:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"download {url}: sha256 {digest} != expected {sha256}")
    if expected_bytes is not None and size != expected_bytes:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"download {url}: {size} bytes != expected {expected_bytes}")
    os.replace(tmp, target)
    return {"path": None, "sha256": digest, "bytes": size, "url": url}


# ---------------------------------------------------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------------------------------------------------
def verify(dest: Path, tiers: List[str], checksums: Dict) -> int:
    files = {rel: rec for rel, rec in checksums["files"].items() if rec["tier"] in tiers}
    missing, mismatched, ok = [], [], 0
    for rel, rec in sorted(files.items()):
        path = dest / rel
        if not path.is_file():
            missing.append(rel)
        elif sha256_of(path) != rec["sha256"]:
            mismatched.append(rel)
        else:
            ok += 1
    log(f"verify ({','.join(tiers)}): {ok} files match, {len(missing)} missing, {len(mismatched)} differ")
    for rel in missing[:20]:
        print(f"  missing   {rel}")
    for rel in mismatched[:20]:
        print(f"  differs   {rel}")
    if len(missing) + len(mismatched) > 40:
        print(f"  ... {len(missing) + len(mismatched) - 40} more")
    return 0 if not missing and not mismatched else 1


def write_checksums(dest: Path, spec: Dict, out: Path) -> None:
    """Developer step: record sha256 of every file a build produces (from a snapshot built or validated by hand)."""
    import fnmatch
    skip = {"MANIFEST_eft.json", "README.md", "DATA_PROVENANCE.json"}
    generated = tuple(spec.get("generated_not_checksummed", []))

    def is_generated(rel: str) -> bool:
        return any(fnmatch.fnmatch(rel, g) or rel.startswith(g.rstrip("*")) for g in generated)
    rules = []
    for repo in spec["repos"]:
        for _src, dst in repo["copy"]:
            rules.append((dst, repo["tier"]))
    for d in spec["downloads"]:
        rules.append((d["path"], d["tier"]))
    overlay = ROOT / spec["overlay"]["dir"]
    for p in overlay.rglob("*"):
        if p.is_file():
            rel = p.relative_to(overlay).as_posix()
            tier = next((t for prefix, t in spec["overlay"]["tiers"].items() if rel.startswith(prefix)), "core")
            rules.append((rel, tier))
    files = {}
    for path in sorted(p for p in dest.rglob("*") if p.is_file() and not p.is_symlink()):
        rel = path.relative_to(dest).as_posix()
        if "__pycache__" in rel or rel.endswith((".pyc", ".pyo")) or rel in skip or rel.endswith("/SOURCE.json") or ".DS_Store" in rel:
            continue
        if is_generated(rel):
            continue
        tier = next((t for dst, t in rules if rel == dst or rel.startswith(dst.rstrip("/") + "/")), None)
        if tier is None:
            continue   # not produced by the recipe (local extras)
        files[rel] = {"sha256": sha256_of(path), "bytes": path.stat().st_size, "tier": tier}
    out.write_text(json.dumps({"schema": "harnessttt-dataset-checksums/1.0", "root": "dataset/self_adapt_harness", "files": files},
                              indent=1, sort_keys=True) + "\n")
    by_tier = {}
    for rec in files.values():
        by_tier[rec["tier"]] = by_tier.get(rec["tier"], 0) + 1
    log(f"checksums written: {len(files)} files {by_tier} -> {out}")


# ---------------------------------------------------------------------------------------------------------------------
def build(dest: Path, tiers: List[str], work: Path, spec: Dict, no_download: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    provenance = {"schema_version": 2, "dataset_snapshot": dest.name, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "tiers": tiers, "sources": {}, "files": []}
    for repo in spec["repos"]:
        if repo["tier"] not in tiers:
            continue
        checkout = fetch_repo(repo["name"], repo["url"], repo["commit"], work)
        copied = copy_rules(checkout, repo["copy"], dest)
        log(f"{repo['name']}: copied {len(copied)} subtrees into {dest}")
        provenance["sources"][repo["name"]] = {"url": repo["url"], "commit": repo["commit"], "copied": copied}
        (dest / repo["snapshot_dir"]).mkdir(parents=True, exist_ok=True)
        (dest / repo["snapshot_dir"] / "SOURCE.json").write_text(json.dumps(
            {"repo": repo["url"], "commit": repo["commit"], "copied": copied, "vendored_by": "scripts/setup/setup_dataset.py"}, indent=1) + "\n")
        for note in repo.get("task_setup", []):
            log(f"{repo['name']}: task data is prepared by its own script: {note}")
    n = apply_overlay(spec, tiers, dest)
    log(f"overlay: {n} locally authored files installed (config/dataset/overlay)")
    for d in spec["downloads"]:
        if d["tier"] not in tiers:
            continue
        target = dest / d["path"]
        if no_download:
            if not target.is_file():
                log(f"--no-download: {d['path']} is missing")
            continue
        rec = download(d["url"], target, d.get("sha256"), d.get("bytes"))
        rec["path"] = d["path"]
        provenance["files"].append(rec)
    eph = dest / "raw/simpletes-47d3413/datasets/astrodynamics/data/ephemerides"
    eph_files = [d for d in spec["downloads"] if d["path"].startswith("raw/simpletes-47d3413/datasets/astrodynamics/data/ephemerides/")]
    if eph.is_dir() and all((dest / d["path"]).is_file() for d in eph_files):
        # the astrodynamics tasks read this marker, written by their own download script in the same format
        (eph / ".verified").write_text("\n".join(f"{d['sha256']}  {Path(d['path']).name}" for d in eph_files) + "\n")
    shutil.copy2(CFG / "MANIFEST_eft.json", dest / "MANIFEST_eft.json")
    shutil.copy2(CFG / "README.snapshot.md", dest / "README.md")
    (dest / "DATA_PROVENANCE.json").write_text(json.dumps(provenance, indent=1) + "\n")
    for shim in (dest / "runtime" / "ahc_native" / "bin").glob("*"):
        shim.chmod(shim.stat().st_mode | 0o111)
    log(f"snapshot ready: {dest} (tiers {','.join(tiers)})")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=os.environ.get("SAH_DATASET_ROOT") or os.environ.get("SAH_DATA_ROOT") or str(ROOT / "dataset" / "self_adapt_harness"),
                    help="snapshot directory (default: $SAH_DATASET_ROOT, $SAH_DATA_ROOT, else <repo>/dataset/self_adapt_harness)")
    ap.add_argument("--tiers", default="core", help="comma-separated subset of core,gpu,extra (default core)")
    ap.add_argument("--work", default=str(ROOT / ".deps" / "dataset_sources"),
                    help="where the upstream checkouts live between runs (default <repo>/.deps/dataset_sources, git-ignored); a rebuild reuses them")
    ap.add_argument("--keep-work", action="store_true", help="(default behaviour; kept for compatibility)")
    ap.add_argument("--clean-work", action="store_true", help="remove the checkouts after a successful build")
    ap.add_argument("--no-download", action="store_true", help="skip the data downloads (verification then reports them missing)")
    ap.add_argument("--verify", action="store_true", help="only verify --dest against config/dataset/CHECKSUMS.json")
    ap.add_argument("--write-checksums", action="store_true", help="developer: regenerate CHECKSUMS.json from --dest")
    args = ap.parse_args(argv)
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    bad = [t for t in tiers if t not in TIERS]
    if bad or not tiers:
        ap.error(f"unknown tiers {bad}; choose from {TIERS}")
    if "core" not in tiers:
        tiers.insert(0, "core")
    spec = json.loads((CFG / "sources.json").read_text())
    dest = Path(args.dest).expanduser().resolve()
    if args.write_checksums:
        write_checksums(dest, spec, CFG / "CHECKSUMS.json")
        return 0
    checksums = json.loads((CFG / "CHECKSUMS.json").read_text())
    if args.verify:
        return verify(dest, tiers, checksums)
    if dest.is_dir() and verify(dest, tiers, checksums) == 0 and not args.no_download:
        log(f"{dest} already verifies for tiers {','.join(tiers)}; nothing to build (use --verify to re-check)")
        return 0
    work = Path(args.work).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    build(dest, tiers, work, spec, args.no_download)
    rc = verify(dest, tiers, checksums)
    if rc == 0 and args.clean_work:
        shutil.rmtree(work, ignore_errors=True)
    if rc == 0:
        log("every file matches config/dataset/CHECKSUMS.json; now run: python3 scripts/tools/build_task_text_registry.py --check")
    return rc


if __name__ == "__main__":
    sys.exit(main())
