"""Program manipulation for EVOLVE-BLOCK discovery tasks.

The EFT / SimpleTES / OpenEvolve task family represents a candidate as a full
program file containing a region between ``# EVOLVE-BLOCK-START`` and
``# EVOLVE-BLOCK-END``. Only that region is evolved; everything else (imports,
the fixed entry function the evaluator calls) is preserved.

This module:
  * splits a program into (prefix, block, suffix) around the markers;
  * reassembles a program from a new block;
  * extracts an evolved program from an LLM response, in either
    full-rewrite (```python ...```) or SEARCH/REPLACE diff form.

The diff/full-rewrite parsing mirrors skydiscover.utils.code_utils (Apache-2.0)
so generations from a Finch/Qwen model trained on that format apply cleanly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

BLOCK_START = "# EVOLVE-BLOCK-START"
BLOCK_END = "# EVOLVE-BLOCK-END"


@dataclass
class SplitProgram:
    """A program split around its (first) EVOLVE-BLOCK region."""

    prefix: str  # text before BLOCK_START (excluding the marker line)
    block: str  # the evolved region (between the marker lines, exclusive)
    suffix: str  # text after BLOCK_END (excluding the marker line)
    had_markers: bool

    def assemble(self, new_block: Optional[str] = None) -> str:
        """Rebuild the full program, optionally replacing the block."""
        body = self.block if new_block is None else new_block
        body = body.strip("\n")
        if not self.had_markers:
            # No markers in the original: the whole thing is the block.
            return body + "\n"
        return f"{self.prefix}{BLOCK_START}\n{body}\n{BLOCK_END}{self.suffix}"


def split_program(program: str) -> SplitProgram:
    """Split ``program`` around the first EVOLVE-BLOCK region.

    If the markers are absent, the whole program is treated as the block
    (mirrors skydiscover.utils.prepare.prepare_program behaviour).
    """
    if BLOCK_START not in program or BLOCK_END not in program:
        return SplitProgram(prefix="", block=program.strip("\n"), suffix="", had_markers=False)

    start = program.index(BLOCK_START)
    end = program.index(BLOCK_END)
    prefix = program[:start]  # preserved verbatim (imports etc.)
    suffix = program[end + len(BLOCK_END):]  # preserved verbatim (fixed entry fn)
    # Strip DUPLICATE end markers that accumulate in the suffix across
    # inheritance rounds (assemble re-adds exactly one END). Without this, each
    # split->assemble cycle leaks one extra "# EVOLVE-BLOCK-END" line forever.
    kept = [ln for ln in suffix.splitlines() if ln.strip() != BLOCK_END]
    real = "\n".join(kept).strip("\n")
    suffix = ("\n" + real) if real else ""
    # block = text strictly between the end of the START marker line and BLOCK_END
    block = program[start:end]
    block = block[block.index("\n") + 1:] if "\n" in block else ""
    return SplitProgram(prefix=prefix, block=block.strip("\n"), suffix=suffix, had_markers=True)


# --------------------------------------------------------------------------- #
# LLM response -> code
# --------------------------------------------------------------------------- #
def parse_full_rewrite(llm_response: str, language: str = "python") -> Optional[str]:
    """Extract a fenced code block from an LLM full-rewrite response."""
    pat = r"```" + re.escape(language) + r"\s*\n(.*?)```"
    m = re.findall(pat, llm_response, re.DOTALL)
    if m:
        return m[0].strip()
    m = re.findall(r"```(.*?)```", llm_response, re.DOTALL)
    if m:
        # drop a leading bare language token if present
        code = m[0]
        code = re.sub(r"^[a-zA-Z0-9_+-]*\n", "", code, count=1)
        return code.strip()
    return None


_DIFF_PATTERN = re.compile(
    r"<<<<<<< SEARCH\s*\n(.*?)=======\s*\n(.*?)>>>>>>> REPLACE",
    re.DOTALL,
)


def extract_diffs(diff_text: str) -> List[Tuple[str, str]]:
    """Parse SEARCH/REPLACE blocks -> list of (search, replace)."""
    return [(s.rstrip("\n"), r.rstrip("\n")) for s, r in _DIFF_PATTERN.findall(diff_text)]


def apply_diff(original: str, diff_text: str) -> Tuple[str, int]:
    """Apply SEARCH/REPLACE diff blocks to ``original``.

    Returns (new_program, num_applied). A block whose SEARCH text is not found
    verbatim is skipped (num_applied counts only the ones that matched).
    """
    lines = original.split("\n")
    applied = 0
    for search, replace in extract_diffs(diff_text):
        s_lines = search.split("\n")
        r_lines = replace.split("\n")
        for i in range(len(lines) - len(s_lines) + 1):
            if lines[i:i + len(s_lines)] == s_lines:
                lines[i:i + len(s_lines)] = r_lines
                applied += 1
                break
    return "\n".join(lines), applied


@dataclass
class EditResult:
    program: Optional[str]  # the assembled full program, or None on failure
    mode: str  # "full_rewrite" | "diff" | "failed"
    note: str


def build_candidate(
    current_program: str,
    llm_response: str,
    *,
    diff_based: bool,
    language: str = "python",
) -> EditResult:
    """Turn an LLM response into a full candidate program.

    diff_based: apply SEARCH/REPLACE diffs to the *full current program*.
    else:       treat the response as a full rewrite of the EVOLVE-BLOCK.
    Falls back across modes so a well-formed generation is not wasted.
    """
    if diff_based and "<<<<<<< SEARCH" in llm_response:
        new_prog, n = apply_diff(current_program, llm_response)
        if n > 0:
            return EditResult(new_prog, "diff", f"applied {n} diff block(s)")
        # fall through to full-rewrite salvage

    code = parse_full_rewrite(llm_response, language)
    if code is None:
        return EditResult(None, "failed", "no code block or diff found in response")

    split = split_program(current_program)
    if BLOCK_START in code and BLOCK_END in code:
        # model returned a full program with markers -> take its block
        code = split_program(code).block
    assembled = split.assemble(code)
    return EditResult(assembled, "full_rewrite", "spliced full-rewrite into EVOLVE-BLOCK")
