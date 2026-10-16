"""Anti-leak text guards (Adaptive port).

Two dangers when free-form text reaches the proposer:

1. **Positive-result claims survive as facts.** A high raw score that actually
   regressed against its matched parent must not be summarized to the proposer
   as "a successful improvement" — that teaches the policy to chase a signal
   the controller already rejected. ``neutralize_claims`` deterministically
   demotes result-claim words to neutral verbs.

2. **Champion/evaluator internals leak.** Static validation already gates generated
   tool code; ``strip_leak_markers`` is the text-side complement, redacting any
   phrase that would import solution/eval knowledge the frozen M0 must derive
   itself.

Both are pure, deterministic string transforms — no model in the loop — so they
can't themselves introduce knowledge. Gated by ``leakage_guard.*`` in the
campaign config (env ``SAH_LEAK_NEUTRALIZE`` / default on).
"""
from __future__ import annotations

import re
from typing import List

# result-claim words -> neutral replacements. Applied case-insensitively while
# preserving the original casing of the first letter.
_CLAIM_MAP = {
    "successful": "attempted",
    "successfully": "",
    "success": "attempt",
    "improvement": "change",
    "improved": "changed",
    "improves": "changes",
    "improve": "change",
    "gain": "delta",
    "gains": "deltas",
    "better": "different",
    "best": "recorded",
    "outperform": "differ from",
    "outperforms": "differs from",
    "beats": "differs from",
    "winning": "recorded",
    "breakthrough": "change",
    "effective": "applied",
    "works well": "was applied",
    "promising": "untested",
}

# phrases that would leak eval/solution knowledge (text-side of the code
# code-level markers). Redacted, not neutralized — their presence is a bug.
_LEAK_MARKERS = ("ground truth", "expected output", "reference solution",
                 "the answer is", "optimal solution", "target score", "sota",
                 "finch", "evaluator internals")


def _preserve_case(original: str, replacement: str) -> str:
    if not replacement:
        return ""
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def neutralize_claims(text: str) -> str:
    """Demote positive-result claim words to neutral verbs so unproven prose
    can't survive as a positive-result fact reaching the proposer."""
    if not text:
        return text
    out = text
    # longest keys first so "successfully" isn't half-matched by "success"
    for word in sorted(_CLAIM_MAP, key=len, reverse=True):
        repl = _CLAIM_MAP[word]

        def _sub(m: "re.Match") -> str:
            r = _preserve_case(m.group(0), repl)
            return r if r else ""

        out = re.sub(rf"\b{re.escape(word)}\b", _sub, out, flags=re.IGNORECASE)
    # collapse any double spaces left by empty replacements
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def strip_leak_markers(text: str) -> List[str]:
    """Return the leak markers present in ``text`` (empty = clean). Callers
    redact/drop rather than ship flagged text — matches the deterministic
    fail-closed policy."""
    low = (text or "").lower()
    return [m for m in _LEAK_MARKERS if m in low]


def sanitize(text: str, *, neutralize: bool = True) -> str:
    """Full guard: drop if a leak marker is present (fail-closed), else
    optionally neutralize claim words. Returns '' when a marker is found."""
    if strip_leak_markers(text):
        return ""
    return neutralize_claims(text) if neutralize else text


# --- anti-code-injection (curated/analyst note hardening) -------------------
#
# Marker words catch evaluator/target-score leakage, but a note that smuggles
# a *program* (the LLM-SQL incident: a verified solution pasted with an
# adopt-verbatim instruction) contains no marker word at all.  Structural
# facts about a scoring rule never need executable code, so any strong code
# signal in a note is treated as attempted solution injection and the note is
# dropped fail-closed.

_CODE_FENCE = "```"
# Statement keywords match only at line start (code style); mid-sentence
# English ("derived from the run", "a class of schedules") stays legal.
_CODE_LINE_TOKENS = ("def ", "import ", "from ", "lambda ", "return ",
                     "class ", "@")
_CODE_ANYWHERE = ("verbatim",)  # adopt-VERBATIM has no legitimate use
NOTE_MAX_CHARS = 1500


def code_signals(text: str) -> List[str]:
    """Names the code/injection signals found in a free-form note."""
    found: List[str] = []
    raw = text or ""
    low = raw.lower()
    if _CODE_FENCE in low:
        found.append("code_fence")
    for tok in _CODE_ANYWHERE:
        if tok in low:
            found.append(f"token:{tok}")
    for line in low.splitlines():
        stripped = line.lstrip(" \t->*")
        for tok in _CODE_LINE_TOKENS:
            if stripped.startswith(tok):
                found.append(f"line_start:{tok.strip() or tok}")
                break
    if len(raw) > NOTE_MAX_CHARS:
        found.append(f"over_length:{len(raw)}>{NOTE_MAX_CHARS}")
    return found


def sanitize_note(text: str) -> str:
    """Full guard for curated/analyst notes: marker scrub + code rejection.

    Returns the sanitized note, or "" (drop the whole note) when any code
    signal is present -- a note is telemetry, never a program.
    """
    if not text:
        return ""
    signals = code_signals(text)
    if signals:
        return ""
    return sanitize(text, neutralize=True)
