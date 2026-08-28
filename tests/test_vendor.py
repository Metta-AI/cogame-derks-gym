"""Tripwire: vendor/upstream/ is byte-pristine (AGENTS.md rule 1).

vendor/UPSTREAM.md records a sha256 per vendored file; this test
re-hashes the tree and asserts (a) the file sets match exactly and
(b) every hash matches. Any in-place edit of vendored source — instead
of a patch in sim/patches/ — fails here immediately.
"""

import hashlib
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = REPO_ROOT / "vendor" / "upstream"
UPSTREAM_MD = REPO_ROOT / "vendor" / "UPSTREAM.md"

# | `vendored file` | `upstream path` | `sha256` |
_ROW = re.compile(
    r"^\|\s*`([^`]+)`\s*\|\s*`[^`]+`\s*\|\s*`([0-9a-f]{64})`\s*\|",
    re.MULTILINE)


def test_vendor_upstream_matches_upstream_md_hashes():
    listed = dict(_ROW.findall(UPSTREAM_MD.read_text()))
    assert len(listed) >= 10, \
        f"UPSTREAM.md sha table parse broke (found {len(listed)} rows)"

    actual = {
        p.relative_to(UPSTREAM_DIR).as_posix()
        for p in UPSTREAM_DIR.rglob("*")
        if p.is_file() and not p.name.startswith(".")}
    assert actual == set(listed), (
        "vendor/upstream file set drifted from the UPSTREAM.md table: "
        f"extra={sorted(actual - set(listed))} "
        f"missing={sorted(set(listed) - actual)}")

    for name, expected in sorted(listed.items()):
        digest = hashlib.sha256(
            (UPSTREAM_DIR / name).read_bytes()).hexdigest()
        assert digest == expected, (
            f"vendor/upstream/{name} is not byte-pristine "
            f"(sha256 {digest} != recorded {expected}); vendored files "
            f"are never edited — changes go in sim/patches/")
