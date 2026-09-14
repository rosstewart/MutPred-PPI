"""The docs must be usable by someone who is not the author.

Three classes of defect this catches, all of which were present:

  * figure numbers invented or drifted from the manuscript, so a reader holding
    the paper could not find the command that made a figure;
  * cited line numbers that moved when the notebook was edited;
  * house-style violations that crept back one commit at a time.
"""
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def tracked_docs():
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=REPO,
                         capture_output=True, text=True).stdout.split()
    return [REPO / p for p in out if (REPO / p).exists()]


class TestStyle:
    """House style, as greps that must find nothing."""

    #: pattern -> why it is banned
    BANNED = {
        "—": "em dash",
        r"\bships\b": "jargon; say 'includes'",
        r"\bships with\b": "jargon; say 'includes'",
        r"\bdoubles as\b": "filler",
        r"\bdrop-in\b": "jargon",
        r"\bfor free\b": "jargon",
        "10.5281/zenodo.17645488": "obsolete DOI",
        "10.5281/zenodo.18701748": "obsolete DOI",
    }

    @pytest.mark.parametrize("pattern,reason", sorted(BANNED.items()))
    def test_pattern_absent(self, pattern, reason):
        offenders = []
        for doc in tracked_docs():
            for i, line in enumerate(doc.read_text().splitlines(), 1):
                if re.search(pattern, line):
                    offenders.append(f"{doc.relative_to(REPO)}:{i}")
        assert not offenders, f"{reason}: {offenders[:8]}"


class TestLinks:
    def test_every_relative_link_resolves(self):
        broken = []
        for doc in tracked_docs():
            text = doc.read_text()
            for m in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
                target = m.group(1)
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                target = target.split("#", 1)[0]
                if not target:
                    continue
                if not (doc.parent / target).resolve().exists():
                    broken.append(f"{doc.relative_to(REPO)} -> {target}")
        assert not broken, f"broken relative links: {broken}"

    def test_no_reference_to_a_merged_away_doc(self):
        """SETUP.md, ZENODO.md and DATA_SOURCES.md were merged into two files."""
        gone = ("docs/SETUP.md", "docs/ZENODO.md", "docs/DATA_SOURCES.md")
        tracked = subprocess.run(["git", "ls-files"], cwd=REPO,
                                 capture_output=True, text=True).stdout.split()
        offenders = []
        for rel in tracked:
            if rel == "tests/test_docs.py":
                continue          # this file names them in order to ban them
            p = REPO / rel
            if not p.is_file() or p.suffix not in (".md", ".py", ".sh", ".gitignore"):
                continue
            try:
                text = p.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            for name in gone:
                if name in text:
                    offenders.append(f"{rel} -> {name}")
        assert not offenders, f"references to merged-away docs: {offenders}"


class TestFigureNumbering:
    """Numbers must come from the manuscript, not from the repository."""

    def test_checker_passes(self):
        from verification import check_figure_numbering as chk

        numbering = chk.manuscript_numbering(REPO)
        if not numbering:
            pytest.skip("manuscript sources not present")
        r = subprocess.run(
            ["python", str(REPO / "src/verification/check_figure_numbering.py"), "--quiet"],
            cwd=REPO, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr

    def test_manuscript_numbering_is_derivable(self):
        from verification import check_figure_numbering as chk

        numbering = chk.manuscript_numbering(REPO)
        if not numbering:
            pytest.skip("manuscript sources not present")
        # Sanity: the three headline figures keep their numbers.
        assert numbering["fig:sahni_fragoza_cv"]["number"] == "Fig 3"
        assert numbering["fig:blind_test_roc"]["number"] == "Fig 4"
        assert numbering["fig:enrichment"]["number"] == "Fig 5"
