"""Every command-line entry point must at least parse its own arguments.

`swing_gcv.py` shipped broken for hours because an added argument was registered
on `p` where the parser is named `ap`. Nothing caught it: the failure is at
argparse-construction time, so it needs `main()` to actually run, and no unit
test invokes these modules as programs. It only surfaced when a 24-hour job was
relaunched and died on startup.

`--help` exercises exactly the path that broke -- import, parser construction,
every `add_argument` -- and exits before doing any work, so this is fast and has
no side effects.
"""
import subprocess
import sys
from pathlib import Path

import pytest


# subprocess-spawns every CLI entry point; see tests/conftest.py for the opt-in flags.
pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[1]

def _discover():
    """Every script under src/ that builds an argparse parser and is runnable.

    Discovered rather than listed: a hand-maintained list is exactly how
    `precompute_prott5_datasets.py` came to raise UnboundLocalError on every
    invocation without any test noticing.
    """
    out = []
    for path in sorted((REPO / "src").rglob("*.py")):
        if path.name.startswith("__"):
            continue
        text = path.read_text()
        if "ArgumentParser(" not in text or '__name__' not in text:
            continue
        out.append(str(path.relative_to(REPO)))
    return out


ENTRYPOINTS = _discover()


@pytest.mark.parametrize("script", ENTRYPOINTS, ids=lambda s: Path(s).stem)
def test_help_succeeds(script):
    path = REPO / script
    if not path.exists():
        pytest.skip(f"{script} not present")
    r = subprocess.run([sys.executable, str(path), "--help"],
                       capture_output=True, text=True, cwd=REPO, timeout=180)
    assert r.returncode == 0, (
        f"{script} --help failed ({r.returncode}):\n{r.stderr[-1500:]}")
    assert "usage" in (r.stdout + r.stderr).lower()


def test_the_specific_regression_that_motivated_this():
    """`swing_gcv.py` must expose both --fold-jobs and --resume."""
    r = subprocess.run([sys.executable, str(REPO / "src/evaluation/swing_gcv.py"), "--help"],
                       capture_output=True, text=True, cwd=REPO, timeout=180)
    assert r.returncode == 0, r.stderr[-1500:]
    for flag in ("--fold-jobs", "--resume"):
        assert flag in r.stdout, f"{flag} missing from swing_gcv --help"
