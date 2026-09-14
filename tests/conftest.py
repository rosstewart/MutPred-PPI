"""Shared pytest fixtures.

No path hacking needed: `pip install -e .` puts `src/` on sys.path, so
`import utils.sequences`, `import contact_graphs`, etc. work directly, exactly
as they do from any script in `src/`.
"""
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


# ── Opt-in test tiers ─────────────────────────────────────────────────────────
# The default suite must pass on a fresh clone with no datasets/ tree and no
# GPU.  Two groups of tests cannot meet that bar, so they are skipped unless
# explicitly requested:
#
#   slow           spawns subprocesses or fits models (tens of seconds each)
#   requires_data  reads the real embedding/subgraph caches, prediction TSVs
#                  or archived ablation artifacts
#
# CI and the pre-deposit check run `pytest tests/ --run-slow --run-data`.

_TIERS = {
    "slow": ("--run-slow", "spawns subprocesses or fits models"),
    "requires_data": ("--run-data", "needs the full datasets/ tree or the variant-DB caches"),
}


def pytest_addoption(parser):
    for _mark, (flag, help_text) in _TIERS.items():
        parser.addoption(flag, action="store_true", default=False,
                         help=f"run tests marked {_mark}: {help_text}")


def pytest_configure(config):
    for mark, (flag, help_text) in _TIERS.items():
        config.addinivalue_line("markers", f"{mark}: {help_text} (opt in with {flag})")


def pytest_collection_modifyitems(config, items):
    for mark, (flag, _help) in _TIERS.items():
        if config.getoption(flag):
            continue
        skip = pytest.mark.skip(reason=f"needs {flag}")
        for item in items:
            if mark in item.keywords:
                item.add_marker(skip)
