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
