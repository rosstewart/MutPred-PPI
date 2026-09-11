"""Every module under src/ must import without side effects.

Two distinct failures this catches, both of which were real on 2026-09-10:

1. **Import-time file I/O.** `analysis/roc_plots.py` opened three data files and
   ran a normalisation loop at module scope, so `import roc_plots` failed
   outright on any machine that did not already have them -- i.e. every fresh
   clone, and every test process.

2. **Flat sibling imports.** Modules in `src/analysis/` imported each other as
   top-level names (`from gcv_curves import ...`), which only resolves when that
   directory happens to be `sys.path[0]` -- true when running
   `python src/analysis/x.py`, false from anywhere else. 22 such imports across
   16 files meant most of the package was unimportable from `tests/`.

A module that needs data should read it inside a function.
"""
import importlib
import pkgutil
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"

# Scripts whose module name is not a valid identifier (`00_make_af3_json_input`)
# are importable only via importlib, which is what the inference tests do.
_SKIP_PREFIXES = ("00_", "01_", "02_")


def _module_names():
    for pkg in sorted(p.name for p in SRC.iterdir()
                      if p.is_dir() and (p / "__init__.py").exists()):
        for mod in pkgutil.walk_packages([str(SRC / pkg)], prefix=f"{pkg}."):
            leaf = mod.name.rsplit(".", 1)[-1]
            if leaf.startswith(_SKIP_PREFIXES):
                continue
            yield mod.name
    for standalone in ("paths", "ids", "model", "contact_graphs"):
        yield standalone


@pytest.mark.parametrize("name", sorted(set(_module_names())))
def test_module_imports(name):
    importlib.import_module(name)
