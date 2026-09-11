"""Shared internals of the three-step inference pipeline.

Named `utils` until 2026-09-10, which **shadowed the repo-level `src/utils/`**.
Because a script puts its own directory first on `sys.path`, running
`python src/inference/00_make_af3_json_input.py` -- the invocation the README
documents -- made `from utils.sequences import ...` resolve to *this* package
and fail with `ModuleNotFoundError`. Steps 1 and 4 of the documented Quick Start
were both dead on any machine, regardless of data. Renaming removes the
collision rather than papering over it with `sys.path` ordering.
"""
