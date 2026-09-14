"""Standalone consistency checks over the built data artifacts.

These are diagnostics, not pipeline steps: none of them writes a figure, table
or deposited artifact. They exist so that a rebuilt structure set, graph store
or cross-validation run can be shown to satisfy the invariants the analysis
code assumes. See docs/REPRODUCING_ANALYSES.md, "Verification utilities".
"""
