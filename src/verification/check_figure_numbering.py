#!/usr/bin/env python3
"""Do the docs and the notebook use the manuscript's figure numbers?

The repository invented its own labels (`S-biclass`, `S-protclass`) and drifted
on six others, so a reader holding the paper could not map a figure to the
command that produced it. The manuscript is the source of truth: this derives
the numbering from the LaTeX and checks every reference against it.

Numbering comes from order of appearance, counting `figure` and `table`
environments separately and skipping commented-out blocks.

    conda run -n ppi python src/verification/check_figure_numbering.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from paths import REPO_ROOT

#: label -> short description, for references that name a figure in prose.
DESCRIPTIONS = {
    "tab:datasets": "training datasets",
    "fig:sahni_fragoza_cv": "Sahni+Fragoza GCV",
    "fig:blind_test_roc": "VarChAMP blind test",
    "fig:enrichment": "variant-repository enrichment",
    "tab:variant_dbs_table": "repository statistics",
    "fig:roc_sahni_fragoza_biclass": "bi-class GCV",
    "fig:sahni_cv": "Sahni-only GCV",
    "fig:varchamp_training_set_comparison": "training-set comparison",
    "fig:ablation": "ablation",
    "fig:combined_robustness_by_class": "robustness by class",
    "fig:sahni_varchamp_cava_cv": "pooled GCV",
    "fig:enrichment_k3": "partner-controlled enrichment",
    "fig:threshold_sensitivity": "threshold sensitivity",
    "fig:protein_class_enrichment": "protein-class enrichment",
    "fig:stability_mechanism": "stability vs interaction",
}


def _strip_comments(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("%"))


def manuscript_numbering(repo: Path) -> dict[str, dict]:
    """label -> {number, images} derived from the two LaTeX sources."""
    mains = sorted(repo.glob("main_*.tex"))
    supps = sorted(repo.glob("supplement_*.tex"))
    if not mains or not supps:
        return {}
    out = {}
    for tag, path in (("MAIN", mains[-1]), ("SUPP", supps[-1])):
        body = _strip_comments(path.read_text())
        n_fig = n_tab = 0
        for m in re.finditer(r"\\begin\{(figure|table)\*?\}(.*?)\\end\{\1\*?\}",
                             body, re.S):
            kind, inner = m.group(1), m.group(2)
            lab = re.search(r"\\label\{([^}]+)\}", inner)
            if not lab:
                continue
            if kind == "figure":
                n_fig += 1
                num = f"Fig {n_fig}" if tag == "MAIN" else f"S{n_fig}"
            else:
                n_tab += 1
                num = f"Table {n_tab}" if tag == "MAIN" else f"Table S{n_tab}"
            out[lab.group(1)] = {
                "number": num,
                # Tables come in via \input{}, figures via \includegraphics{}.
                "images": (re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", inner)
                           + re.findall(r"\\input\{([^}]+)\}", inner)),
            }
    return out


def expected_file_to_number(numbering: dict[str, dict]) -> dict[str, str]:
    """basename -> manuscript number, for every figure the manuscript includes."""
    out = {}
    for label, info in numbering.items():
        for img in info["images"]:
            out[Path(img).name] = info["number"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    repo = REPO_ROOT
    numbering = manuscript_numbering(repo)
    if not numbering:
        print("no manuscript sources found; nothing to check against")
        return 0

    by_file = expected_file_to_number(numbering)
    if not args.quiet:
        print(f"{len(numbering)} labelled floats, {len(by_file)} figure files\n")
        for label, info in numbering.items():
            desc = DESCRIPTIONS.get(label, "")
            imgs = ", ".join(Path(i).name for i in info["images"]) or "(no image)"
            print(f"  {info['number']:<10} {desc:<32} {imgs}")

    problems = []

    # Hand-drawn schematics and renderings are not pipeline outputs, so they are
    # not expected in figures/.
    HAND_DRAWN = {"MutPred-PPI_pipeline.png", "MutPred-PPI_architecture.png",
                  "CDC42_WASP_Y64.png", "brca1_bard1_C61.png"}
    for f in by_file:
        if f in HAND_DRAWN:
            continue
        if not (repo / "figures" / f).exists():
            problems.append(f"figures/{f} is referenced by the manuscript but absent")

    # The notebook manifest must map each label to the manuscript's own file.
    nb = repo / "notebooks" / "reproduce_all_figures.py"
    if nb.exists():
        # Only the manuscript manifest is cross-referenced. _PRESENTATION_MANIFEST
        # holds figures for talks, which carry no manuscript number by design.
        block = re.search(r"\n_MANIFEST = \[(.*?)\n\]", nb.read_text(), re.S)
        if block:
            for label, path in re.findall(r'\("([^"]+)",\s*(.+?)\),\n', block.group(1)):
                fname = re.findall(r'"([^"]+\.(?:png|tex))"', path)
                if not fname:
                    continue
                want = by_file.get(fname[-1])
                if want is None:
                    problems.append(f"notebook labels {fname[-1]} as {label!r}, but the "
                                    f"manuscript does not include that file")
                elif want != label:
                    problems.append(f"notebook labels {fname[-1]} as {label!r}; the "
                                    f"manuscript numbers it {want!r}")

    # Retired labels must not survive anywhere.
    retired = ("S-biclass", "S-protclass", "S-stability", "S-robustness",
               "S-abl", "S-new", "S-cosmic-stat", "S-stability-density",
               "S-stability-scatter")
    # Only tracked docs; the gitignored internal notes are not user-facing.
    import subprocess
    tracked = subprocess.run(["git", "ls-files", "*.md"], cwd=repo,
                             capture_output=True, text=True).stdout.split()
    searched = [repo / t for t in tracked] + [
        repo / "notebooks" / "reproduce_all_figures.py"]
    for path in searched:
        if not path.exists():
            continue
        text = path.read_text()
        for label in retired:
            if label in text:
                problems.append(f"{path.relative_to(repo)} still uses the retired "
                                f"label {label!r}")

    # Docs cite the line QUICK sits on. That number drifts whenever the notebook
    # is edited, and a wrong one sends a reader to the wrong line.
    nb_path = repo / "notebooks" / "reproduce_all_figures.py"
    if nb_path.exists():
        quick_line = next((i for i, l in enumerate(nb_path.read_text().splitlines(), 1)
                           if l.startswith("QUICK = ")), None)
        if quick_line:
            for path in searched:
                if path.suffix != ".md" or not path.exists():
                    continue
                for cited in re.findall(r"\(line (\d+)\)", path.read_text()):
                    if int(cited) != quick_line:
                        problems.append(
                            f"{path.relative_to(repo)} cites line {cited} for QUICK; "
                            f"it is on line {quick_line}")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  {p}")
        return 1
    if not args.quiet:
        print("\nevery manuscript figure is present and no retired label survives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
