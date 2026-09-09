#!/usr/bin/env python
"""Graph + embedding tensors for MutPred-PPI, built from the canonical tables.

MutPred-PPI is the one method whose inputs a rows table cannot supply directly:
it needs an AF3 contact graph and ProtT5 embeddings per row. This module builds
those from the canonical representation, replacing the previous loader entirely.

What it replaces, and why:

    _load_graphs()        enumerated .mat files and parsed accessions out of
                          FILENAMES, with a per-dataset `id_parts_fn`
    _build_emb_dict()     keyed embeddings by those parsed names
    _find_pair_graph()    tried four directory/orientation combinations
    _remap_vt_ids()       reconciled gene-symbol and UniProt namespaces

All of that existed because identity lived in filenames. Here a pair is
identified by its two sequences (see graphs.GraphResolver), so there is nothing
to parse and nothing to remap.

Node indices stay 0-based -- `mutation_idx` addresses a row of the graph, not a
residue in a mutation string. Cache keys and mutation strings are 1-based.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

from contact_graphs import check_embedding_lengths  # noqa: E402
from evaluation.graphs import GraphResolver  # noqa: E402
from paths import DATASETS_DIR  # noqa: E402

TABLES = DATASETS_DIR / "mapped090826"


def prott5_path(dataset: str) -> Path:
    return TABLES / f"{dataset}_prott5.pkl"


def load_prott5(dataset: str) -> dict:
    p = prott5_path(dataset)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found -- run src/data_processing/precompute_prott5_datasets.py "
            f"--dataset {dataset}")
    with open(p, "rb") as f:
        return pickle.load(f)


class MissingInputs(RuntimeError):
    """Raised when a row has no structure or no embedding.

    Comparability is the reason this is fatal: every method is scored on the
    same rows, so a row MutPred-PPI cannot build must be fixed upstream (fold the
    structure, extend the cache) rather than quietly dropped from its denominator
    alone.
    """


def build_tensors(rows, dataset: str, *, resolver: GraphResolver | None = None,
                  t5: dict | None = None, verbose: bool = True,
                  use_wt_emb: bool = False, require_complete: bool = True):
    """Per-row graph and embedding tensors, aligned to `rows` by position.

    Returns a dict of lists plus `usable`, a boolean mask. A row is usable only
    if its structure matches by sequence AND all three embeddings are present.
    With the default `require_complete`, any unusable row raises `MissingInputs`
    listing the counted reasons, so a coverage gap surfaces immediately instead
    of shrinking this method's denominator relative to the others.

    `use_wt_emb` is the `wt-emb` ablation: the interactor's node features become
    its wild-type embedding instead of the mutant one. `mut_diff` is deliberately
    unaffected -- it stays mutant minus wild type, as in the original loader.
    """
    resolver = resolver or GraphResolver()
    t5 = t5 if t5 is not None else load_prott5(dataset)

    n = len(rows)
    out = {k: [None] * n for k in
           ("edge_mat", "node_emb", "mut_diff", "mutation_idx",
            "pos_labels", "neg_labels", "seq_lengths", "clusters")}
    usable = np.zeros(n, dtype=bool)
    reasons = {"no_graph": 0, "no_wt_emb": 0, "no_mut_emb": 0,
               "no_partner_emb": 0, "length_mismatch": 0, "ok": 0}
    examples: list[str] = []   # a few concrete stale-data cases, for the message

    graph_cache: dict[tuple[str, str], np.ndarray] = {}

    for i, r in enumerate(rows.itertuples(index=False)):
        a_seq, b_seq = r.interactor_sequence, r.partner_sequence
        key = (a_seq, b_seq)
        if key not in graph_cache:
            G, found = resolver.load(a_seq, b_seq)
            graph_cache[key] = G if found else None
        G = graph_cache[key]
        if G is None:
            reasons["no_graph"] += 1
            continue

        wt = t5.get(r.interactor)
        mt = t5.get(f"{r.interactor}_{r.mutation}")
        pb = t5.get(r.partner)
        if wt is None:
            reasons["no_wt_emb"] += 1; continue
        if mt is None:
            reasons["no_mut_emb"] += 1; continue
        if pb is None:
            reasons["no_partner_emb"] += 1; continue

        # Graph/sequence agreement is guaranteed by the store itself; what still
        # has to be checked here is that the EMBEDDINGS describe these sequences
        # and are not stale. Both wild-type and mutant embed the interactor, so
        # both are checked against it.
        bad = (check_embedding_lengths(interactor_seq=a_seq, partner_seq=b_seq,
                                       interactor_emb=wt, partner_emb=pb,
                                       label="WT ")
               or check_embedding_lengths(interactor_seq=a_seq, partner_seq=b_seq,
                                          interactor_emb=mt, label="MUT "))
        if bad or G.shape[0] != len(a_seq) + len(b_seq):
            reasons["length_mismatch"] += 1
            if len(examples) < 5:
                examples.append(f"{r.interactor}/{r.partner}: "
                                f"{bad or 'graph size disagrees with sequences'}")
            continue

        mut_idx = int(r.position) - 1          # 0-based NODE index into the graph

        # The mutation must land on the interactor's own residue. This is the
        # check that makes reading the partner as the interactor impossible: a
        # swapped pair indexes into the wrong chain and almost never finds the
        # expected wild-type residue there. It is a raise, not a skip -- a
        # position that disagrees with its sequence is corrupt input, and the
        # graph node index is derived from it.
        if mut_idx < 0 or mut_idx >= len(a_seq):
            raise MissingInputs(
                f"{r.interactor}/{r.partner} {r.mutation}: position "
                f"{r.position} is outside the interactor ({len(a_seq)} aa)")
        if a_seq[mut_idx] != r.wt_aa:
            raise MissingInputs(
                f"{r.interactor}/{r.partner} {r.mutation}: interactor has "
                f"{a_seq[mut_idx]!r} at position {r.position}, the table says "
                f"{r.wt_aa!r} -- the sequence and the mutation disagree, or the "
                f"chains are swapped")

        em = G.copy()
        np.fill_diagonal(em, 1)

        out["edge_mat"][i] = em
        out["node_emb"][i] = np.concatenate([wt if use_wt_emb else mt, pb])
        out["mut_diff"][i] = mt[mut_idx] - wt[mut_idx]
        out["mutation_idx"][i] = mut_idx
        out["seq_lengths"][i] = [len(a_seq), len(b_seq)]
        out["clusters"][i] = str(r.cluster)
        # Label encoding train_fold expects: a non-empty pos_labels marks a
        # disrupted interaction. Both hold NODE indices, which are 0-based.
        if bool(r.perturbed):
            out["pos_labels"][i], out["neg_labels"][i] = [mut_idx], []
        else:
            out["pos_labels"][i], out["neg_labels"][i] = [], [mut_idx]
        usable[i] = True
        reasons["ok"] += 1

    if verbose:
        print(f"  MutPred-PPI tensors: {reasons['ok']}/{n} usable", flush=True)
        for k, v in reasons.items():
            if k != "ok" and v:
                print(f"    excluded, {k}: {v}", flush=True)
        for e in examples:
            print(f"      e.g. {e}", flush=True)

    if require_complete and not usable.all():
        missing = "; ".join(f"{k}: {v}" for k, v in reasons.items()
                            if k != "ok" and v)
        detail = ("\n  e.g. " + "; ".join(examples)) if examples else ""
        raise MissingInputs(
            f"{dataset}: {int((~usable).sum())} of {n} rows lack inputs "
            f"({missing}). Every method must be scored on the same rows, so this "
            f"is an error rather than a NaN. Fold the missing structures and "
            f"extend the ProtT5 cache, or pass require_complete=False to score "
            f"the covered subset.{detail}"
        )

    out["usable"] = usable
    out["reasons"] = reasons
    return out
