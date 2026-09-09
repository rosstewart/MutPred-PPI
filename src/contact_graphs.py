#!/usr/bin/env python
"""Content-addressed contact-graph store: one HDF5 container, sparse edges.

Replaces the previous representation -- one `.mat` per pair holding a dense-ish
`G`, a char array `L`, and `NRR`, a bare integer marking where chain A ends.

What was wrong with that:

* **`NRR` states a boundary, not an identity.** Nothing in the file says which
  protein each chain is, so identity had to live out-of-band in
  `af3_index.csv.gz`. A `.mat` on its own could not be validated, and the file
  and the index could drift.
* **Orientation was implicit.** Whether the interactor came first depended on a
  `swapped` flag resolved at build time and re-derived at read time.
* **`L` round-tripped badly.** A list of one-character strings through MATLAB
  needs `"".join(str(x) for x in np.asarray(m["L"]).ravel())` to read back.
* **Consumers densified a 0.5%-dense matrix.** `.mat` stored it sparse,
  `mutpred_ppi_data` called `.toarray()` and copied per row, and `train_fold` then
  called `dense_to_sparse` to get back to where it started -- 157 GB of dense
  float64 for graphs that occupy 0.23 GB as edge lists.

This store fixes all four. The key IS the content: a group is named by the
sorted pair of chain-sequence SHAs, so lookup needs no filename parsing and a
mislabelled file cannot be found under the wrong key. Both sequences are stored
as attributes, so every record is self-validating and `af3_index` becomes a
derived cache rather than a second source of truth.

Layout:

    /graphs/{sha_min}_{sha_max}
        edge_index  (2, nnz) int32   upper triangle only, i < j, symmetric implied
      attrs:
        seq_a, seq_b        str      chain sequences, in STORED order
        len_a, len_b        int
        a_is_first          bool     True when seq_a hashes to sha_min
        source              str      file this record was derived from
        contact_threshold   float    angstroms, any-heavy-atom
    root attrs:
        format_version, created, n_graphs

**Self-loops are part of the canonical graph.** Every model in this repo trains
on graphs with the diagonal set, but the diagonal used to be added by each
consumer separately -- and `run_stability_inference` forgot, so the stability GAT
was trained with self-attention and run without it. Rather than leave that to
the caller, `load_dense`/`load_edge_index` always include self-loops, and the
operation is idempotent, so a record written either way reads back the same.
Pass `self_loops=False` only to inspect the raw contact geometry.

The diagonal is not written to disk: it is exactly the N pairs (i, i) for every
graph, so storing it would be redundant. `canonical_self_loops` in the root
attrs records that readers add it.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
DEFAULT_THRESHOLD = 4.5


def sha(seq: str) -> str:
    """16-hex-char content address of a chain sequence (matches graphs.sha)."""
    return hashlib.sha256(str(seq).encode()).hexdigest()[:16]


def pair_key(seq_a: str, seq_b: str = "") -> str:
    """Order-independent key for a chain pair, or a `mono_` key for one chain.

    Monomers exist because the stability pretraining set is single-chain. They
    get their own prefix rather than being faked as a pair with an empty second
    chain, and the prefix cannot collide with a pair key (two 16-hex halves).
    """
    if not seq_b:
        return f"mono_{sha(seq_a)}"
    ha, hb = sha(seq_a), sha(seq_b)
    return f"{min(ha, hb)}_{max(ha, hb)}"


def edges_from_dense(G: np.ndarray) -> np.ndarray:
    """(2, nnz) int32 upper-triangle edge list from a symmetric adjacency."""
    iu, ju = np.nonzero(np.triu(np.asarray(G) != 0, k=1))
    return np.vstack([iu, ju]).astype(np.int32)


def dense_from_edges(edge_index: np.ndarray, n: int,
                     dtype=np.float64) -> np.ndarray:
    """Symmetric dense adjacency, for parity checks against the old format."""
    G = np.zeros((n, n), dtype=dtype)
    i, j = edge_index
    G[i, j] = 1
    G[j, i] = 1
    return G


def symmetric_edge_index(edge_index: np.ndarray, n: int,
                         self_loops: bool = True) -> np.ndarray:
    """Both directions (and optionally self-loops), as torch_geometric wants.

    The old path reached the same place via
    `dense_to_sparse(torch.tensor(fill_diagonal(G.toarray(), 1)))`, which is why
    `self_loops` defaults True -- it reproduces that graph exactly.
    """
    i, j = edge_index
    off = i != j                      # drop any stored diagonal before mirroring
    src = np.concatenate([i[off], j[off]])
    dst = np.concatenate([j[off], i[off]])
    if self_loops:
        # Idempotent: the diagonal is added here and only here, so a record that
        # already carried one and a record that did not read back identical.
        d = np.arange(n, dtype=src.dtype)
        src = np.concatenate([src, d])
        dst = np.concatenate([dst, d])
    order = np.lexsort((dst, src))
    return np.vstack([src[order], dst[order]]).astype(np.int64)


def check_embedding_lengths(*, interactor_seq: str, partner_seq: str,
                            interactor_emb=None, partner_emb=None,
                            label: str = "") -> str | None:
    """Return a reason string when an embedding disagrees with its sequence.

    The graph store already guarantees the graph itself is self-consistent. This
    is the other half: a per-residue embedding must have exactly one row per
    residue of the sequence it claims to describe, or it is stale -- generated
    from a different version of that protein.

    Three call sites hand-rolled this check and then silently `continue`d on
    failure (`mutpred_ppi_data.py`, `run_variant_db_inference.py:323`,
    `compress_to_subgraphs.py:196`), which is how 107 mis-ordered clinvar graphs
    left the subgraph store with no trace. Callers get a reason back so they can
    count it; they must not discard it.
    """
    if interactor_emb is not None and len(interactor_emb) != len(interactor_seq):
        return (f"{label}interactor embedding has {len(interactor_emb)} rows for "
                f"a {len(interactor_seq)}-residue sequence")
    if partner_emb is not None and len(partner_emb) != len(partner_seq):
        return (f"{label}partner embedding has {len(partner_emb)} rows for "
                f"a {len(partner_seq)}-residue sequence")
    return None


class StructureResolver:
    """Locate the structure file for a pair, by sequence.

    SAAMBE-3D and DDMut-PPI score a PDB/mmCIF rather than a contact matrix, so
    they need a path on disk. They used to get one by taking the `.mat` a
    sequence lookup returned and transforming its FILENAME into a structure name
    -- an exact first hop followed by a name-based second hop, which reintroduced
    exactly the filename coupling the rest of the pipeline had removed.

    This resolves the structure directly from the canonical directory's manifest,
    keyed on the same unordered pair of sequence hashes as everything else, so
    there is no `.mat` hop and no filename parsing. `interactor`/`partner` are
    keyword-only for the same reason they are on the store: the two are not
    interchangeable and a positional swap must not be expressible.
    """

    def __init__(self, manifest: Path | str | None = None):
        import csv
        from paths import DATASETS_DIR
        p = Path(manifest or
                 DATASETS_DIR / "af3_structures_canonical" / "manifest.csv").resolve()
        self.manifest_path = p
        self._idx: dict[frozenset, list[Path]] = {}
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found -- run repro_test/canonicalize_structures.py")
        root = p.parent
        with open(p) as fh:
            for r in csv.DictReader(fh):
                key = frozenset((r["seq_a_sha"], r["seq_b_sha"]))
                self._idx.setdefault(key, []).append(root / r["filename"])
        self.n_indexed = sum(len(v) for v in self._idx.values())

    def find(self, *, interactor: str, partner: str,
             suffix: str | None = None) -> Path | None:
        """Structure path for this pair, or None.

        `suffix` restricts the format -- SAAMBE-3D parses PDB and cannot read
        mmCIF, so it passes ".pdb". Returning a file the caller cannot read would
        be worse than returning nothing.
        """
        hits = self._idx.get(frozenset((sha(interactor), sha(partner))), [])
        for h in hits:
            if suffix is None or h.name.endswith(suffix) or \
               h.name.endswith(suffix + ".gz"):
                return h
        return None


class ContactGraphStore:
    """Read/write access to the HDF5 contact-graph container."""

    def __init__(self, path: Path | str, mode: str = "r"):
        import h5py
        self.path = Path(path)
        self._h5 = h5py.File(self.path, mode)
        if mode in ("w", "w-", "x"):
            self._h5.attrs["format_version"] = FORMAT_VERSION
            self._h5.attrs["created"] = datetime.now(timezone.utc).isoformat()
            # Readers add the diagonal; it is not stored (see module docstring).
            self._h5.attrs["canonical_self_loops"] = True
            self._h5.create_group("graphs")
        self._g = self._h5["graphs"]

    # ── writing ──────────────────────────────────────────────────────────────

    def put(self, seq_a: str, seq_b: str = "", edge_index: np.ndarray = None, *,
            source: str = "", threshold: float = DEFAULT_THRESHOLD,
            overwrite: bool = False) -> str:
        """Store one graph. `seq_b=""` stores a single-chain (monomer) graph."""
        key = pair_key(seq_a, seq_b)
        if key in self._g:
            if not overwrite:
                return key
            del self._g[key]
        n = len(seq_a) + len(seq_b)
        ei = np.asarray(edge_index, dtype=np.int32)
        if ei.size and (ei.max() >= n or ei.min() < 0):
            raise ValueError(f"{key}: edge index out of range for {n} residues")
        d = self._g.create_dataset(key, data=ei, compression="gzip",
                                   compression_opts=4, shuffle=True)
        d.attrs["seq_a"], d.attrs["seq_b"] = seq_a, seq_b
        d.attrs["len_a"], d.attrs["len_b"] = len(seq_a), len(seq_b)
        d.attrs["a_is_first"] = sha(seq_a) <= sha(seq_b)
        d.attrs["source"] = str(source)
        d.attrs["contact_threshold"] = float(threshold)
        return key

    # ── reading ──────────────────────────────────────────────────────────────

    def __contains__(self, key: str) -> bool:
        return key in self._g

    def __len__(self) -> int:
        return len(self._g)

    def keys(self):
        return self._g.keys()

    def get(self, seq_a: str, seq_b: str):
        """(edge_index, n_residues, interactor_is_first) oriented for seq_a.

        Returns None on a miss. `edge_index` is renumbered when the stored order
        is the reverse of the requested one, so the caller always sees seq_a
        occupying residues [0, len(seq_a)).
        """
        key = pair_key(seq_a, seq_b)
        d = self._g.get(key)
        if d is None:
            return None
        s_a, s_b = d.attrs["seq_a"], d.attrs["seq_b"]
        la, lb = int(d.attrs["len_a"]), int(d.attrs["len_b"])
        n = la + lb
        ei = np.asarray(d, dtype=np.int32)

        # Stale-data guard, applied here so no consumer can forget it. Every
        # caller used to hand-roll a dimension check and then SILENTLY `continue`
        # on failure -- which is how 107 reversed clinvar graphs disappeared from
        # the subgraph store without leaving a count. A record whose declared
        # lengths disagree with its own sequences, or whose edges address nodes
        # that do not exist, is corrupt; that is an error, not a skip.
        if len(s_a) != la or len(s_b) != lb:
            raise ValueError(
                f"{key}: declared lengths ({la}, {lb}) disagree with the stored "
                f"sequences ({len(s_a)}, {len(s_b)}) -- stale or corrupt record")
        if ei.size and (int(ei.max()) >= n or int(ei.min()) < 0):
            raise ValueError(
                f"{key}: edge index range [{int(ei.min())}, {int(ei.max())}] "
                f"is outside the {n} residues this graph declares")

        if s_a == seq_a and s_b == seq_b:
            return ei, n, True
        if not seq_b:
            raise ValueError(f"{key}: monomer record does not match its key")
        if s_a == seq_b and s_b == seq_a:
            # Stored partner-first: rotate indices so the requested seq_a leads.
            shift = int(d.attrs["len_a"])
            ei = np.where(ei < shift, ei + (n - shift), ei - shift).astype(np.int32)
            return ei, n, True
        raise ValueError(
            f"{key}: stored sequences do not match the requested pair -- the "
            f"content address collided or the record is corrupt")

    # ── the four consumer shapes ─────────────────────────────────────────────
    # Every live call site in the repo wanted one of these and hand-rolled it,
    # which is how six of them ended up assuming chain A is the interactor and
    # how stability training and inference ended up disagreeing about
    # self-loops. Going through these is what keeps that from recurring.
    #
    # These are KEYWORD-ONLY and named by ROLE, deliberately. Orientation cannot
    # be re-derived from the data: measured on the combined set, 3.8% of rows
    # would survive both the length check and the wild-type residue assertion if
    # the two chains were transposed, and for the 831 true homodimers the two
    # chains are the same sequence, so no assertion can ever tell them apart.
    # The only defence is to make the mistake unwriteable -- there is no argument
    # order to get wrong, and `interactor=` must be spelled out at every call.
    # `interactor` is the mutated protein and always occupies residues
    # [0, len(interactor)) in everything returned here.

    def load_dense(self, *, interactor: str, partner: str = "",
                   self_loops: bool = True, dtype=np.float64):
        """Dense adjacency, interactor first. None on a miss."""
        hit = self.get(interactor, partner)
        if hit is None:
            return None
        ei, n, _ = hit
        G = dense_from_edges(ei, n, dtype=dtype)
        if self_loops:
            np.fill_diagonal(G, 1)
        return G

    def load_edge_index(self, *, interactor: str, partner: str = "",
                        self_loops: bool = True):
        """(2, E) int64 edge_index, both directions, interactor first.

        This is what the GAT actually consumes. The previous path reached it by
        densifying and calling `dense_to_sparse`, which is why the graphs cost
        157 GB of RAM instead of 0.23 GB.
        """
        hit = self.get(interactor, partner)
        if hit is None:
            return None
        ei, n, _ = hit
        return symmetric_edge_index(ei, n, self_loops=self_loops)

    def load_seqs(self, *, interactor: str, partner: str = ""):
        """(interactor, partner, len_interactor), or None on a miss."""
        if self.get(interactor, partner) is None:
            return None
        return interactor, partner, len(interactor)

    def neighbourhood(self, seq_a: str, seq_b: str, node_idx: int, hops: int = 2):
        """(sorted node ids, local edge_index) for the n-hop ball around a node.

        Serves the variant-DB subgraph compression, which previously walked a
        CSR matrix it built itself.
        """
        hit = self.get(seq_a, seq_b)
        if hit is None:
            return None
        ei, n, _ = hit
        adj = [[] for _ in range(n)]
        for i, j in zip(ei[0], ei[1]):
            adj[i].append(j)
            adj[j].append(i)
        frontier, seen = {int(node_idx)}, {int(node_idx)}
        for _ in range(hops):
            nxt = {v for u in frontier for v in adj[u]} - seen
            seen |= nxt
            frontier = nxt
            if not frontier:
                break
        nodes = np.array(sorted(seen), dtype=np.int64)
        remap = {int(g): k for k, g in enumerate(nodes)}
        keep = [(remap[int(i)], remap[int(j)])
                for i, j in zip(ei[0], ei[1])
                if int(i) in remap and int(j) in remap]
        local = (np.array(keep, dtype=np.int64).T if keep
                 else np.zeros((2, 0), dtype=np.int64))
        return nodes, local

    def load_aliases(self, path: Path | str | None = None) -> dict[str, str]:
        """`complex_id` -> `pair_key`, from the sidecar written at migration.

        Purely for OUTPUT STRINGS. Variant-DB consumers print the old `.mat` stem
        as the key of their prediction TSVs, and those TSVs must keep the same
        keys after the migration. An alias must never be used to decide which
        chain is the interactor -- that is what the sequences are for, and the
        stems are exactly what was found to lie about chain order.
        """
        import csv
        p = Path(path or self.path.with_name("aliases.csv"))
        if not p.exists():
            raise FileNotFoundError(f"{p} not found -- re-run the migration")
        with open(p) as fh:
            return {r["complex_id"]: r["pair_key"] for r in csv.DictReader(fh)}

    def meta(self) -> list[dict]:
        """One row per graph: the derived index, without touching edge data."""
        out = []
        for k, d in self._g.items():
            out.append({
                "key": k,
                "seq_a_sha": sha(d.attrs["seq_a"]),
                "seq_b_sha": sha(d.attrs["seq_b"]),
                "len_a": int(d.attrs["len_a"]),
                "len_b": int(d.attrs["len_b"]),
                "a_is_first": bool(d.attrs["a_is_first"]),
                "nnz": int(d.shape[1]) if d.ndim == 2 else 0,
                "source": str(d.attrs.get("source", "")),
            })
        return out

    def close(self):
        self._h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
