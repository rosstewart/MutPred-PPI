#!/usr/bin/env python
"""Contact graphs and structures for the training/evaluation datasets, by sequence.

    from utils.structures import open_store, Structures

    store = open_store()
    G  = store.load_dense(interactor=iseq, partner=pseq)       # None on a miss
    ei = store.load_edge_index(interactor=iseq, partner=pseq)  # what the GAT wants

Everything here is keyed on the CONTENT of the two chain sequences. Accessions
change between mappings while sequences do not, so an ID-keyed lookup discards
valid structures and can accept one whose name matches but whose sequence does
not.

What this module used to be, and why none of it survives:

* `GraphResolver` read a `.mat` per pair through `scipy.io.loadmat`, pulled the
  chain boundary out of the bare integer `NRR`, rebuilt the concatenated
  sequence from the char array `L`, and then rotated the matrix when a `swapped`
  flag said the partner was stored first. `ContactGraphStore` returns the graph
  already oriented to the requested interactor, so the boundary, the flag and
  the rotation all disappear -- along with `orient()`, which existed only to
  apply the flag.
* `PDBResolver` mapped a `.mat` path to a `.pdb` by normalising FILENAME stems
  (`_pdb_key`) across two incompatible naming conventions. That was an exact
  sequence lookup followed by a name-based second hop, which reintroduced the
  filename coupling the first hop had just removed. `Structures` below resolves
  the structure straight from the canonical manifest, on the same unordered pair
  of sequence hashes as the store.

`interactor`/`partner` are keyword-only throughout, for the reason given in
`contact_graphs`: the two are not interchangeable, orientation cannot be
re-derived from the data, and a positional transposition must be unwriteable.
"""
from __future__ import annotations

import csv
import gzip
import shutil
from pathlib import Path

from contact_graphs import ContactGraphStore, StructureResolver, sha  # noqa: E402
from paths import DATASETS_DIR, TRAINING_EVAL_DIR  # noqa: E402

# The train/eval namespace. Variant-DB consumers have their own store; they must
# not share this one, because a pair can be present in one and absent from the
# other and the coverage difference is a reportable number.
STORE_PATH = TRAINING_EVAL_DIR / "contact_graphs.h5"

STRUCTURE_MANIFEST = DATASETS_DIR / "af3_structures_canonical" / "manifest.csv"


def open_store(path: Path | str = STORE_PATH) -> ContactGraphStore:
    """Open the training/evaluation contact-graph store (read-only)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found -- run repro_test/migrate_graphs_to_store.py")
    return ContactGraphStore(p)


class Structures:
    """Structure file and the interactor's chain id, for the structure-based methods.

    SAAMBE-3D and DDMut-PPI score a PDB rather than a contact matrix, and they
    must be told which chain carries the mutation. Both facts come from the one
    canonical manifest: the file is found by the unordered pair of sequence
    hashes, and the chain id is whichever of `chain_a_id`/`chain_b_id` belongs to
    the interactor's own hash. Neither is inferred from a filename, and neither
    assumes the interactor is chain A -- that assumption is exactly what put the
    mutation on the wrong chain for the pairs whose stored order contradicts
    their name.
    """

    def __init__(self, manifest: Path | str | None = None,
                 pdb_cache: Path | str | None = None):
        self._res = StructureResolver(manifest)
        self.manifest_path = self._res.manifest_path
        self.n_indexed = self._res.n_indexed
        # sha -> chain id, per unordered pair. Homodimers share one hash and one
        # entry, which is correct: either chain carries the same sequence.
        self._chains: dict[frozenset, dict[str, str]] = {}
        with open(self.manifest_path) as fh:
            for r in csv.DictReader(fh):
                key = frozenset((r["seq_a_sha"], r["seq_b_sha"]))
                self._chains.setdefault(key, {})[r["seq_a_sha"]] = r["chain_a_id"]
                self._chains[key].setdefault(r["seq_b_sha"], r["chain_b_id"])
        self._pdb_cache = Path(pdb_cache) if pdb_cache else None

    def find(self, *, interactor: str, partner: str,
             suffix: str | None = None) -> tuple[Path | None, str]:
        """(structure path, interactor chain id), or (None, "") on a miss."""
        path = self._res.find(interactor=interactor, partner=partner, suffix=suffix)
        if path is None:
            return None, ""
        key = frozenset((sha(interactor), sha(partner)))
        chain = self._chains.get(key, {}).get(sha(interactor), "A")
        return path, chain

    def find_pdb(self, *, interactor: str, partner: str,
                 workdir: Path | str | None = None) -> tuple[Path | None, str]:
        """As `find`, but guarantees an uncompressed PDB on disk.

        The canonical tree stores gzipped mmCIF, which neither SAAMBE-3D (PDB
        parser only) nor the DDMut-PPI upload endpoint accepts. Converting here
        keeps the single content-addressed tree as the one source of structures
        instead of maintaining a parallel directory of PDBs whose names would
        have to be matched -- which is how `_pdb_key` came about.

        Conversions are cached by content address, so a pair is converted once
        per run. Returns (None, "") when the pair has no structure or when the
        complex cannot be expressed in PDB format (too many atoms/chains).
        """
        path, chain = self.find(interactor=interactor, partner=partner)
        if path is None:
            return None, ""
        if path.name.endswith(".pdb"):
            return path, chain

        out_dir = Path(workdir or self._pdb_cache or
                       (self.manifest_path.parent / "_pdb_cache"))
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = path.name.split(".")[0]
        out = out_dir / f"{stem}.pdb"
        if out.exists():
            return out, chain

        try:
            if path.name.endswith(".cif.gz") or path.name.endswith(".cif"):
                import gemmi
                if path.name.endswith(".gz"):
                    tmp = out_dir / f"{stem}.cif"
                    with gzip.open(path, "rb") as fi, open(tmp, "wb") as fo:
                        shutil.copyfileobj(fi, fo)
                    src = tmp
                else:
                    src = path
                st = gemmi.read_structure(str(src))
                st.setup_entities()
                st.write_pdb(str(out))
                if src != path:
                    src.unlink()
            elif path.name.endswith(".pdb.gz"):
                with gzip.open(path, "rb") as fi, open(out, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
            else:
                return None, ""
        except Exception as exc:                      # noqa: BLE001
            print(f"  cannot convert {path.name} to PDB: {exc}", flush=True)
            if out.exists():
                out.unlink()
            return None, ""
        return out, chain
