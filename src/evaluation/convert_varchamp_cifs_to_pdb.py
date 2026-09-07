#!/usr/bin/env python
"""Convert AF3 CIF outputs for VarChAMP 2026 to PDB format.

Reads CIF files from 2026/af3_out/models/ and writes PDB files to
2026/af3_out/pdbs/ (creating the directory if needed).

Uses gemmi (available in the ppi conda env).

Usage:
    conda run -n ppi python convert_varchamp_cifs_to_pdb.py
"""

import sys
from pathlib import Path

import gemmi

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import REVISIONS_DIR  # noqa: E402


_CIF_DIR = REVISIONS_DIR / "af3_out" / "models"
_PDB_DIR = REVISIONS_DIR / "af3_out" / "pdbs"
def main():
    _PDB_DIR.mkdir(parents=True, exist_ok=True)
    cif_files = sorted(_CIF_DIR.glob("*.cif"))
    print(f"Found {len(cif_files)} CIF files in {_CIF_DIR}", flush=True)

    converted, skipped, errors = 0, 0, 0
    for i, cif in enumerate(cif_files):
        pdb = _PDB_DIR / (cif.stem + ".pdb")
        if pdb.exists():
            skipped += 1
            continue
        try:
            st = gemmi.read_structure(str(cif))
            st.write_pdb(str(pdb))
            converted += 1
        except Exception as exc:
            errors += 1
            print(f"  ERROR {cif.name}: {exc}", flush=True)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(cif_files)}: converted={converted} skipped={skipped} errors={errors}",
                  flush=True)

    print(f"\nDone: {converted} converted, {skipped} skipped, {errors} errors", flush=True)
    print(f"PDB files in: {_PDB_DIR}", flush=True)


if __name__ == "__main__":
    main()
