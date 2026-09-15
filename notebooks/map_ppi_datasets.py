# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python (ppi)
#     language: python
#     name: ppi
# ---

# %% [markdown]
# # Map PPI perturbation datasets to UniProt isoforms
#
# Builds one **master** record of every labeled PPI-perturbation observation across six
# sources, mapped to UniProt (isoform-aware), with every mutation validated against its
# sequence. Then emits dataset-specific training files.
#
# | Tag | Source | Tier |
# |---|---|---|
# | `VarChAMP_Maxim_2026` | 1% VarChAMP 2026 | 2026 |
# | `VarChAMP_Luke_pooled_2026` | pooled Y2H 2026 | 2026 |
# | `VarChAMP_Flo_vc1p_2025` | VarChAMP 1% legacy | legacy |
# | `VarChAMP_Flo_cava1p_2025` | CAVA pillar legacy | legacy |
# | `Sahni_2015` | raw Sahni WT+MT Y2H scores | published |
# | `Fragoza_2019` | raw Fragoza cosmic/exac/hgmd | published |
#
# ### Two conventions that matter
#
# **1. "interactor" means opposite things in the sources vs. our schema.**
# Source files (`df_luke`, Sahni, Fragoza) use `interactor_symbol` /
# `uniprot_ac_interactor` / `Interactor_Gene_ID` for the **partner** — the protein that is
# *not* mutated. Our output schema uses `interactor` for the **mutated** protein.
# The old notebook collided on this: it assigned `df["interactor_symbol"] = df["symbol"]`
# and *then* read that column for `partner_symbol`, destroying 17,412/17,966 partner
# symbols. Every loader here reads source columns into explicitly-named locals
# (`src_partner_*`) before building anything, and never writes into a column it later reads.
#
# **2. Canonical isoforms are identified by sequence, never by suffix number.**
# `ACC-1` is usually canonical but not always. We strip a suffix **iff** that isoform's
# sequence is byte-identical to the canonical sequence, so a genuinely different isoform
# is never collapsed.
#
# ### Master vs. dataset files
# The **master is not deduplicated and keeps conflicting labels** — one row per source
# observation, fully traceable. Dedup and conflict removal apply only to the
# dataset-specific files.

# %%
import gzip
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# --- Paths -------------------------------------------------------------------
# Every path resolves through `src/paths.py`, the repo's single point of path
# resolution -- no absolute paths here. Migrated into the repo 2026-09-10 from
# ~/ppi_lossgain/2026/map_ppi_datasets_090826.py, which had `IL` and `RAW_SF`
# hardcoded and wrote to an external MAPPING_090826/ tree that a user cloning
# this repo would never see. That was `REPRODUCIBILITY.md` gap 8: the mapping
# CSVs were inputs with no in-repo producer.
#
# Requires `pip install -e .` (see README) so that `paths` is importable.
sys.path.insert(0, str(Path.cwd().parent / "src"))   # notebook-friendly fallback
from utils import mutations  # noqa: E402
from paths import (SOURCE_DATA_DIR, SOURCE_DATA_RESTRICTED_DIR,  # noqa: E402
                   MAPPING_DIR)

# Published sources (Sahni 2015, Fragoza 2019) -- see docs/DATA_PREPARATION.md for
# the citations and where to download them.
RAW_SF = SOURCE_DATA_DIR
# Unpublished IGVF/VarChAMP data. Not redistributable; absent from Zenodo.
SRC_2026 = SOURCE_DATA_RESTRICTED_DIR

OUT_DIR = MAPPING_DIR

# Output layout:
#   master_data/        the complete un-deduplicated record (one row per observation)
#   datasets/           the primary combined training sets
#     varchamp_other/     narrower VarChAMP slices
#     single_source/      one published source each
#   intermediate_files/ audit trails + QC (why rows were dropped, remapped, collapsed)
#   cache/              UniProt responses, so re-runs are offline
MASTER_DIR = OUT_DIR / "master_data"
DATASETS_DIR = OUT_DIR / "datasets"
VARCHAMP_OTHER_DIR = DATASETS_DIR / "varchamp_other"
SINGLE_SOURCE_DIR = DATASETS_DIR / "single_source"
INTERMEDIATE_DIR = OUT_DIR / "intermediate_files"
CACHE_DIR = OUT_DIR / "cache"

for _d in (MASTER_DIR, DATASETS_DIR, VARCHAMP_OTHER_DIR, SINGLE_SOURCE_DIR,
           INTERMEDIATE_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

PATHS = {
    "maxim": SRC_2026 / "ppi_level_scores_perturbation_status_for_Ross_260212.csv",
    "luke": SRC_2026 / "pooled-Y2H_2026-07-24.tsv",
    "flo_vc1p": SRC_2026 / "seq_confirmed_VarChAMP1percentEdgotypingScores_SeqConfirmed_ONLY.csv",
    "flo_cava": SRC_2026 / "seq_confirmed_VarChampPillarEdgotypingScoresWithAlleleInfo_ToShare.csv",
    "sahni": RAW_SF / "sahni_wt_and_mt_y2h_scores.csv",
    "fragoza_cosmic": RAW_SF / "fragoza_cosmic.csv",
    "fragoza_exac": RAW_SF / "fragoza_exac.csv",
    "fragoza_hgmd": RAW_SF / "fragoza_hgmd.csv",
}
_RESTRICTED = {"maxim", "luke", "flo_vc1p", "flo_cava"}

# Exactly which supplementary table each published file is, so a missing one is
# actionable at the point of failure rather than after a hunt through the docs.
_PUBLISHED_SOURCE = {
    "sahni": ("Table S3A of Sahni et al., Cell 2015;161(3):647-660, "
              "doi:10.1016/j.cell.2015.04.013"),
    "fragoza_exac": ("Supplementary Data 2 (ExAC variants) of Fragoza et al., "
                     "Nat Commun 10, 4141 (2019), doi:10.1038/s41467-019-11959-3"),
    "fragoza_cosmic": ("Supplementary Data 3 (COSMIC somatic mutations) of "
                       "Fragoza et al., Nat Commun 10, 4141 (2019), "
                       "doi:10.1038/s41467-019-11959-3"),
    "fragoza_hgmd": ("Supplementary Data 4 (HGMD disease-associated mutations) "
                     "of Fragoza et al., Nat Commun 10, 4141 (2019), "
                     "doi:10.1038/s41467-019-11959-3"),
}

for name, p in PATHS.items():
    if not p.exists():
        if name in _RESTRICTED:
            tier = ("datasets/source_data_restricted/ (unpublished IGVF/VarChAMP "
                    "data, not redistributable -- see docs/DATA_PREPARATION.md)")
        else:
            tier = (f"datasets/source_data/ -- download it from "
                    f"{_PUBLISHED_SOURCE[name]}")
        raise FileNotFoundError(f"missing input '{name}': {p}\n  Expected in {tier}")

# --- Dataset tags ------------------------------------------------------------
# Every dataset tag and dataset filename carries this suffix so this mapping run
# is never confused with files produced under earlier conventions.  It is the
# SAME constant the rest of the codebase reads, so re-mapping is a one-line
# change here and nothing downstream has to be told about it.
from utils.legacy_guard import DATASET_SUFFIX as SUFFIX

TAG_MAXIM = f"VarChAMP_Maxim_2026{SUFFIX}"
TAG_LUKE = f"VarChAMP_Luke_pooled_2026{SUFFIX}"
TAG_VC1P = f"VarChAMP_Flo_vc1p_2025{SUFFIX}"
TAG_CAVA = f"VarChAMP_Flo_cava1p_2025{SUFFIX}"
TAG_SAHNI = f"Sahni_2015{SUFFIX}"
TAG_FRAGOZA = f"Fragoza_2019{SUFFIX}"

TIER = {
    TAG_MAXIM: "2026",
    TAG_LUKE: "2026",
    TAG_VC1P: "legacy_2025",
    TAG_CAVA: "legacy_2025",
    TAG_SAHNI: "published",
    TAG_FRAGOZA: "published",
}

# Conflict policy.
#
# Supersession is a VarChAMP-INTERNAL generational rule, not a cross-team authority
# ranking: a 2025 VarChAMP measurement is superseded only when the SAME team's 2026
# assay covers that same variant-partner pair. It carries no weight against Sahni or
# Fragoza, who are a different group entirely -- a 2025-VarChAMP vs published
# disagreement is a genuine cross-team conflict and drops every row for the key, exactly
# like Luke vs Maxim or Sahni vs Fragoza.
VC_2026 = {TAG_LUKE, TAG_MAXIM}
VC_2025 = {TAG_VC1P, TAG_CAVA}

print(f"OUT_DIR = {OUT_DIR}")

# %% [markdown]
# ## 1. Shared utilities

# %%
AA3_TO_AA1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
MISSENSE_RE = re.compile(r"^[A-Z]\d+[A-Z]$")
_THREE_RE = re.compile(r"^([A-Za-z]{3})(\d+)([A-Za-z]{3})$")


def three_to_one(aa_change):
    """'Pro263Leu' -> 'P263L'. None for non-missense or synonymous."""
    m = _THREE_RE.match(str(aa_change).strip())
    if not m:
        return None
    wt3, pos, mt3 = m.groups()
    wt3, mt3 = wt3.capitalize(), mt3.capitalize()
    if wt3 not in AA3_TO_AA1 or mt3 not in AA3_TO_AA1 or wt3 == mt3:
        return None
    return f"{AA3_TO_AA1[wt3]}{pos}{AA3_TO_AA1[mt3]}"


def clean_missense(mut):
    """Normalize an already-1-letter mutation; None if not simple missense/synonymous."""
    s = str(mut).strip().upper()
    if not MISSENSE_RE.match(s) or s[0] == s[-1]:
        return None
    return s


def parse_mutation(mut):
    """'E80K' -> ('E', 80, 'K'), 1-based position.

    Delegates to `utils.mutations.parse`, the repo's single mutation parser,
    which validates the string with a regex instead of assuming
    `mut[0] / int(mut[1:-1]) / mut[-1]` -- an inline form that cannot reject
    junk and silently produced a garbage triple for anything malformed.
    Verified 2026-09-10 to be identical on all 53,239 mutations across the five
    canonical datasets, so this changes nothing about the mapping's output.
    """
    return mutations.parse(mut)


def validate_mutation(seq, mut):
    """True iff seq[pos-1] == wt (1-based mutation position)."""
    if not isinstance(seq, str) or not isinstance(mut, str) or not MISSENSE_RE.match(mut):
        return False
    wt, pos, _ = parse_mutation(mut)
    return 0 < pos <= len(seq) and seq[pos - 1] == wt


def base_acc(acc):
    """'P12345-2' -> 'P12345'. Passes through None/NaN."""
    if acc is None or (isinstance(acc, float) and pd.isna(acc)):
        return None
    return str(acc).split("-")[0].strip()


def iso_suffix(acc):
    """Isoform number, or 0 for a bare (canonical) accession."""
    s = str(acc)
    return int(s.split("-")[1]) if "-" in s else 0


def norm_id(x):
    """Normalize an optional identifier to a clean string or None."""
    if x is None or (isinstance(x, float) and pd.isna(x)) or str(x).strip() in ("", "nan", "NULL", "-"):
        return None
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s


# Raw schema every loader emits, before UniProt resolution.
# Continuous measurements are kept in SOURCE-SPECIFIC columns rather than one shared
# pair. When a variant-partner pair is measured by both 2026 assays (280 such keys) the
# duplicate rows collapse into one, and a single `log2fc` column would silently keep
# whichever source sorted first and discard the other. Separate columns preserve both:
# groupby "first" skips NaN, so each column picks up its own assay's value.
#   *_1p     -> VarChAMP Maxim 2026 (1% assay)
#   *_pooled -> VarChAMP Luke 2026 (pooled Y2H)
RAW_COLS = [
    "dataset", "dataset_tier", "fragoza_source", "source_row_id",
    "int_acc", "int_refseq", "int_entrez", "int_symbol", "int_orf",
    "prt_acc", "prt_refseq", "prt_entrez", "prt_symbol", "prt_orf",
    "mutation", "perturbed",
    "log2fc_1p", "perturbation_LLR_1p",
    "log2fc_pooled", "perturbation_LLR_pooled",
]


def finalize_raw(df, tag):
    """Add missing raw columns, normalize identifier dtypes, order columns."""
    df = df.copy()
    df["dataset"] = tag
    df["dataset_tier"] = TIER[tag]
    for c in RAW_COLS:
        if c not in df.columns:
            df[c] = None
    for c in ["int_acc", "int_refseq", "int_entrez", "int_symbol", "int_orf",
              "prt_acc", "prt_refseq", "prt_entrez", "prt_symbol", "prt_orf"]:
        df[c] = df[c].map(norm_id)
    # Backstop for every loader: never let a missing label become a real one.
    # `.astype(bool)` is dangerous on its own -- bool(float('nan')) is True -- so a NaN
    # reaching here would silently become "perturbed". Fail loudly instead.
    if df["perturbed"].isna().any():
        n = int(df["perturbed"].isna().sum())
        raise ValueError(f"{tag}: {n} rows have a NaN 'perturbed' label; "
                         "the loader must drop or resolve these before finalize_raw()")
    df["perturbed"] = df["perturbed"].astype(bool)
    return df[RAW_COLS].reset_index(drop=True)


LOAD_STATS = []


def log_stage(tag, stage, n):
    LOAD_STATS.append({"dataset": tag, "stage": stage, "rows": n})
    print(f"  {tag:28s} {stage:34s} {n:>7,}")


# %% [markdown]
# ## 2. Cached UniProt client
#
# Replaces the old manual UniProt-web-tool round trip. Every response is cached under
# `datasets/source_mapping/cache/`, so re-runs are offline and fast.
#
# * `idmap` — the ID-mapping API. Resolves obsolete/secondary accessions that a plain
#   `accession:` search misses, and handles `GeneID` / `RefSeq_Protein` / `Gene_Name`.
# * `fetch_isoforms` — `uniprotkb/stream` with `includeIsoform=true`, chunked at 40
#   accessions (150 returns HTTP 400: URI too long).

# %%
SESSION = requests.Session()
UNIPROT = "https://rest.uniprot.org"
HUMAN = "9606"


def _cache_path(kind, key):
    h = hashlib.sha1(key.encode()).hexdigest()[:24]
    return CACHE_DIR / f"{kind}_{h}.json.gz"


def _cache_load(kind, key):
    p = _cache_path(kind, key)
    if p.exists():
        with gzip.open(p, "rt") as f:
            return json.load(f)
    return None


def _cache_save(kind, key, value):
    with gzip.open(_cache_path(kind, key), "wt") as f:
        json.dump(value, f)


def idmap(ids, from_db, to_db="UniProtKB",
          fields="accession,reviewed,organism_id,gene_primary,sequence", chunk=5000):
    """UniProt ID-mapping. Returns a DataFrame with a 'From' column. Cached.

    `to_db="UniProtKB-Swiss-Prot"` restricts to reviewed entries *server-side*, which is
    both correct and dramatically cheaper: for Gene_Name it returns 147 rows instead of
    5,660 (all organisms) and runs ~8x faster. Note the results endpoint's `query`
    parameter returns HTTP 500 and must not be used -- filter client-side instead.
    """
    ids = sorted({str(i) for i in ids if norm_id(i) is not None})
    if not ids:
        return pd.DataFrame(columns=["From"])
    frames = []
    for i in range(0, len(ids), chunk):
        part = ids[i:i + chunk]
        key = f"{from_db}|{to_db}|{fields}|{','.join(part)}"
        cached = _cache_load("idmap", key)
        if cached is None:
            r = SESSION.post(f"{UNIPROT}/idmapping/run",
                             data={"from": from_db, "to": to_db, "ids": ",".join(part)},
                             timeout=120)
            r.raise_for_status()
            job = r.json()["jobId"]
            for _ in range(600):
                st = SESSION.get(f"{UNIPROT}/idmapping/status/{job}", timeout=120).json()
                if st.get("jobStatus") in ("RUNNING", "NEW"):
                    time.sleep(2)
                    continue
                break
            texts, url = [], f"{UNIPROT}/idmapping/uniprotkb/results/{job}"
            params = {"format": "tsv", "fields": fields, "size": 500}
            while url:
                rr = SESSION.get(url, params=params, timeout=300)
                rr.raise_for_status()
                texts.append(rr.text if not texts else "\n".join(rr.text.splitlines()[1:]))
                url = rr.links.get("next", {}).get("url")
                params = None
            cached = "\n".join(texts)
            _cache_save("idmap", key, cached)
        if cached.strip():
            frames.append(pd.read_csv(io.StringIO(cached), sep="\t", dtype=str))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["From"])


def search_human_symbols(symbols, chunk=25):
    """{gene symbol -> reviewed human accession}, restricted to human SERVER-SIDE.

    The ID-mapping `Gene_Name` route is unusable here: it returns hits for every
    organism (BRCA1 -> rat/mouse/human) and, at ~1,700 ids, silently returns an empty
    result set. The search API takes `organism_id:9606 AND reviewed:true` directly in
    the query, so only human Swiss-Prot entries are ever downloaded.
    """
    syms = sorted({s for s in (norm_id(x) for x in symbols) if s})
    out, todo = {}, []
    for s in syms:
        c = _cache_load("sym", s)
        out[s] = c["acc"] if c and c.get("acc") else None
        if c is None:
            todo.append(s)
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        q = ("(" + " OR ".join(f"gene_exact:{s}" for s in part) + ")"
             f" AND (organism_id:{HUMAN}) AND (reviewed:true)")
        texts, url = [], f"{UNIPROT}/uniprotkb/search"
        params = {"query": q, "fields": "accession,gene_primary,gene_names",
                  "format": "tsv", "size": 500}
        while url:
            r = SESSION.get(url, params=params, timeout=180)
            r.raise_for_status()
            texts.append(r.text if not texts else "\n".join(r.text.splitlines()[1:]))
            url = r.links.get("next", {}).get("url")
            params = None
        blob = "\n".join(texts)
        found = {}
        if blob.strip():
            d = pd.read_csv(io.StringIO(blob), sep="\t", dtype=str)
            lower = {s.lower(): s for s in part}
            for rec in d.to_dict("records"):
                primary = str(rec.get("Gene Names (primary)") or "").strip()
                names = str(rec.get("Gene Names") or "").split()
                for nm in [primary] + names:
                    key = lower.get(str(nm).lower())
                    # a primary-name hit always wins over a synonym hit
                    if key and (key not in found or primary.lower() == key.lower()):
                        found[key] = rec["Entry"]
                        if primary:
                            ACC_GENE[rec["Entry"]] = primary
        for s in part:
            _cache_save("sym", s, {"acc": found.get(s)})
            out[s] = found.get(s)
        if (i // chunk) % 20 == 0:
            print(f"    symbol search {i + len(part):,}/{len(todo):,}")
    return {k: v for k, v in out.items() if v}


def fetch_isoforms(accessions, chunk=40):
    """{isoform_id: sequence} for the canonical + all isoforms of each accession."""
    accs = sorted({a for a in (base_acc(x) for x in accessions) if a})
    out = {}
    todo = []
    for a in accs:
        c = _cache_load("iso", a)
        if c is None:
            todo.append(a)
        else:
            out.update(c)
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        q = " OR ".join(f"accession:{a}" for a in part)
        r = SESSION.get(f"{UNIPROT}/uniprotkb/stream",
                        params={"query": q, "format": "fasta", "includeIsoform": "true"},
                        timeout=300)
        r.raise_for_status()
        got, cur = {}, None
        for line in r.text.splitlines():
            if line.startswith(">"):
                cur = line.split("|")[1]
                got[cur] = []
            elif cur:
                got[cur].append(line.strip())
        got = {k: "".join(v) for k, v in got.items()}
        for a in part:  # cache per accession, including empty results
            per = {k: v for k, v in got.items() if base_acc(k) == a}
            _cache_save("iso", a, per)
            out.update(per)
        if (i // chunk) % 25 == 0:
            print(f"    isoform fetch {i + len(part):,}/{len(todo):,}")
    return out


print("UniProt client ready.")

# %% [markdown]
# ## 3. Per-source loaders
#
# Each returns the same `RAW_COLS` schema. Source-specific logic only.

# %% [markdown]
# ### 3a. VarChAMP Maxim (1% 2026)
# No ORF ids in this file, so `int_orf` / `prt_orf` stay blank (per decision: no
# symbol-based ORF guessing).

# %%
def load_maxim():
    tag = TAG_MAXIM
    df = pd.read_csv(PATHS["maxim"])
    log_stage(tag, "raw rows", len(df))

    df = df[df["perturbation_status"].isin(["perturbed", "unperturbed"])].copy()
    log_stage(tag, "labeled (drop uncertain/NaN)", len(df))

    df["mutation"] = df["aa_change"].map(three_to_one)
    df = df.dropna(subset=["mutation"])
    log_stage(tag, "missense mutation parsed", len(df))

    out = pd.DataFrame({
        "source_row_id": [f"{tag}:{i}" for i in df.index],
        "int_symbol": df["symbol"].values,
        "prt_symbol": df["ad_symbol"].values,
        "mutation": df["mutation"].values,
        "perturbed": (df["perturbation_status"] == "perturbed").values,
        "log2fc_1p": df["log2fc"].values,
        "perturbation_LLR_1p": df["perturbation_LLR"].values,
    })
    return finalize_raw(out, tag)


# %% [markdown]
# ### 3b. VarChAMP Luke (pooled 2026)
#
# **The partner-symbol fix.** `df_luke.interactor_symbol` *is* the partner symbol
# (`LITAF` → `TAX1BP1`). We read it into `src_partner_symbol` before constructing
# anything, so it cannot be clobbered.
#
# `aa_change_uniprot_isoform` is expressed relative to `uniprot_isoform_ac`, so that
# accession is carried as the isoform seed for §5.

# %%
OBSOLETE_ACC_FIX = {"V9GYU4": "Q8N319-2"}  # was left commented out in the old notebook


def load_luke():
    tag = TAG_LUKE
    df = pd.read_csv(PATHS["luke"], sep="\t", low_memory=False)
    log_stage(tag, "raw rows", len(df))

    df = df[df["perturbation_status"].isin(["perturbed", "unperturbed"])].copy()
    log_stage(tag, "labeled (drop uncertain/NaN)", len(df))

    # Read source columns FIRST -- 'interactor_*' here means the PARTNER.
    src_partner_symbol = df["interactor_symbol"].copy()
    src_partner_acc = df["uniprot_ac_interactor"].replace(OBSOLETE_ACC_FIX).copy()
    src_partner_orf = df["interactor_id"].copy()
    src_mut_symbol = df["symbol"].copy()
    src_mut_acc = df["uniprot_isoform_ac"].replace(OBSOLETE_ACC_FIX).copy()
    src_mut_orf = df["orf_id_wt"].copy()

    mutation = df["aa_change_uniprot_isoform"].map(three_to_one)
    keep = mutation.notna()
    log_stage(tag, "missense mutation parsed", int(keep.sum()))

    out = pd.DataFrame({
        "source_row_id": [f"{tag}:{i}" for i in df.index[keep]],
        "int_acc": src_mut_acc[keep].values,
        "int_symbol": src_mut_symbol[keep].values,
        "int_orf": src_mut_orf[keep].values,
        "prt_acc": src_partner_acc[keep].values,
        "prt_symbol": src_partner_symbol[keep].values,
        "prt_orf": src_partner_orf[keep].values,
        "mutation": mutation[keep].values,
        "perturbed": (df.loc[keep, "perturbation_status"] == "perturbed").values,
        "log2fc_pooled": df.loc[keep, "log2FC_combined"].values,
        "perturbation_LLR_pooled": df.loc[keep, "perturbation_LLR"].values,
    })
    return finalize_raw(out, tag)


# %% [markdown]
# ### 3c. VarChAMP Flo legacy (vc1p + CAVA pillar)
#
# `df_flo_vc1p` contains gene symbols truncated at a trailing digit (`CARD`, `KRT`,
# `NAA`, `PLA2G`, `PUF`, `TMEM`, `MAGEB`, `ZMYND`). These are repaired from the
# hORFeome ORF id **before** mapping, using an ORF→symbol table built from df_luke and
# the CAVA file. Two ORFs are not covered there and are fixed explicitly.

# %%
MANUAL_SYMBOL_FIX = {  # hORFeome ORF id -> gene symbol (validated by ORF id)
    9866: "ZMYND10",
    52806: "MAGEB2",
}


def build_orf_symbol_map():
    luke = pd.read_csv(PATHS["luke"], sep="\t", low_memory=False)
    cava = pd.read_csv(PATHS["flo_cava"])
    frames = [
        luke[["orf_id_wt", "symbol"]].rename(columns={"orf_id_wt": "orf", "symbol": "sym"}),
        luke[["interactor_id", "interactor_symbol"]].rename(
            columns={"interactor_id": "orf", "interactor_symbol": "sym"}),
        cava[["db_orf_id", "symbol"]].rename(columns={"db_orf_id": "orf", "symbol": "sym"}),
        cava[["ad_orf_id", "ad_symbol"]].rename(columns={"ad_orf_id": "orf", "ad_symbol": "sym"}),
    ]
    tbl = pd.concat(frames, ignore_index=True).dropna()
    tbl["orf"] = tbl["orf"].astype(int)
    # Keep only ORFs with a single unambiguous symbol.
    counts = tbl.groupby("orf")["sym"].nunique()
    tbl = tbl[tbl["orf"].isin(counts[counts == 1].index)]
    m = dict(zip(tbl["orf"], tbl["sym"]))
    m.update(MANUAL_SYMBOL_FIX)
    return m


ORF_TO_SYMBOL = build_orf_symbol_map()
print(f"ORF -> symbol table: {len(ORF_TO_SYMBOL):,} unambiguous ORF ids")


def repair_symbols(symbols, orfs, tag, side):
    """Replace truncated symbols with the ORF-derived symbol where they disagree."""
    fixed, changes = [], []
    for s, o in zip(symbols, orfs):
        s_norm = norm_id(s)
        try:
            o_int = int(float(o))
        except (TypeError, ValueError):
            fixed.append(s_norm)
            continue
        cand = ORF_TO_SYMBOL.get(o_int)
        # Only override when the source symbol is a strict prefix of the ORF symbol,
        # i.e. a truncation such as CARD -> CARD10. Never silently rename a real symbol.
        if cand and s_norm and cand != s_norm and cand.startswith(s_norm):
            changes.append({"dataset": tag, "side": side, "orf_id": o_int,
                            "symbol_in_file": s_norm, "symbol_repaired": cand})
            fixed.append(cand)
        else:
            fixed.append(s_norm)
    return fixed, changes


SYMBOL_REPAIRS = []


def load_flo(which):
    tag = TAG_VC1P if which == "flo_vc1p" else TAG_CAVA
    df = pd.read_csv(PATHS[which])
    log_stage(tag, "raw rows", len(df))

    df = df[df["edgotype_wt_2"].isin(["perturbed", "not perturbed"])].copy()
    log_stage(tag, "labeled (drop inconclusive/NaN)", len(df))

    df["mutation"] = df["aa_change"].map(three_to_one)
    df = df.dropna(subset=["mutation"])
    log_stage(tag, "missense mutation parsed", len(df))

    int_sym, ch1 = repair_symbols(df["symbol"], df["db_orf_id"], tag, "interactor")
    prt_sym, ch2 = repair_symbols(df["ad_symbol"], df["ad_orf_id"], tag, "partner")
    SYMBOL_REPAIRS.extend(ch1 + ch2)

    out = pd.DataFrame({
        "source_row_id": [f"{tag}:{i}" for i in df.index],
        "int_symbol": int_sym,
        "int_orf": df["db_orf_id"].values,
        "prt_symbol": prt_sym,
        "prt_orf": df["ad_orf_id"].values,
        "mutation": df["mutation"].values,
        "perturbed": (df["edgotype_wt_2"] == "perturbed").values,
    })
    return finalize_raw(out, tag)


# %% [markdown]
# ### 3d. Sahni 2015 (raw)
#
# Two corrections versus the old SWING-derived path:
#
# 1. **Label polarity.** In the raw file `Y2H_score == 1` means *interaction retained* —
#    every wild-type row scores 1. So `perturbed = (Y2H_score == 0)`. The old notebook
#    read SWING's flipped encoding and mapped `1 -> perturbed`.
# 2. **Wild-type filter.** Keep a mutant row only if its `(gene, partner)` wild-type row
#    actually interacts (`Y2H_score == 1`).

# %%
def load_sahni():
    tag = TAG_SAHNI
    df = pd.read_csv(PATHS["sahni"])
    log_stage(tag, "raw rows", len(df))

    wt = df[df["Category"] == "Wild-type"]
    mut = df[df["Category"] != "Wild-type"].copy()
    log_stage(tag, "mutant rows", len(mut))

    wt_interacts = set(
        zip(wt.loc[wt["Y2H_score"] == 1, "Entrez_Gene_ID"],
            wt.loc[wt["Y2H_score"] == 1, "Interactor_Gene_ID"])
    )
    pair = list(zip(mut["Entrez_Gene_ID"], mut["Interactor_Gene_ID"]))
    mut = mut[[p in wt_interacts for p in pair]].copy()
    log_stage(tag, "WT counterpart interacts", len(mut))

    # 'NP_001079:p.A129T' -> refseq NP_001079, mutation A129T
    # Guard the label BEFORE deriving it. `NaN == 0` is False, so an unguarded
    # `perturbed = (Y2H_score == 0)` would silently relabel a missing score as
    # "not perturbed" rather than dropping the row. Currently a no-op (the column is
    # int64 with values {0,1}), but the source file must not be trusted to stay that way.
    mut = mut[mut["Y2H_score"].isin([0, 1])].copy()
    log_stage(tag, "label present (0/1)", len(mut))

    parts = mut["Mutation_RefSeq_AA"].astype(str).str.split(":p.", regex=False, expand=True)
    mut["int_refseq"] = parts[0]
    mut["mutation"] = parts[1].map(clean_missense)
    mut = mut.dropna(subset=["mutation"])
    log_stage(tag, "missense mutation parsed", len(mut))

    out = pd.DataFrame({
        "source_row_id": [f"{tag}:{i}" for i in mut.index],
        "int_refseq": mut["int_refseq"].values,
        "int_entrez": mut["Entrez_Gene_ID"].values,
        "int_symbol": mut["Symbol"].values,
        "prt_entrez": mut["Interactor_Gene_ID"].values,
        "prt_symbol": mut["Interactor_symbol"].values,
        "mutation": mut["mutation"].values,
        "perturbed": (mut["Y2H_score"] == 0).values,   # 0 = interaction lost
    })
    return finalize_raw(out, tag)


# %% [markdown]
# ### 3e. Fragoza 2019 (raw, three files)
#
# The three files are concatenated with a `fragoza_source` column. `UniProt` packs the
# accession and mutation together (`"Q9UJC3-1,S433L"`); 457 rows already carry an isoform
# suffix, which is preserved as the isoform seed. The partner is an Entrez GeneID.
#
# These files contain **no wild-type rows**, so no WT filter applies — Fragoza only
# assayed pairs whose wild-type interaction was already established.

# %%
def load_fragoza():
    tag = TAG_FRAGOZA
    frames = []
    for src in ["cosmic", "exac", "hgmd"]:
        d = pd.read_csv(PATHS[f"fragoza_{src}"])
        d["fragoza_source"] = src
        d["source_row_id"] = [f"{tag}:{src}:{i}" for i in d.index]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    log_stage(tag, "raw rows (3 files)", len(df))

    # Same guard as Sahni: `NaN == 1` is False, which would silently mean "not perturbed".
    df = df[df["Disruption"].isin([0, 1])].copy()
    log_stage(tag, "label present (0/1)", len(df))

    up = df["UniProt"].astype(str).str.split(",", n=1, expand=True)
    df["int_acc"] = up[0].str.strip()
    df["mutation"] = up[1].str.strip().map(clean_missense)
    df = df.dropna(subset=["mutation"])
    log_stage(tag, "missense mutation parsed", len(df))

    out = pd.DataFrame({
        "source_row_id": df["source_row_id"].values,
        "fragoza_source": df["fragoza_source"].values,
        "int_acc": df["int_acc"].values,
        "int_entrez": df["Target Entrez GeneID"].values,
        "prt_entrez": df["Interactor Entrez GeneID"].values,
        "mutation": df["mutation"].values,
        "perturbed": (df["Disruption"] == 1).values,
    })
    return finalize_raw(out, tag)


# %%
print("Loading sources:")
raw = pd.concat(
    [load_maxim(), load_luke(), load_flo("flo_vc1p"), load_flo("flo_cava"),
     load_sahni(), load_fragoza()],
    ignore_index=True,
)
df_repairs = pd.DataFrame(SYMBOL_REPAIRS).drop_duplicates()
print(f"\nTotal raw mapped-pending rows: {len(raw):,}")
print(raw.dataset.value_counts().to_string())
print(f"\nTruncated-symbol repairs ({len(df_repairs)}):")
print(df_repairs.to_string(index=False) if len(df_repairs) else "  none")

# %% [markdown]
# ## 4. Resolve every identifier to UniProt
#
# Priority per protein: explicit accession → RefSeq protein → Entrez GeneID →
# hORFeome ORF id → gene symbol. Swiss-Prot (reviewed) is preferred at every step.

# %%
ACC_REVIEWED = {}   # accession -> True if Swiss-Prot
ACC_GENE = {}       # accession -> primary gene symbol (used to backfill missing symbols)


def _pick(df_map, human_only):
    """Collapse an idmap result to {From: accession}, preferring reviewed + human.

    `human_only=True` is a hard filter, used for the gene-level routes (symbol, Entrez,
    RefSeq) where every dataset here is human and an ortholog must never win. For an
    explicit accession supplied by a source file we only *prefer* human, since that
    accession is authoritative and dropping it would lose the row entirely.

    Ties are broken on **sequence length, longest first**, before falling back to
    accession order. A handful of GeneIDs cross-reference more than one reviewed human
    protein, and plain accession order picks among them arbitrarily -- wrongly in every
    case where the answer is checkable:
      51207 DUSP13  -> DUSP13A Q6B8I1 (188aa) beat DUSP13B Q9UII6 (198aa), but NCBI's
                      symbol for 51207 is DUSP13B, and the two share only 3.7% identity
      83871 RAB34   -> P0DI83 (198aa) is "isoform NARR", a readthrough product, beating
                      Q9BZG1 (259aa), the actual Rab-34 GTPase
      9465  AKAP7   -> the 104aa alpha form beat the 348aa gamma form, and disagreed with
                      what the symbol route independently gives for the same gene
    Longest-first gets all three right and keeps the Entrez route consistent with the
    symbol route. Where the candidates are sequence-identical (the PRR20 and CT45A
    families) length cannot decide and accession order breaks the tie harmlessly.
    """
    if df_map.empty or "From" not in df_map.columns:
        return {}
    d = df_map.copy()
    d["_rev"] = (d.get("Reviewed", "") == "reviewed").astype(int)
    d["_hum"] = (d["Organism (ID)"].astype(str) == HUMAN).astype(int) \
        if "Organism (ID)" in d.columns else 1
    d["_len"] = d["Sequence"].fillna("").str.len() if "Sequence" in d.columns else 0
    ACC_REVIEWED.update(dict(zip(d["Entry"], d["_rev"] == 1)))
    if "Gene Names (primary)" in d.columns:
        ACC_GENE.update({a: g for a, g in zip(d["Entry"], d["Gene Names (primary)"])
                         if isinstance(g, str) and g.strip()})
    if human_only:
        d = d[d["_hum"] == 1]
        if d.empty:
            return {}
    d = d.sort_values(["From", "_hum", "_rev", "_len", "Entry"],
                      ascending=[True, False, False, False, True])
    # keep="first" is essential: dict(zip(...)) would otherwise retain the WORST match.
    d = d.drop_duplicates(subset=["From"], keep="first")
    return dict(zip(d["From"], d["Entry"]))


# Collect identifiers needing resolution
accs = set(raw["int_acc"].dropna()) | set(raw["prt_acc"].dropna())
refseqs = set(raw["int_refseq"].dropna()) | set(raw["prt_refseq"].dropna())
entrez = set(raw["int_entrez"].dropna()) | set(raw["prt_entrez"].dropna())
symbols = set(raw["int_symbol"].dropna()) | set(raw["prt_symbol"].dropna())
print(f"to resolve -> accessions {len(accs):,} | refseq {len(refseqs):,} | "
      f"entrez {len(entrez):,} | symbols {len(symbols):,}")

# Entrez ids that UniProt's GeneID cross-reference resolves badly or not at all.
#
# Age of the ids is NOT the problem. All 1,966 Entrez ids used by Sahni (2015) and
# Fragoza (2019) were checked against NCBI esummary: 1,964 are still live and only two
# carry a CurrentID redirect (both fixed below). NCBI never recycles a GeneID, so a
# 2015/2019 id either still denotes the same gene or is explicitly redirected. Sahni's
# own recorded symbols confirm this independently -- 100% still denote the same gene,
# with 23 pure HGNC renames (ADCK3->COQ8A, GNB2L1->RACK1, QARS->QARS1, ...) where the
# old symbol survives as an NCBI alias. What actually breaks is UniProt's xref curation:
#
#   145946 SPATA8      protein-coding, but NO UniProtKB entry carries its GeneID xref;
#                      the symbol route reaches the reviewed entry cleanly
#   285733 -> 100507203 (SMLR1)   NCBI-replaced id, reviewed entry exists
#   440321 -> 100132565 (GOLGA8F) NCBI-replaced id, reviewed entry exists
#
# Ids whose xref returns several reviewed proteins are NOT a filter failure. Both
# filters are on and working: the Entrez route requests `to_db=UniProtKB-Swiss-Prot`
# (reviewed-only, server-side) and `_pick(human_only=True)` (hard client-side human
# filter). Every one of the 14 ambiguous candidates below is `reviewed` + `9606`.
# UniProt genuinely cross-references one NCBI GeneID to several distinct reviewed
# human entries, in two structurally different situations:
#
#   (a) sequence-IDENTICAL paralog families collapsed onto one GeneID --
#       122183 PRR20A-E (all 221 aa), 441521 CT45A5/6/7 (all 189 aa). The tie is
#       harmless: whichever entry wins, the sequence is the same, and sequence is
#       all anything downstream uses. Left to _pick's fallback ordering.
#
#   (b) genuinely DIFFERENT proteins sharing one GeneID -- the three below. Here a
#       wrong pick is not a near-miss: DUSP13A and DUSP13B share 3.7% identity.
#       These are PINNED EXPLICITLY rather than left to the length heuristic, so the
#       decision is recorded with its reason instead of being re-derived from a proxy
#       every run. The pins agree exactly with what longest-first selects, so this is
#       output-identical -- it makes the choice auditable, not different.
ENTREZ_AMBIGUOUS_PINS = {
    # 51207 DUSP13 is a genuinely dual-ORF locus encoding two unrelated phosphatases:
    # DUSP13A/TMDP (Q6B8I1, 188 aa, testis) and DUSP13B/MDSP (Q9UII6, 198 aa, muscle).
    # Accession order picked DUSP13A; NCBI's own gene record for 51207 leads with
    # DUSP13B, and the symbol route agrees. 3.7% identity between them, so this is the
    # single most consequential pin in the file.
    "51207": "Q9UII6",
    # 83871 RAB34: P0DI83 (198 aa) is protein NARR, the NAT6-RAB34 readthrough product,
    # not the GTPase. Q9BZG1 (259 aa) is Rab-34 itself. Both carry primary gene symbol
    # "RAB34", so symbol matching cannot separate them -- only the annotation can.
    "83871": "Q9BZG1",
    # 9465 AKAP7: alpha (O43687, 104 aa) and gamma (Q9P0M2, 348 aa) are curated as
    # separate entries under one symbol. Gamma is the full-length product and is what
    # the symbol route independently returns for AKAP7.
    "9465": "Q9P0M2",
}
#
# Six ids stay unresolved (390535 GOLGA8EP, 729862 LSP1P3, 155060, 401508 FAM74A4,
# 541471 MIR4435-2HG, 202459 OSTCP1): pseudogenes and lncRNA host genes with no protein
# product and no UniProt entry at all, not even TrEMBL. 13 Fragoza rows, correctly lost.
ENTREZ_MANUAL_FIX = {
    "145946": "Q6RVD6",   # SPATA8
    "285733": "H3BR10",   # SMLR1
    "440321": "P0DX52",   # GOLGA8F
}

SP = "UniProtKB-Swiss-Prot"   # server-side reviewed-only filter

# Explicit accessions: full UniProtKB, so a source-supplied TrEMBL id survives as itself
# (only 8 of Fragoza's 849 accessions; §4b handles those). Human preferred, not forced.
print("\nresolving accessions...")
map_acc = _pick(idmap({base_acc(a) for a in accs}, "UniProtKB_AC-ID"), human_only=False)

# Gene-level routes: reviewed server-side, human enforced client-side.
print("resolving refseq...")
map_refseq = _pick(idmap(refseqs, "RefSeq_Protein", to_db=SP), human_only=True)
print("resolving entrez...")
_entrez_raw = idmap(entrez, "GeneID", to_db=SP)
map_entrez = _pick(_entrez_raw, human_only=True)
map_entrez.update({k: v for k, v in ENTREZ_MANUAL_FIX.items() if k in entrez})
map_entrez.update({k: v for k, v in ENTREZ_AMBIGUOUS_PINS.items() if k in entrez})

# A GeneID that cross-references more than one reviewed human protein is ambiguous.
# Ambiguity itself is fine and expected -- see ENTREZ_AMBIGUOUS_PINS above for why it
# is real rather than a filter failure. What must NOT happen is a NEW ambiguous id
# being resolved silently by a length proxy on data nobody has looked at.
#
# So the rule is: an ambiguous GeneID whose candidates are SEQUENCE-IDENTICAL is
# harmless and passes (case (a) -- paralog families; the pick cannot matter because
# every candidate carries the same sequence). An ambiguous GeneID whose candidates
# have DIFFERENT sequences must be pinned explicitly, or this raises. That is the
# class where a wrong pick silently substitutes a different protein.
_amb = _entrez_raw[_entrez_raw.get("Organism (ID)", pd.Series(HUMAN, index=_entrez_raw.index))
                   .astype(str) == HUMAN] if len(_entrez_raw) else _entrez_raw
_ALL_PINS = {**ENTREZ_MANUAL_FIX, **ENTREZ_AMBIGUOUS_PINS}
if len(_amb):
    _n = _amb.groupby("From")["Entry"].nunique()
    _n = _n[_n > 1]
    _amb = (_amb[_amb["From"].isin(_n.index)]
            .assign(pinned=lambda d: d["From"].map(_ALL_PINS))
            .sort_values(["From", "Entry"]))
    _amb.to_csv(INTERMEDIATE_DIR / "ambiguous_entrez_ids.csv", index=False)

    _identical, _divergent = [], []
    for _e in sorted(set(_n.index)):
        _seqs = set(_amb.loc[_amb["From"] == _e, "Sequence"].fillna(""))
        (_identical if len(_seqs) == 1 else _divergent).append(_e)

    print(f"  GeneIDs mapping to >1 reviewed human protein: {len(_n)} "
          f"({len(_identical)} sequence-identical, {len(_divergent)} divergent)")
    for _e in _identical:
        _c = _amb.loc[_amb["From"] == _e, "Entry"].tolist()
        print(f"    {_e}: {_c} -> identical sequences, pick is immaterial "
              f"(taking {map_entrez.get(_e)})")
    for _e in _divergent:
        _c = _amb.loc[_amb["From"] == _e, "Entry"].tolist()
        _pin = _ALL_PINS.get(_e)
        print(f"    {_e}: {_c} -> {map_entrez.get(_e)}"
              f"{' (PINNED)' if _pin else ' (UNPINNED -- ERROR)'}")

    _unpinned_divergent = [e for e in _divergent if e not in _ALL_PINS]
    if _unpinned_divergent:
        def _describe(e):
            _sub = _amb[_amb["From"] == e]
            return f"    GeneID {e}: " + ", ".join(
                f"{_r['Entry']} ({_r['Gene Names (primary)']}, "
                f"{len(_r['Sequence'] or '')} aa)"
                for _, _r in _sub.iterrows())
        _detail = "\n".join(_describe(e) for e in _unpinned_divergent)
        raise ValueError(
            f"{len(_unpinned_divergent)} GeneID(s) cross-reference several reviewed "
            f"human proteins with DIFFERENT sequences and are not pinned:\n{_detail}\n"
            f"  A length heuristic must not decide this silently -- DUSP13A vs DUSP13B "
            f"share 3.7% identity, so the wrong pick substitutes an unrelated protein.\n"
            f"  Resolve each against its NCBI gene record and add it to "
            f"ENTREZ_AMBIGUOUS_PINS with the reason, then re-run.")
print("resolving symbols (human + reviewed, server-side)...")
map_symbol = search_human_symbols(symbols)

# hORFeome ORF -> accession, from df_luke (conflict-free: no ORF maps to >1 accession)
_luke = pd.read_csv(PATHS["luke"], sep="\t", low_memory=False)
map_orf = {}
for orf_col, acc_col in [("orf_id_wt", "uniprot_isoform_ac"), ("interactor_id", "uniprot_ac_interactor")]:
    sub = _luke[[orf_col, acc_col]].dropna().drop_duplicates()
    for o, a in zip(sub[orf_col], sub[acc_col]):
        map_orf.setdefault(norm_id(o), base_acc(OBSOLETE_ACC_FIX.get(a, a)))

print(f"\nresolved: acc {len(map_acc):,}/{len(accs):,} | refseq {len(map_refseq):,}/{len(refseqs):,} | "
      f"entrez {len(map_entrez):,}/{len(entrez):,} | symbol {len(map_symbol):,}/{len(symbols):,} | "
      f"orf {len(map_orf):,}")


# %%
def resolve_side(row, side):
    """Return (base_accession, isoform_seed, route) for one side of an interaction."""
    acc, refseq = row[f"{side}_acc"], row[f"{side}_refseq"]
    ent, sym, orf = row[f"{side}_entrez"], row[f"{side}_symbol"], row[f"{side}_orf"]
    seed = acc if (acc and "-" in str(acc)) else None
    if acc:
        b = map_acc.get(base_acc(acc))
        if b:
            return b, seed, "accession"
    if refseq and refseq in map_refseq:
        return map_refseq[refseq], seed, "refseq"
    if ent and ent in map_entrez:
        return map_entrez[ent], seed, "entrez"
    if orf and orf in map_orf:
        return map_orf[orf], seed, "orf"
    if sym and sym in map_symbol:
        return map_symbol[sym], seed, "symbol"
    return None, seed, "UNRESOLVED"


res = raw.copy()
for side, pfx in [("int", "interactor"), ("prt", "partner")]:
    triples = [resolve_side(r, side) for r in res.to_dict("records")]
    res[f"{pfx}_base"] = [t[0] for t in triples]
    res[f"{pfx}_seed"] = [t[1] for t in triples]
    res[f"{pfx}_route"] = [t[2] for t in triples]

print("interactor route:\n", res.interactor_route.value_counts().to_string())
print("\npartner route:\n", res.partner_route.value_counts().to_string())

unresolved = res[res.interactor_base.isna() | res.partner_base.isna()]
print(f"\nrows with an unresolved side: {len(unresolved):,} / {len(res):,} "
      f"({100 * len(unresolved) / len(res):.2f}%)")

# %%
# Audit the unresolved identifiers, then drop those rows (they cannot be mapped at all).
_u = []
for side, pfx in [("int", "interactor"), ("prt", "partner")]:
    m = res[f"{pfx}_base"].isna()
    if m.any():
        _u.append(pd.DataFrame({
            "dataset": res.loc[m, "dataset"], "side": pfx,
            "acc": res.loc[m, f"{side}_acc"], "refseq": res.loc[m, f"{side}_refseq"],
            "entrez": res.loc[m, f"{side}_entrez"], "symbol": res.loc[m, f"{side}_symbol"],
            "orf_id": res.loc[m, f"{side}_orf"], "source_row_id": res.loc[m, "source_row_id"],
        }))
df_unresolved = pd.concat(_u, ignore_index=True) if _u else pd.DataFrame()
df_unresolved.to_csv(INTERMEDIATE_DIR / "unresolved_identifiers.csv", index=False)
print(f"unresolved_identifiers.csv: {len(df_unresolved):,} rows")
if len(df_unresolved):
    print(df_unresolved.groupby(["dataset", "side"]).size().to_string())

res = res[res.interactor_base.notna() & res.partner_base.notna()].reset_index(drop=True)
print(f"\nrows carried forward: {len(res):,}")

# %% [markdown]
# ### 4b. Normalize TrEMBL to Swiss-Prot (sequence-identical only)
#
# Sahni/Fragoza carry TrEMBL accessions while VarChAMP is Swiss-Prot. Left alone, 530
# overlapping gene symbols share only 384 accessions, so cross-dataset duplicates and
# conflicts are invisible to `resolve()`.
#
# A TrEMBL accession is remapped to the Swiss-Prot entry for the same gene **only when
# its sequence is byte-identical to one of that entry's isoforms** — never on gene
# identity alone, which could silently change the sequence a mutation was validated
# against. Every remap is logged.

# %%
def _swissprot_candidate(row, side):
    """Swiss-Prot accession for this protein's gene, via the human+reviewed routes."""
    for key, mapping in ((f"{side}_entrez", map_entrez), (f"{side}_symbol", map_symbol)):
        v = row[key]
        if v and v in mapping:
            return mapping[v]
    return None


in_use = set(res["interactor_base"]) | set(res["partner_base"])
unreviewed = {a for a in in_use if not ACC_REVIEWED.get(a, False)}
print(f"accessions in use: {len(in_use):,} | unreviewed (TrEMBL): {len(unreviewed):,}")

cands = {}
for side, pfx in (("int", "interactor"), ("prt", "partner")):
    for row in res[res[f"{pfx}_base"].isin(unreviewed)].to_dict("records"):
        c = _swissprot_candidate(row, side)
        if c and c != row[f"{pfx}_base"]:
            cands.setdefault(row[f"{pfx}_base"], set()).add(c)
print(f"TrEMBL accessions with a Swiss-Prot candidate: {len(cands):,}")

print("fetching sequences (TrEMBL + candidates)...")
SEQS = fetch_isoforms(in_use | {c for v in cands.values() for c in v})

TREMBL_REMAP, _remap_log = {}, []
for tr, options in cands.items():
    tr_seq = SEQS.get(tr)
    if not tr_seq:
        continue
    for sp in sorted(options):
        hit = next((i for i in sorted(SEQS) if base_acc(i) == sp and SEQS[i] == tr_seq), None)
        if hit:
            TREMBL_REMAP[tr] = base_acc(hit)
            _remap_log.append({"trembl": tr, "swissprot": base_acc(hit),
                               "matched_isoform": hit, "length": len(tr_seq)})
            break

df_remap = pd.DataFrame(_remap_log)
df_remap.to_csv(INTERMEDIATE_DIR / "mapping_trembl_to_swissprot.csv", index=False)
print(f"remapped {len(TREMBL_REMAP):,} TrEMBL -> Swiss-Prot (sequence-identical); "
      f"{len(cands) - len(TREMBL_REMAP):,} candidates rejected on sequence mismatch")

for pfx in ("interactor", "partner"):
    res[f"{pfx}_base"] = res[f"{pfx}_base"].map(lambda a: TREMBL_REMAP.get(a, a))
    res[f"{pfx}_seed"] = res[f"{pfx}_seed"].map(
        lambda s: None if s and base_acc(s) in TREMBL_REMAP else s)

still = {a for a in set(res.interactor_base) | set(res.partner_base)
         if not ACC_REVIEWED.get(a, False)}
print(f"remaining TrEMBL accessions in use: {len(still):,}")

# %% [markdown]
# ## 5. Isoform assignment (per-clone consensus)
#
# For each **clone** — `(base accession, ORF id)`, falling back to `(base accession, None)`
# where the source has no ORF id — score every candidate isoform by how many of that
# clone's mutations validate against it, and take the highest scorer.
#
# Ties break on: the **source-provided isoform seed** (authoritative — `df_luke` expresses
# its mutations relative to `uniprot_isoform_ac`), then the **canonical** isoform, then
# lowest suffix number. Canonical is identified by sequence identity with the bare
# accession, never by suffix number.
#
# Finally the suffix is stripped **iff** the chosen isoform's sequence is identical to
# canonical — so `-1`-style aliases collapse and real isoforms never do.

# %%
needed = set(res["interactor_base"]) | set(res["partner_base"])
SEQS.update(fetch_isoforms(needed))   # cached; no-op for anything already held
print(f"accessions in play: {len(needed):,} | sequences held: {len(SEQS):,}")

CANONICAL = {b: SEQS[b] for b in needed if b in SEQS}
ISOFORMS = {}
for iso_id in SEQS:
    ISOFORMS.setdefault(base_acc(iso_id), []).append(iso_id)
missing_seq = sorted(b for b in needed if b not in CANONICAL)
print(f"accessions with no canonical sequence: {len(missing_seq)} {missing_seq[:10]}")

# %%
ISOFORM_TIES = []


def choose_isoform(bacc, muts, seeds):
    """Pick the isoform of `bacc` that validates the most of `muts`."""
    cands = sorted(ISOFORMS.get(bacc, []), key=iso_suffix)
    if not cands:
        return bacc, 0
    canon_seq = CANONICAL.get(bacc)
    seed_set = {s for s in seeds if s}
    scored = [(sum(validate_mutation(SEQS[c], m) for m in muts), c) for c in cands]
    best = max(s for s, _ in scored)
    tied = [c for s, c in scored if s == best]
    if len(tied) > 1:
        ISOFORM_TIES.append({"base_accession": bacc, "n_mutations": len(muts),
                             "score": best, "candidates": ",".join(tied)})
    for c in tied:                                   # 1. source seed
        if c in seed_set:
            return c, best
    for c in tied:                                   # 2. canonical (by sequence)
        if canon_seq is not None and SEQS[c] == canon_seq:
            return c, best
    return sorted(tied, key=iso_suffix)[0], best     # 3. lowest suffix


# Clone key: (base accession, ORF id, dataset). The dataset must be part of the key:
# where a source has no ORF id (Maxim, Sahni, Fragoza) a (base, None) key would pool
# rows from *different experiments* into one "clone" and let one dataset's majority pick
# an isoform that breaks the other's mutations. Real case: P22557 (ALAS2) -- Maxim and
# Sahni rows pooled, P22557-2 wins 7/11, and the 4 Sahni mutations that validate on the
# canonical are lost. Keying on dataset recovers 12 rows overall (90 -> 78 failures).
res["_clone"] = list(zip(res["interactor_base"], res["int_orf"], res["dataset"]))
choice = {}
for clone, grp in res.groupby("_clone", dropna=False):
    bacc = clone[0]
    choice[clone] = choose_isoform(bacc, grp["mutation"].tolist(),
                                   grp["interactor_seed"].tolist())

res["interactor"] = [choice[c][0] for c in res["_clone"]]

# Collapse the suffix iff the chosen isoform's sequence == canonical.
collapsed = 0
final_ids = []
for iso_id in res["interactor"]:
    b = base_acc(iso_id)
    if iso_id != b and CANONICAL.get(b) is not None and SEQS.get(iso_id) == CANONICAL[b]:
        final_ids.append(b)
        collapsed += 1
    else:
        final_ids.append(iso_id)
res["interactor"] = final_ids
res["interactor_is_isoform"] = res["interactor"].str.contains("-")
res["partner"] = res["partner_base"]

df_ties = pd.DataFrame(ISOFORM_TIES).drop_duplicates()
df_ties.to_csv(INTERMEDIATE_DIR / "isoform_ties.csv", index=False)

print(f"clones scored              : {len(choice):,}")
print(f"rows collapsed to canonical: {collapsed:,}")
print(f"rows on a real isoform     : {int(res.interactor_is_isoform.sum()):,}")
print(f"isoform ties logged        : {len(df_ties):,}")

# Sanity: nothing was collapsed whose sequence differs from canonical.
for iso_id in set(res["interactor"]):
    if "-" not in iso_id:
        continue
    assert SEQS.get(iso_id) != CANONICAL.get(base_acc(iso_id)), \
        f"{iso_id} identical to canonical but not collapsed"
print("collapse rule verified.")

# %% [markdown]
# ## 6. Attach sequences and validate every mutation
#
# Rows failing validation are **kept** with `mutation_validated = False` (per decision)
# and also written to `failed_mutation_validation.csv`.

# %%
res["interactor_sequence"] = res["interactor"].map(SEQS)
res["partner_sequence"] = res["partner"].map(CANONICAL)
res["mutation_validated"] = [
    validate_mutation(s, m) for s, m in zip(res["interactor_sequence"], res["mutation"])
]

print("mutation_validated by dataset:")
print((res.groupby("dataset")["mutation_validated"]
       .agg(n="size", validated="sum", pct=lambda s: round(100 * s.mean(), 2))
       .to_string()))
print(f"\noverall: {100 * res.mutation_validated.mean():.2f}%")

bad = res[~res.mutation_validated]
failed = bad[["dataset", "source_row_id", "interactor", "int_symbol", "mutation",
              "interactor_route", "interactor_is_isoform", "partner"]].copy()
failed["interactor_length"] = [
    len(s) if isinstance(s, str) else 0 for s in bad["interactor_sequence"]
]
failed.to_csv(INTERMEDIATE_DIR / "failed_mutation_validation.csv", index=False)
print(f"\nfailed_mutation_validation.csv: {len(failed):,} rows")

# %% [markdown]
# ### 6b. Backfill missing gene symbols from UniProt
#
# The symbol columns are required on every row, but Fragoza's raw files carry only
# Entrez ids and accessions — no symbols at all. UniProt's primary gene name for the
# resolved accession fills them in. Source-provided symbols always win; this only ever
# fills a blank.

# %%
before = {s: res[f"{s}_symbol"].notna().mean() for s in ("int", "prt")}
renamed = []
for side, pfx in (("int", "interactor"), ("prt", "partner")):
    col = f"{side}_symbol"
    canon = res[pfx].map(lambda a: ACC_GENE.get(base_acc(a)))
    # Prefer UniProt's CURRENT primary symbol over the source's. Sources were published
    # years apart and use different-era HGNC names for the same accession, so taking the
    # source symbol leaves one accession with two names and makes the collapsed value
    # depend on row order. Observed: Q8NA61 SPERT/CBY2, Q8N6L0 CCDC155/KASH5,
    # Q8NI60 ADCK3/COQ8A, Q8IYA8 CCDC36/IHO1, Q08117 AES/TLE5 -- UniProt has the current
    # name in every case. Fall back to the source symbol only where UniProt has none.
    diff = res[col].notna() & canon.notna() & (res[col] != canon)
    renamed.append(pd.DataFrame({"side": pfx, "accession": res.loc[diff, pfx],
                                 "symbol_in_source": res.loc[diff, col],
                                 "symbol_uniprot": canon[diff]}).drop_duplicates())
    res[col] = canon.where(canon.notna(), res[col])

df_renamed = pd.concat(renamed, ignore_index=True) if renamed else pd.DataFrame()
df_renamed.to_csv(INTERMEDIATE_DIR / "symbol_renamed_to_uniprot.csv", index=False)
print(f"symbols replaced by UniProt's current primary: {len(df_renamed):,} distinct "
      f"(accession, side) pairs -> symbol_renamed_to_uniprot.csv")

print("gene-symbol coverage (source -> after UniProt backfill):")
for side, pfx in (("int", "interactor"), ("prt", "partner")):
    print(f"  {pfx:11s} {100 * before[side]:6.2f}% -> {100 * res[f'{side}_symbol'].notna().mean():6.2f}%")
print("\nby dataset, after backfill:")
print((res.groupby("dataset")[["int_symbol", "prt_symbol"]]
       .apply(lambda g: pd.Series({"interactor_symbol_pct": round(100 * g.int_symbol.notna().mean(), 2),
                                   "partner_symbol_pct": round(100 * g.prt_symbol.notna().mean(), 2)}))
       .to_string()))

# %% [markdown]
# ## 7. Build the master
#
# One row per source observation. **Not deduplicated, conflicts retained.**
# `key_n_rows` and `key_has_conflict` are informational only — they filter nothing.

# %%
MASTER_COLS = [
    "interactor", "interactor_symbol", "interactor_orf_id",
    "partner", "partner_symbol", "partner_orf_id",
    "mutation", "perturbed", "dataset", "dataset_tier", "fragoza_source",
    "interactor_sequence", "partner_sequence",
    "log2fc_1p", "perturbation_LLR_1p",
    "log2fc_pooled", "perturbation_LLR_pooled",
    "mutation_validated", "source_row_id",
]

master = pd.DataFrame({
    "interactor": res["interactor"],
    "interactor_symbol": res["int_symbol"],
    "interactor_orf_id": res["int_orf"],
    "partner": res["partner"],
    "partner_symbol": res["prt_symbol"],
    "partner_orf_id": res["prt_orf"],
    "mutation": res["mutation"],
    "perturbed": res["perturbed"],
    "dataset": res["dataset"],
    "dataset_tier": res["dataset_tier"],
    "fragoza_source": res["fragoza_source"],
    "interactor_sequence": res["interactor_sequence"],
    "partner_sequence": res["partner_sequence"],
    "log2fc_1p": res["log2fc_1p"],
    "perturbation_LLR_1p": res["perturbation_LLR_1p"],
    "log2fc_pooled": res["log2fc_pooled"],
    "perturbation_LLR_pooled": res["perturbation_LLR_pooled"],
    "mutation_validated": res["mutation_validated"],
    "source_row_id": res["source_row_id"],
})[MASTER_COLS]

KEY = ["interactor", "mutation", "partner"]
g = master.groupby(KEY)
master["key_n_rows"] = g["perturbed"].transform("size")
master["key_has_conflict"] = g["perturbed"].transform("nunique") > 1

master.to_csv(MASTER_DIR / f"master_ppi_perturbation{SUFFIX}.csv", index=False)
print(f"master_ppi_perturbation{SUFFIX}.csv: {len(master):,} rows x {master.shape[1]} cols")
print(f"  rows in duplicate keys  : {int((master.key_n_rows > 1).sum()):,}")
print(f"  rows in conflicting keys: {int(master.key_has_conflict.sum()):,}")
print(f"  unique keys             : {master.groupby(KEY).ngroups:,}")
master.head(3)

# %% [markdown]
# ## 8. Dedup + conflict resolution, and the dataset files
#
# `resolve()` is applied **per subset**, never to the master. Applying it globally and
# then filtering would let a df_flo-vs-df_luke conflict delete rows from the
# df_luke-only file, where no conflict exists.

# %%
DROPPED_CONFLICTS = []
SUPERSEDED = []


def resolve(df, name):
    """Priority-drop legacy rows, drop remaining conflicts, collapse duplicates."""
    d = df.copy()
    n0 = len(d)

    # 1. VarChAMP-internal supersession ONLY: drop a 2025 VarChAMP row for a key that the
    #    same team's 2026 assay also measured. Never applied against Sahni/Fragoza.
    has_2026 = d.groupby(KEY)["dataset"].transform(lambda x: x.isin(VC_2026).any())
    superseded = d["dataset"].isin(VC_2025) & has_2026
    if superseded.any():
        SUPERSEDED.append(d[superseded].assign(subset=name))
    d = d[~superseded]
    n1 = len(d)

    # 2. Any surviving disagreement drops the whole key.
    conflict = d.groupby(KEY)["perturbed"].transform("nunique") > 1
    if conflict.any():
        c = d[conflict]
        DROPPED_CONFLICTS.append(
            c.groupby(KEY).agg(datasets=("dataset", lambda x: ",".join(sorted(set(x)))),
                               labels=("perturbed", lambda x: ",".join(map(str, sorted(set(x))))),
                               n_rows=("perturbed", "size"))
             .reset_index().assign(subset=name))
    d = d[~conflict]
    n2 = len(d)

    # 3. Collapse true duplicates; merge corroborating dataset tags.
    agg = {c: "first" for c in d.columns if c not in KEY + ["dataset"]}
    agg["dataset"] = lambda x: ",".join(sorted(set(x)))
    # a collapsed key can span tiers; joining avoids implying a single one
    agg["dataset_tier"] = lambda x: ",".join(sorted(set(x)))
    # fragoza_source can legitimately be multiple (the same variant appears in e.g. both
    # the exac and hgmd files), so join it too rather than keeping an arbitrary one.
    agg["fragoza_source"] = lambda x: ",".join(sorted({v for v in x if pd.notna(v)})) or None
    out = d.groupby(KEY, as_index=False).agg(agg)
    out = out[[c for c in MASTER_COLS if c in out.columns]]

    print(f"  {name:32s} {n0:>7,} -> vc-superseded {n0 - n1:>5,} -> conflict-drop "
          f"{n1 - n2:>5,} -> collapse {n2 - len(out):>5,} = {len(out):>7,}")
    return out


SUBSETS = {
    # name: (source tags, destination directory)
    "sahni_fragoza_varchamp_all": (list(TIER), DATASETS_DIR),
    "varchamp_all": ([TAG_LUKE, TAG_MAXIM, TAG_VC1P, TAG_CAVA], DATASETS_DIR),
    "sahni_fragoza": ([TAG_SAHNI, TAG_FRAGOZA], DATASETS_DIR),
    "varchamp_2026_only": ([TAG_LUKE, TAG_MAXIM], VARCHAMP_OTHER_DIR),
    "varchamp_luke_only": ([TAG_LUKE], VARCHAMP_OTHER_DIR),
    "sahni_only": ([TAG_SAHNI], SINGLE_SOURCE_DIR),
    "fragoza_only": ([TAG_FRAGOZA], SINGLE_SOURCE_DIR),
}

# Dataset files are training data: every row must have a mutation that actually matches
# its sequence. Rows failing validation (position past the end, or a different residue at
# that position) are excluded here -- they remain in the master, which is the complete
# record, and are itemised in intermediate_files/failed_mutation_validation.csv.
#
# This filter runs BEFORE resolve() on purpose: dropping an invalid row can change the
# outcome for its key, e.g. a valid/invalid pair that looked like a label conflict is no
# longer one once the invalid row is gone, so the valid row survives instead of both
# being dropped.
pool = master.drop(columns=["key_n_rows", "key_has_conflict"])
n_before = len(pool)
pool = pool[pool["mutation_validated"]].copy()
print(f"excluded {n_before - len(pool):,} rows failing mutation validation "
      f"({n_before:,} -> {len(pool):,}); master retains all of them")
print(master.loc[~master.mutation_validated, "dataset"].value_counts().to_string())

print("\nbuilding dataset files:")
outputs = {}
for name, (tags, dest) in SUBSETS.items():
    sub = pool[pool["dataset"].isin(tags)]
    out = resolve(sub, name)
    out.to_csv(dest / f"{name}{SUFFIX}.csv", index=False)
    outputs[f"{name}{SUFFIX}"] = out

df_conf = (pd.concat(DROPPED_CONFLICTS, ignore_index=True)
           if DROPPED_CONFLICTS else pd.DataFrame())
df_conf.to_csv(INTERMEDIATE_DIR / "dropped_conflicts.csv", index=False)
df_sup = (pd.concat(SUPERSEDED, ignore_index=True) if SUPERSEDED else pd.DataFrame())
if len(df_sup):
    df_sup = df_sup[["subset", "dataset", "interactor", "mutation", "partner",
                     "perturbed", "interactor_symbol", "partner_symbol", "source_row_id"]]
df_sup.to_csv(INTERMEDIATE_DIR / "varchamp_superseded_rows.csv", index=False)
print(f"varchamp_superseded_rows.csv: {len(df_sup):,} rows "
      f"(2025 VarChAMP rows replaced by the same team's 2026 assay)")
df_repairs.to_csv(INTERMEDIATE_DIR / "symbol_repairs.csv", index=False)
print(f"\ndropped_conflicts.csv: {len(df_conf):,} key/subset rows")

# %% [markdown]
# ## 9. QC report

# %%
def summarize(name, d):
    return {
        "file": name, "rows": len(d),
        "perturbed_pct": round(100 * d.perturbed.mean(), 2),
        "unique_interactors": d.interactor.nunique(),
        "unique_partners": d.partner.nunique(),
        "validated_pct": round(100 * d.mutation_validated.mean(), 2),
        "isoform_pct": round(100 * d.interactor.str.contains("-").mean(), 2),
        "with_int_orf_pct": round(100 * d.interactor_orf_id.notna().mean(), 2),
        "with_prt_orf_pct": round(100 * d.partner_orf_id.notna().mean(), 2),
        "int_symbol_pct": round(100 * d.interactor_symbol.notna().mean(), 2),
        "prt_symbol_pct": round(100 * d.partner_symbol.notna().mean(), 2),
    }


qc = pd.DataFrame(
    [summarize(f"master_ppi_perturbation{SUFFIX}", master)]
    + [summarize(f"  by dataset: {t}", master[master.dataset == t]) for t in TIER]
    + [summarize(n, d) for n, d in outputs.items()]
)
qc.to_csv(INTERMEDIATE_DIR / "qc_summary.csv", index=False)
pd.DataFrame(LOAD_STATS).to_csv(INTERMEDIATE_DIR / "load_stages.csv", index=False)
print(qc.to_string(index=False))

# %%
# --- Assertions --------------------------------------------------------------
assert master["mutation"].str.match(MISSENSE_RE).all(), "bad mutation format"
assert master["perturbed"].notna().all(), "NaN in perturbed"
assert master["interactor"].notna().all() and master["partner"].notna().all()

seq_conflicts = pd.concat([
    master[["interactor", "interactor_sequence"]].rename(
        columns={"interactor": "id", "interactor_sequence": "seq"}),
    master[["partner", "partner_sequence"]].rename(
        columns={"partner": "id", "partner_sequence": "seq"}),
]).dropna().groupby("id")["seq"].nunique()
assert (seq_conflicts <= 1).all(), f"accessions with 2 sequences: {seq_conflicts[seq_conflicts > 1]}"

# Dataset files must contain no unvalidated mutations; the master may (by design).
for _n, _d in outputs.items():
    assert _d["mutation_validated"].all(), f"{_n} contains unvalidated mutations"
print(f"all {len(outputs)} dataset files are 100% mutation-validated; "
      f"master retains {int((~master.mutation_validated).sum())} flagged rows")

loaded = sum(1 for _ in res.index)
print(f"master rows == mapped rows: {len(master) == loaded} ({len(master):,})")
print("all assertions passed.")

# %%
# --- Targeted spot-checks from the plan ---------------------------------------
luke_rows = master[master.dataset == TAG_LUKE]
same = (luke_rows.interactor_symbol == luke_rows.partner_symbol).mean()
print(f"[partner-symbol fix] df_luke rows where interactor_symbol == partner_symbol: "
      f"{100 * same:.2f}%  (old notebook: 100%)")
print(luke_rows[luke_rows.interactor_symbol == "LITAF"]
      [["interactor", "interactor_symbol", "partner", "partner_symbol"]].head(3).to_string(index=False))

# Isoform suffixes that add no information must be gone: a suffix survives only when
# that isoform's sequence actually differs from the canonical one.
seeded = res[res.interactor_seed.notna()]
collapsed_seeds = (seeded.interactor_seed != seeded.interactor).sum()
print(f"\n[suffix stripping] source-supplied isoform seeds: {len(seeded):,} | "
      f"collapsed to a different id: {collapsed_seeds:,}")
is_iso = master.interactor.str.contains("-")   # derived, not stored: cannot drift from the id
print(f"  ids still carrying a suffix: {is_iso.sum():,} ({100 * is_iso.mean():.2f}%)")
surviving = sorted({i for i in master.interactor if "-" in i})
# The invariant is sequence-based, NOT suffix-based. A '-1' can legitimately survive:
# for e.g. Q9BRI3 and Q9H0P0 UniProt returns ACC-1 alongside the canonical with a
# genuinely different sequence (323 vs 372 aa, 297 vs 336 aa) -- the canonical is not
# isoform 1 there. Asserting "no -1 survives" would be wrong and would strip a real isoform.
for i in surviving:
    assert SEQS[i] != CANONICAL[base_acc(i)], f"{i} is sequence-identical to canonical"
kept_ones = [i for i in surviving if i.endswith("-1")]
print(f"  verified: every surviving suffix differs in sequence from its canonical "
      f"({len(surviving):,} distinct isoform ids; {len(kept_ones)} of them '-1', "
      f"where canonical is not isoform 1: {kept_ones})")

plp1 = master[(master.interactor_symbol == "PLP1") & (master.mutation == "L189P")]
print(f"\n[isoform kept] PLP1 L189P -> {sorted(set(plp1.interactor))} (expect P60201-2)")
print(f"[canonical collapsed] any bare Q99732 rows: "
      f"{(master.interactor == 'Q99732').sum():,} | any Q99732-1: "
      f"{(master.interactor == 'Q99732-1').sum():,} (expect 0)")

sahni = master[master.dataset == TAG_SAHNI]
print(f"\n[label polarity] Sahni perturbed rate: {100 * sahni.perturbed.mean():.2f}% "
      f"(raw file: 586/1613 = 36.33%)")

print(f"\nFiles written under {OUT_DIR}:")
for f in sorted(OUT_DIR.rglob("*.csv")):
    print(f"  {str(f.relative_to(OUT_DIR)):58s} {f.stat().st_size / 1e6:8.2f} MB")
