"""Tests for utils/sequences.py -- the shared FASTA reader.

Covers the three real input formats found across the repo (WT_VT, UniProt,
NeuroDev) and, specifically, the `h5_safe_key` whole-vs-first-token collision
that would silently merge every variant embedding onto its wild type if it
were ever migrated with the wrong default. That bug was caught before any
caller adopted it -- these tests are what keeps it from coming back.
"""
import gzip

import pytest

from utils.sequences import (
    accession_only,
    first_token,
    h5_safe_key,
    iter_fasta,
    pipe_field,
    read_fasta,
    whole_header,
)


def _write_fasta(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


# -- header policies ----------------------------------------------------------

def test_first_token():
    assert first_token("P25054 S305R") == "P25054"
    assert first_token("P25054") == "P25054"
    assert first_token("") is None


def test_accession_only_drops_variant_records():
    assert accession_only("P25054") == "P25054"
    assert accession_only("P25054 S305R") is None


def test_pipe_field():
    parser = pipe_field(1)
    assert parser("sp|P43277|H13_MOUSE Histone H1.3") == "P43277"
    assert parser("no-pipes-here") is None


def test_whole_header():
    assert whole_header("P25054 S305R") == "P25054 S305R"
    assert whole_header("P25054") == "P25054"
    assert whole_header("") is None


# -- the h5_safe_key collision (§C latent bug) --------------------------------

def test_h5_safe_key_requires_whole_argument():
    # No default is offered -- a caller must decide. TypeError if it doesn't.
    with pytest.raises(TypeError):
        h5_safe_key("P25054")  # type: ignore[call-arg]


def test_h5_safe_key_whole_false_collapses_wt_and_variant():
    # This IS the collision -- documented, not accidental, for callers that
    # only ever key on the accession (want first-token behaviour on purpose).
    assert h5_safe_key("P25054", whole=False) == "P25054"
    assert h5_safe_key("P25054 S305R", whole=False) == "P25054"


def test_h5_safe_key_whole_true_keeps_wt_and_variant_distinct():
    # This is the fix: the WT_VT ProtT5 writers MUST pass whole=True, or every
    # variant collapses onto its wild type and the embedding store silently
    # loses ~all of its variant records.
    wt_key = h5_safe_key("P25054", whole=True)
    vt_key = h5_safe_key("P25054 S305R", whole=True)
    assert wt_key != vt_key
    assert wt_key == "P25054"
    assert vt_key == "P25054 S305R"


def test_h5_safe_key_mangles_slash_and_dot():
    assert h5_safe_key("NP_000005.3 A2M/variant", whole=True) == "NP_000005_3 A2M_variant"


def test_h5_safe_key_empty_header_is_none():
    assert h5_safe_key("", whole=True) is None
    assert h5_safe_key("", whole=False) is None


# -- reading: the three real formats ------------------------------------------

def test_iter_fasta_wt_vt_format(tmp_path):
    # Format A, from clinvar/cosmic/hgmd/etc: bare accession (WT) and
    # space-delimited accession+mutation (VT) in the same file.
    path = _write_fasta(tmp_path, "wt_vt.fasta",
                        ">P25054\nMKTAYIAKQRQ\n>P25054 S305R\nMKTAYIAKQRR\n")
    records = list(iter_fasta(path, whole_header))
    assert records == [("P25054", "MKTAYIAKQRQ"), ("P25054 S305R", "MKTAYIAKQRR")]


def test_iter_fasta_uniprot_format(tmp_path):
    # Format B: canonical UniProt sp|acc|name header.
    path = _write_fasta(
        tmp_path, "uniprot.fasta",
        ">sp|P43277|H13_MOUSE Histone H1.3 OS=Mus musculus\nMSETAPAAPAAP\n")
    records = list(iter_fasta(path, pipe_field(1)))
    assert records == [("P43277", "MSETAPAAPAAP")]


def test_iter_fasta_neurodev_format(tmp_path):
    # Format C: pipe fields + trailing tab cohort tag. record.id-equivalent
    # (first_token) drops the tab-delimited cohort tag but keeps the pipe
    # composite, matching map_autism.py's actual behaviour.
    path = _write_fasta(
        tmp_path, "neurodev.fasta",
        ">NP_000005|A2M|p.G779R\tcase_dataset\nMGKNKLL\n")
    records = list(iter_fasta(path, first_token))
    assert records == [("NP_000005|A2M|p.G779R", "MGKNKLL")]


def test_iter_fasta_gzip(tmp_path):
    path = tmp_path / "wt.fasta.gz"
    with gzip.open(path, "wt") as fh:
        fh.write(">P25054\nMKTAYIAKQRQ\n")
    records = list(iter_fasta(path, first_token))
    assert records == [("P25054", "MKTAYIAKQRQ")]


def test_iter_fasta_multiline_sequence(tmp_path):
    path = _write_fasta(tmp_path, "multiline.fasta",
                        ">P25054\nMKTA\nYIAK\nQRQ\n")
    records = list(iter_fasta(path, first_token))
    assert records == [("P25054", "MKTAYIAKQRQ")]


# -- read_fasta duplicate policies ---------------------------------------------

def test_read_fasta_on_duplicate_first(tmp_path):
    path = _write_fasta(tmp_path, "dup.fasta", ">A\nAAA\n>A\nBBB\n")
    assert read_fasta(path, first_token, on_duplicate="first") == {"A": "AAA"}


def test_read_fasta_on_duplicate_last(tmp_path):
    path = _write_fasta(tmp_path, "dup.fasta", ">A\nAAA\n>A\nBBB\n")
    assert read_fasta(path, first_token, on_duplicate="last") == {"A": "BBB"}


def test_read_fasta_on_duplicate_raise(tmp_path):
    path = _write_fasta(tmp_path, "dup.fasta", ">A\nAAA\n>A\nBBB\n")
    with pytest.raises(ValueError):
        read_fasta(path, first_token, on_duplicate="raise")


def test_read_fasta_on_duplicate_all(tmp_path):
    path = _write_fasta(tmp_path, "dup.fasta", ">A\nAAA\n>A\nBBB\n>A\nAAA\n")
    assert read_fasta(path, first_token, on_duplicate="all") == {"A": ["AAA", "BBB"]}


def test_accession_only_drops_records_with_no_valid_parser_result(tmp_path):
    path = _write_fasta(tmp_path, "mixed.fasta",
                        ">P25054\nAAA\n>P25054 S305R\nBBB\n")
    # accession_only returns None for the variant record -> dropped.
    assert read_fasta(path, accession_only) == {"P25054": "AAA"}
