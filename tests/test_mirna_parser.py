# -*- coding: utf-8 -*-
"""Parser miRNA Expression Quantification: kolumny, typy, walidacja."""
from __future__ import annotations

import polars as pl
import pytest

from src.ingest.mirna_parser import MirnaParserError, parse_mirna_quantification

HEADER = "miRNA_ID\tread_count\treads_per_million_miRNA_mapped\tcross-mapped"
ROWS = [
    "hsa-let-7a-1\t55243\t9869.306676\tN",
    "hsa-mir-21\t12000\t2144.5\tN",
    "hsa-mir-1-1\t0\t0\tN",
    "hsa-mir-155\t843\t150.6\tY",
]


@pytest.fixture
def mirna_file(tmp_path):
    path = tmp_path / "mirnas.quantification.txt"
    path.write_text("\n".join([HEADER, *ROWS]) + "\n", encoding="utf-8")
    return path


def test_parsuje_wszystkie_wiersze(mirna_file):
    df = parse_mirna_quantification(mirna_file)
    assert df.height == len(ROWS)
    assert df.columns == ["miRNA_ID", "read_count", "reads_per_million_miRNA_mapped", "cross-mapped"]


def test_typy_liczbowe(mirna_file):
    df = parse_mirna_quantification(mirna_file)
    assert df["read_count"].dtype == pl.Int64
    assert df["reads_per_million_miRNA_mapped"].dtype == pl.Float64


def test_zachowuje_zerowe_miRNA(mirna_file):
    # miRNA o zerowej ekspresji nie może być odfiltrowane na etapie parsera
    df = parse_mirna_quantification(mirna_file)
    zero = df.filter(pl.col("miRNA_ID") == "hsa-mir-1-1")
    assert zero.height == 1
    assert zero["read_count"][0] == 0


def test_rozpoznaje_rodziny_let_i_mir(mirna_file):
    df = parse_mirna_quantification(mirna_file)
    ids = set(df["miRNA_ID"].to_list())
    assert "hsa-let-7a-1" in ids and "hsa-mir-21" in ids


def test_brak_pliku_zglasza_blad():
    with pytest.raises(FileNotFoundError):
        parse_mirna_quantification("/nie/ma/mirnas.txt")


def test_brak_wymaganych_kolumn(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("kol_a\tkol_b\n1\t2\n", encoding="utf-8")
    with pytest.raises(MirnaParserError):
        parse_mirna_quantification(bad)


def test_identyfikatory_spoza_mirbase(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text(f"{HEADER}\nENSG00000146648\t10\t1.0\tN\n", encoding="utf-8")
    with pytest.raises(MirnaParserError):
        parse_mirna_quantification(bad)


def test_duplikaty_mirna_id(tmp_path):
    dup = tmp_path / "dup.txt"
    dup.write_text("\n".join([HEADER, ROWS[0], ROWS[0]]) + "\n", encoding="utf-8")
    with pytest.raises(MirnaParserError):
        parse_mirna_quantification(dup)
