# -*- coding: utf-8 -*-
"""Komenda repair-clinical: naprawa kowariantów bez ruszania etykiet i genów."""
from __future__ import annotations

import polars as pl
import pytest
from typer.testing import CliRunner

from src.cli import app

COVARIATES = ["age_at_index", "gender", "ajcc_pathologic_stage"]
runner = CliRunner()


@pytest.fixture
def broken_dataset(survival_ds, tmp_path):
    """Zbiór z pustą kolumną płci — objaw budowy z niepełnej kliniki."""
    path = tmp_path / "survival_dataset.parquet"
    survival_ds.with_columns(pl.lit(None, dtype=pl.String).alias("gender")).write_parquet(path)
    return path


def _run(dataset, clinical_tsv, output):
    return runner.invoke(app, [
        "repair-clinical",
        "--dataset", str(dataset),
        "--clinical", str(clinical_tsv),
        "--output", str(output),
    ])


def test_naprawia_pusta_kolumne(broken_dataset, clinical_tsv_path, tmp_path):
    out = tmp_path / "fixed.parquet"
    result = _run(broken_dataset, clinical_tsv_path, out)

    assert result.exit_code == 0, result.output
    fixed = pl.read_parquet(out)
    filled = int(fixed["gender"].is_not_null().sum())
    assert filled / fixed.height > 0.9


def test_nie_rusza_etykiet_przezycia_ani_genow(broken_dataset, clinical_tsv_path, tmp_path):
    out = tmp_path / "fixed.parquet"
    before = pl.read_parquet(broken_dataset)
    assert _run(broken_dataset, clinical_tsv_path, out).exit_code == 0

    after = pl.read_parquet(out)
    assert after["time"].to_list() == before["time"].to_list()
    assert after["event"].to_list() == before["event"].to_list()
    genes = [c for c in before.columns if c.startswith("ENSG")]
    assert after.select(genes).equals(before.select(genes))


def test_zachowuje_kolejnosc_kolumn(broken_dataset, clinical_tsv_path, tmp_path):
    out = tmp_path / "fixed.parquet"
    assert _run(broken_dataset, clinical_tsv_path, out).exit_code == 0
    assert pl.read_parquet(out).columns == pl.read_parquet(broken_dataset).columns


def test_raportuje_kompletnosc_przed_i_po(broken_dataset, clinical_tsv_path, tmp_path):
    result = _run(broken_dataset, clinical_tsv_path, tmp_path / "fixed.parquet")
    assert "PRZED" in result.output and "PO" in result.output
    assert "gender 0.0%" in result.output


def test_brak_zbioru_konczy_bledem(clinical_tsv_path, tmp_path):
    result = _run(tmp_path / "nie_ma.parquet", clinical_tsv_path, tmp_path / "out.parquet")
    assert result.exit_code == 1


def test_zbior_bez_case_id_konczy_bledem(survival_ds, clinical_tsv_path, tmp_path):
    dataset = tmp_path / "bez_case.parquet"
    survival_ds.drop("case_id").write_parquet(dataset)
    result = _run(dataset, clinical_tsv_path, tmp_path / "out.parquet")
    assert result.exit_code == 1
    assert "case_id" in result.output
