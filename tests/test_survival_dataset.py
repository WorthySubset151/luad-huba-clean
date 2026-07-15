# -*- coding: utf-8 -*-
"""Budowa zbioru przeżywalności + strażnik kompletności kowariantów klinicznych."""
from __future__ import annotations

import polars as pl
import pytest

from src.transform.survival_dataset import (
    METADATA_COLUMNS,
    SurvivalDatasetError,
    build_survival_dataset,
)

COVARIATES = ["age_at_index", "gender", "ajcc_pathologic_stage"]


def test_zbior_ma_metadane_i_geny(survival_ds, gene_columns):
    assert survival_ds.height > 0
    assert gene_columns
    for col in METADATA_COLUMNS:
        assert col in survival_ds.columns


def test_build_zachowuje_kowarianty_kliniczne(survival_ds):
    # regresja: wdrożony zbiór potrafił mieć kolumnę kliniczną pustą w 100%
    for col in COVARIATES:
        filled = int(survival_ds[col].is_not_null().sum())
        assert filled / survival_ds.height > 0.9, f"{col}: {filled}/{survival_ds.height}"


def test_kolejnosc_kolumn_metadane_przed_genami(survival_ds, gene_columns):
    assert survival_ds.columns[: len(METADATA_COLUMNS)] == METADATA_COLUMNS
    assert set(survival_ds.columns) == set(METADATA_COLUMNS) | set(gene_columns)


def test_kazda_probka_ma_etykiete(survival_ds):
    assert survival_ds["time"].null_count() == 0
    assert survival_ds["event"].null_count() == 0
    assert set(survival_ds["event"].unique().to_list()) <= {0, 1}


def test_probki_maja_pacjenta_z_kliniki(survival_ds, clinical):
    known = set(clinical["case_submitter_id"].to_list())
    assert set(survival_ds["case_id"].to_list()) <= known


def test_straznik_przerywa_gdy_kowariant_pusty(clinical, expression_inputs):
    matrix, sheet = expression_inputs
    broken = clinical.with_columns(pl.lit(None, dtype=pl.String).alias("gender"))
    with pytest.raises(SurvivalDatasetError) as exc:
        build_survival_dataset(matrix, sheet, broken)
    message = str(exc.value)
    assert "gender" in message and "płeć" in message
    assert "repair-clinical" in message  # komunikat wskazuje remedium


def test_straznik_ostrzega_przy_czesciowych_brakach(clinical, expression_inputs, capsys):
    matrix, sheet = expression_inputs
    values = clinical["gender"].to_list()
    half = [None if i % 2 == 0 else v for i, v in enumerate(values)]
    partial = clinical.with_columns(pl.Series("gender", half, dtype=pl.String))

    dataset = build_survival_dataset(matrix, sheet, partial)

    assert dataset.height > 0  # ostrzega, ale nie blokuje
    assert "UWAGA" in capsys.readouterr().err


def test_straznik_przepuszcza_dobre_dane(survival_ds):
    # fixture przechodzi przez build — brak wyjątku jest tu asercją
    assert survival_ds.height > 0
