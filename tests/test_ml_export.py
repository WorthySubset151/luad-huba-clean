# -*- coding: utf-8 -*-
"""Eksport zbioru pod ML: brak wycieku, fit tylko na train, format i determinizm.

Najważniejsze testy w repozytorium — błąd wycieku nie wywala się głośno, tylko
po cichu zawyża wyniki modelu, więc jedyną obroną jest asercja.
"""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.analysis.ml_export import (
    DEFAULT_EXPORT_FORMAT,
    EXPORT_FORMATS,
    build_ml_bundle,
    prepare_ml_dataset,
)

TOP_K = 20


@pytest.fixture
def result(survival_ds):
    return prepare_ml_dataset(survival_ds, top_k=TOP_K, test_frac=0.25, seed=42)


# --- brak wycieku ---------------------------------------------------------

def test_brak_wspolnych_pacjentow_train_test(result):
    train_cases = set(result["y_train"]["case_id"])
    test_cases = set(result["y_test"]["case_id"])
    assert not (train_cases & test_cases)


def test_manifest_potwierdza_brak_wycieku(result):
    split = result["manifest"]["podzial"]
    assert split["pacjenci_wspolni_train_test"] == 0
    assert split["brak_wycieku_pacjenta"] is True


def test_brak_wycieku_takze_bez_deduplikacji(survival_ds):
    """Bez dedup pacjent ma kilka próbek i tylko grupowanie chroni przed wyciekiem."""
    result = prepare_ml_dataset(survival_ds, top_k=TOP_K, test_frac=0.25, seed=42, dedup=False)
    assert result["manifest"]["deduplikacja"]["wlaczona"] is False
    assert not set(result["y_train"]["case_id"]) & set(result["y_test"]["case_id"])
    assert result["manifest"]["podzial"]["pacjenci_wspolni_train_test"] == 0


def test_standaryzacja_liczona_tylko_na_train(result):
    """Asercje per gen, nie globalne.

    Średnia po całej macierzy wychodzi ~0 nawet gdy standaryzacja zerknie do testu
    (odchylenia genów się znoszą), więc globalna asercja przepuszcza wyciek.
    Fit wyłącznie na train wymusza, by KAŻDA kolumna train miała średnią 0 i sd 1.
    """
    train = result["X_train"].drop(columns=["sample_id"]).to_numpy()
    assert np.abs(train.mean(axis=0)).max() < 1e-8
    assert np.abs(train.std(axis=0) - 1.0).max() < 1e-8
    # test przeskalowany parametrami train, więc własnej średniej 0 mieć nie może
    test = result["X_test"].drop(columns=["sample_id"]).to_numpy()
    assert np.abs(test.mean(axis=0)).max() > 1e-8


def test_standaryzacja_nie_zaglada_do_testu(survival_ds):
    """Zmiana wartości wyłącznie w teście nie może ruszyć macierzy treningowej."""
    import polars as pl

    base = prepare_ml_dataset(survival_ds, top_k=TOP_K, test_frac=0.25, seed=42)
    test_samples = list(set(base["y_test"]["sample_id"]))
    genes = [c for c in survival_ds.columns if c.startswith("ENSG")]
    shifted = survival_ds.with_columns([
        pl.when(pl.col("sample_id").is_in(test_samples))
        .then(pl.col(g) + 500.0)
        .otherwise(pl.col(g))
        .alias(g)
        for g in genes
    ])
    after = prepare_ml_dataset(shifted, top_k=TOP_K, test_frac=0.25, seed=42)

    assert list(after["X_train"]["sample_id"]) == list(base["X_train"]["sample_id"])
    np.testing.assert_allclose(
        after["X_train"].drop(columns=["sample_id"]).to_numpy(),
        base["X_train"].drop(columns=["sample_id"]).to_numpy(),
    )


def test_selekcja_cech_nie_zaglada_do_testu(survival_ds):
    """Zmiana wyłącznie wartości w teście nie może zmienić wyboru cech."""
    base = prepare_ml_dataset(survival_ds, top_k=TOP_K, test_frac=0.25, seed=42)
    test_samples = set(base["y_test"]["sample_id"])

    import polars as pl
    genes = [c for c in survival_ds.columns if c.startswith("ENSG")]
    perturbed = survival_ds.with_columns([
        pl.when(pl.col("sample_id").is_in(list(test_samples)))
        .then(pl.col(g) * 1000.0)
        .otherwise(pl.col(g))
        .alias(g)
        for g in genes
    ])
    after = prepare_ml_dataset(perturbed, top_k=TOP_K, test_frac=0.25, seed=42)
    assert base["selected_genes"] == after["selected_genes"]


# --- deduplikacja i podział ----------------------------------------------

def test_deduplikacja_zostawia_jedna_probke_na_pacjenta(result):
    dedup = result["manifest"]["deduplikacja"]
    assert dedup["n_po"] < dedup["n_przed"]
    for split in ("y_train", "y_test"):
        cases = result[split]["case_id"]
        assert len(set(cases)) == len(cases)


def test_podzial_niepusty_i_rozlaczny(result):
    assert len(result["X_train"]) > 0
    assert len(result["X_test"]) > 0
    assert not set(result["y_train"]["sample_id"]) & set(result["y_test"]["sample_id"])


def test_stratyfikacja_zblizone_odsetki_zdarzen(result):
    split = result["manifest"]["podzial"]
    assert abs(split["event_rate_train"] - split["event_rate_test"]) < 0.25


# --- cechy ----------------------------------------------------------------

def test_te_same_kolumny_cech_w_train_i_test(result):
    assert list(result["X_train"].columns) == list(result["X_test"].columns)
    assert len(result["selected_genes"]) == TOP_K


def test_odrzuca_geny_o_zerowej_wariancji(result):
    assert result["manifest"]["selekcja_cech"]["zerowa_wariancja_odrzucone"] >= 0
    train = result["X_train"].drop(columns=["sample_id"]).to_numpy()
    assert (train.std(axis=0) > 0).all()


def test_determinizm_dla_tego_samego_seeda(survival_ds):
    a = prepare_ml_dataset(survival_ds, top_k=TOP_K, test_frac=0.25, seed=42)
    b = prepare_ml_dataset(survival_ds, top_k=TOP_K, test_frac=0.25, seed=42)
    assert list(a["X_train"]["sample_id"]) == list(b["X_train"]["sample_id"])
    assert a["selected_genes"] == b["selected_genes"]


def test_inny_seed_daje_inny_podzial(survival_ds):
    a = prepare_ml_dataset(survival_ds, top_k=TOP_K, test_frac=0.25, seed=1)
    b = prepare_ml_dataset(survival_ds, top_k=TOP_K, test_frac=0.25, seed=2)
    assert list(a["X_train"]["sample_id"]) != list(b["X_train"]["sample_id"])


def test_brak_kolumn_genow_zglasza_blad(survival_ds, gene_columns):
    with pytest.raises(ValueError):
        prepare_ml_dataset(survival_ds.drop(gene_columns), top_k=TOP_K)


# --- archiwum -------------------------------------------------------------

def test_domyslny_format_to_parquet():
    assert DEFAULT_EXPORT_FORMAT == "parquet"


@pytest.mark.parametrize("fmt", EXPORT_FORMATS)
def test_archiwum_zawiera_komplet_plikow(result, fmt):
    bundle = zipfile.ZipFile(io.BytesIO(build_ml_bundle(result, fmt=fmt)))
    expected = {f"X_train.{fmt}", f"X_test.{fmt}", f"y_train.{fmt}", f"y_test.{fmt}",
                "selected_genes.txt", "manifest.json", "README.txt"}
    assert expected <= set(bundle.namelist())


def test_parquet_zachowuje_typy_i_ksztalt(result):
    bundle = zipfile.ZipFile(io.BytesIO(build_ml_bundle(result, fmt="parquet")))
    x = pd.read_parquet(io.BytesIO(bundle.read("X_train.parquet")))
    y = pd.read_parquet(io.BytesIO(bundle.read("y_train.parquet")))
    assert x.shape == result["X_train"].shape
    assert list(x.columns) == list(result["X_train"].columns)
    assert all(t.kind == "f" for t in x.drop(columns=["sample_id"]).dtypes)
    # event jest boolowski (tego oczekuje scikit-survival); int też byłby poprawny
    assert y["event"].dtype.kind in "iub"


def test_manifest_w_archiwum_opisuje_uzyty_format(result):
    bundle = zipfile.ZipFile(io.BytesIO(build_ml_bundle(result, fmt="csv")))
    manifest = json.loads(bundle.read("manifest.json"))
    assert manifest["eksport"]["format_tabel"] == "csv"
    assert "X_train.csv" in manifest["eksport"]["pliki"]


def test_lista_genow_zgodna_z_kolumnami(result):
    bundle = zipfile.ZipFile(io.BytesIO(build_ml_bundle(result)))
    listed = bundle.read("selected_genes.txt").decode("utf-8").split("\n")
    assert listed == result["selected_genes"]


def test_manifest_niesie_rekomendacje_modelowe(result):
    # rzeczy, których nie wolno wpiec w dane, muszą zostać przekazane modelarzowi
    hints = " ".join(result["manifest"]["zalecane_ustawienia_modelu"]).lower()
    for keyword in ("groupkfold", "regularyzacj", "wag"):
        assert keyword in hints


def test_nieznany_format_zglasza_blad(result):
    with pytest.raises(ValueError):
        build_ml_bundle(result, fmt="xlsx")
