# -*- coding: utf-8 -*-
"""Metryki gotowości pod ML: mapowanie metryki, kompletność, odporność PCA, werdykt."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.analysis.readiness_report import _eta_squared, _pc_scores, ml_readiness_report


def _metric(report, fragment):
    for group in report["groups"]:
        for item in group["metrics"]:
            if fragment.lower() in item["label"].lower():
                return item
    raise AssertionError(f"brak metryki zawierającej {fragment!r}")


@pytest.fixture
def report(survival_ds):
    return ml_readiness_report(survival_ds, {"distribution": {"metric": "TPM"}})


@pytest.mark.parametrize(("etykieta", "status"), [
    ("TPM", "green"),
    ("FPKM/inne (nieokreślona)", "yellow"),
    ("zliczenia (counts)", "red"),
])
def test_mapowanie_metryki_normalizacji(survival_ds, etykieta, status):
    # regresja: porównanie ze skrótem ("tpm") zamiast etykietą opisową wrzucało
    # każdą metrykę do czerwonego "counts"
    report = ml_readiness_report(survival_ds, {"distribution": {"metric": etykieta}})
    item = _metric(report, "Metryka normalizacji")
    assert item["status"] == status
    assert item["value"] == etykieta


def test_kompletność_zielona_dla_dobrych_danych(report):
    assert _metric(report, "Kompletność")["status"] == "green"


def test_kompletnosc_wskazuje_puste_pole(survival_ds):
    broken = survival_ds.with_columns(pl.lit(None, dtype=pl.String).alias("gender"))
    item = _metric(ml_readiness_report(broken, None), "Kompletność")
    assert item["status"] == "red"
    assert "płeć 0%" in item["value"]
    assert "płeć" in item["action"]


def test_kompletnosc_raportuje_kazde_pole_osobno(report):
    value = _metric(report, "Kompletność")["value"]
    for field in ("wiek", "płeć", "stadium"):
        assert field in value


def test_metryki_wymiarow(report, survival_ds, gene_columns):
    assert _metric(report, "Liczba cech")["value"].replace(" ", "") == str(len(gene_columns))
    assert _metric(report, "p/n")["status"] in {"green", "yellow", "red"}
    assert _metric(report, "EPV")["status"] in {"green", "yellow", "red"}


def test_wykrywa_duplikaty_probek(report):
    # fixture celowo daje część pacjentów z dwiema próbkami
    assert int(_metric(report, "Duplikaty")["value"].replace(" ", "")) > 0


def test_werdykt_i_zalecane_kroki(report):
    summary = report["summary"]
    assert summary["verdict_status"] in {"green", "yellow", "red"}
    assert summary["verdict"]
    assert isinstance(summary["actions"], list)


def test_kazda_metryka_ma_status_i_opis(report):
    for group in report["groups"]:
        for item in group["metrics"]:
            assert item["status"] in {"green", "yellow", "red", "info"}
            assert item["label"] and item["note"]


def test_pca_odporne_na_wartosci_niepoprawne():
    dane = np.array([
        [1.0, -5.0, np.nan],
        [2.0, np.inf, 3.0],
        [0.0, 1.0, 2.0],
        [5.0, 6.0, 7.0],
    ])
    scores, ratio = _pc_scores(dane)
    if scores is not None:
        assert np.isfinite(scores).all()
        assert np.isfinite(ratio).all()


def test_pca_zwraca_none_dla_zbyt_malej_macierzy():
    scores, ratio = _pc_scores(np.array([[1.0, 2.0]]))
    assert scores is None and ratio is None


def test_eta_squared_zakres_i_skrajnosci():
    idealny = _eta_squared([1.0, 1.0, 5.0, 5.0], ["a", "a", "b", "b"])
    assert idealny == pytest.approx(1.0)
    zaden = _eta_squared([1.0, 5.0, 1.0, 5.0], ["a", "a", "a", "a"])
    assert zaden == pytest.approx(0.0)
