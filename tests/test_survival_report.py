# -*- coding: utf-8 -*-
"""Raporty przeżycia zasilające dashboard: sygnatura, KM pojedynczy/wielogenowy, Cox.

Te funkcje nie miały pokrycia — refaktor modalności wstawił do nich odwołanie do
nieistniejącej zmiennej ``modality`` (NameError w GUI), a suite tego nie wykrył.
Testy wywołują każdą z nich na syntetycznym zbiorze, żeby taki błąd padał tutaj,
nie na wykresie u użytkownika.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.survival_report import (
    SIGNATURE_PANEL,
    cohort_summary,
    cox_clinical_report,
    cox_genes_report,
    multi_gene_km_report,
    signature_score,
    single_gene_km_report,
)
from src.modality import RNASEQ


# Realne identyfikatory z panelu LUAD, obecne w fixture.
NKX2_1 = ("NKX2-1", "ENSG00000136352")
MKI67 = ("MKI67", "ENSG00000148773")
BIRC5 = ("BIRC5", "ENSG00000089685")


def test_signature_score_zwraca_wektor_na_probke(survival_ds):
    score, n_genes = signature_score(survival_ds)
    assert len(score) == survival_ds.height
    assert n_genes > 0
    assert np.isfinite(score).all()


def test_cohort_summary_zlicza_probki_i_zdarzenia(survival_ds, gene_columns):
    summary = cohort_summary(survival_ds)
    assert summary["n_samples"] == survival_ds.height
    assert summary["n_genes"] == len(gene_columns)
    assert 0 <= summary["n_events"] <= survival_ds.height


def test_cohort_summary_z_jawna_modalnoscia(survival_ds):
    assert cohort_summary(survival_ds, modality=RNASEQ)["n_samples"] == survival_ds.height


def test_single_gene_km_report_liczy_log_rank(survival_ds):
    report = single_gene_km_report(survival_ds, NKX2_1[1], NKX2_1[0])
    assert report["symbol"] == NKX2_1[0]
    assert "error" not in report
    assert report["logrank_p"] is None or 0.0 <= report["logrank_p"] <= 1.0


def test_multi_gene_km_report_po_jednym_na_gen(survival_ds):
    genes = [NKX2_1, MKI67, BIRC5]
    reports = multi_gene_km_report(survival_ds, genes)
    assert len(reports) == len(genes)
    for r in reports:
        assert "symbol" in r and "p_value" in r


def test_cox_genes_report_porownuje_c_index(survival_ds):
    report = cox_genes_report(survival_ds)
    assert "error" not in report, report
    # panel dokłada wartość ponad klinikę => raport niesie oba C-index i deltę
    assert "c_index_clinical" in report
    assert "c_index_genes" in report
    assert "delta" in report
    assert 0.0 <= report["c_index_genes"] <= 1.0


def test_cox_clinical_report_dziala(survival_ds):
    report = cox_clinical_report(survival_ds)
    assert "error" not in report, report
    assert "c_index" in report
    assert 0.0 <= report["c_index"] <= 1.0


def test_funkcje_panelowe_uzywaja_panelu_rnaseq(survival_ds):
    # sygnatura opiera się na panelu genów LUAD, nie na dowolnej modalności
    _, n_genes = signature_score(survival_ds)
    assert n_genes <= len(SIGNATURE_PANEL)


def test_single_gene_zglasza_brak_genu_bez_wyjatku(survival_ds):
    # gen spoza macierzy nie może wywalać raportu — ma wrócić z polem error
    report = single_gene_km_report(survival_ds, "ENSG99999999999", "NIE-MA")
    assert "error" in report and report["symbol"] == "NIE-MA"


def test_gen_o_stalej_ekspresji_nie_wywala_km(survival_ds, gene_columns):
    """Gen o niemal stałej ekspresji (jak ALK, bliski zeru w LUAD) degeneruje podział
    high/low względem mediany — jedna grupa bywa pusta. KM ma to znieść, nie crashować."""
    import polars as pl

    target = gene_columns[0]
    flat = survival_ds.with_columns(pl.lit(0.0).alias(target))
    report = single_gene_km_report(flat, target, "STAŁY")
    assert "symbol" in report  # brak wyjątku — raport się zwraca
