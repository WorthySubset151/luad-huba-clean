# -*- coding: utf-8 -*-
"""Parser clinical.tsv: filtrowanie diagnoz, wyliczanie przeżycia, markery braków GDC."""
from __future__ import annotations

import polars as pl
import pytest

from src.ingest.clinical_parser import ClinicalParserError, parse_clinical

COVARIATES = ["age_at_index", "gender", "ajcc_pathologic_stage"]


def test_zwraca_jeden_wiersz_na_pacjenta(clinical):
    assert clinical.height > 0
    assert clinical["case_submitter_id"].n_unique() == clinical.height


def test_odfiltrowuje_diagnozy_niepodstawowe(clinical, clinical_tsv_path):
    raw = pl.read_csv(clinical_tsv_path, separator="\t", infer_schema_length=0)
    primary = raw.filter(pl.col("diagnoses.diagnosis_is_primary_disease") == "true")
    # plik jest zdenormalizowany: wierszy jest więcej niż pacjentów,
    # a parser ma zejść do liczby diagnoz podstawowych
    assert raw.height > primary.height
    assert clinical.height <= primary.height


def test_kowarianty_kliniczne_sa_kompletne(clinical):
    for col in COVARIATES:
        filled = int(clinical[col].is_not_null().sum())
        assert filled / clinical.height > 0.9, f"{col} wypełnione tylko w {filled}/{clinical.height}"


def test_markery_brakow_gdc_daja_null_nie_tekst(clinical):
    # "'--" i "not reported" nie mogą przeciekać jako wartości tekstowe
    for col in ("gender", "ajcc_pathologic_stage", "race"):
        values = set(clinical[col].drop_nulls().to_list())
        assert not values & {"'--", "--", "not reported", "Not Reported", "NA", ""}


def test_kolumny_wyjsciowe_i_typy(clinical):
    for col in ("case_submitter_id", "vital_status", "time", "event", *COVARIATES):
        assert col in clinical.columns
    assert clinical["age_at_index"].dtype in (pl.Int64, pl.Int32)


def test_event_zgodny_z_vital_status(clinical):
    dead = clinical.filter(pl.col("vital_status") == "Dead")
    alive = clinical.filter(pl.col("vital_status") == "Alive")
    if dead.height:
        assert set(dead["event"].to_list()) == {1}
    if alive.height:
        assert set(alive["event"].to_list()) == {0}


def test_czas_przezycia_nieujemny(clinical):
    assert clinical["time"].min() >= 0
    assert clinical["time"].null_count() == 0


def test_brak_pliku_zglasza_blad():
    with pytest.raises((ClinicalParserError, FileNotFoundError)):
        parse_clinical("/nie/ma/takiego/clinical.tsv")


def test_plik_bez_wymaganych_kolumn_zglasza_blad(tmp_path):
    bad = tmp_path / "clinical.tsv"
    bad.write_text("kolumna_a\tkolumna_b\n1\t2\n", encoding="utf-8")
    with pytest.raises(ClinicalParserError):
        parse_clinical(bad)
