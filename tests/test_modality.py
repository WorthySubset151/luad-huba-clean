# -*- coding: utf-8 -*-
"""Abstrakcja modalności: rozpoznawanie cech, rejestr, gotowość na kolejne modalności."""
from __future__ import annotations

import polars as pl
import pytest

from src.modality import (
    DEFAULT_MODALITY,
    REGISTRY,
    RNASEQ,
    Modality,
    detect_modality,
    get_modality,
)

# Modalność wyłącznie testowa — sprawdza, że rdzeń nie jest przywiązany do ENSG.
MIRNA_LIKE = Modality(
    id="mirna-test",
    label="miRNA (modalność testowa)",
    feature_prefix="hsa-mir",
    feature_noun="miRNA",
    feature_noun_plural="miRNA",
    default_metric="RPM",
)


@pytest.fixture
def mixed_frame() -> pl.DataFrame:
    return pl.DataFrame({
        "sample_id": ["TCGA-44-0001-01A"],
        "case_id": ["TCGA-44-0001"],
        "ENSG00000146648": [1.0],
        "ENSG00000133703.5": [2.0],
        "hsa-mir-21": [3.0],
        "hsa-mir-155": [4.0],
    })


def test_domyslna_modalnosc_to_rnaseq():
    assert DEFAULT_MODALITY is RNASEQ
    assert RNASEQ.feature_prefix == "ENSG"


def test_feature_columns_filtruje_po_prefiksie(mixed_frame):
    assert RNASEQ.feature_columns(mixed_frame) == ["ENSG00000146648", "ENSG00000133703.5"]


def test_abstrakcja_dziala_dla_innego_prefiksu(mixed_frame):
    """Sedno refaktoru: rdzeń rozpoznaje cechy przez modalność, nie przez ENSG."""
    assert MIRNA_LIKE.feature_columns(mixed_frame) == ["hsa-mir-21", "hsa-mir-155"]
    assert not set(MIRNA_LIKE.feature_columns(mixed_frame)) & set(
        RNASEQ.feature_columns(mixed_frame)
    )


def test_feature_columns_zachowuje_kolejnosc(survival_ds):
    columns = RNASEQ.feature_columns(survival_ds)
    assert columns == [c for c in survival_ds.columns if c.startswith("ENSG")]


def test_feature_columns_dziala_dla_pandas(mixed_frame):
    assert RNASEQ.feature_columns(mixed_frame.to_pandas()) == [
        "ENSG00000146648", "ENSG00000133703.5",
    ]


def test_has_features(mixed_frame):
    assert RNASEQ.has_features(mixed_frame)
    assert MIRNA_LIKE.has_features(mixed_frame)
    assert not RNASEQ.has_features(pl.DataFrame({"sample_id": ["x"]}))


def test_brak_kolumn_cech_daje_pusta_liste():
    assert RNASEQ.feature_columns(pl.DataFrame({"sample_id": ["x"], "time": [1.0]})) == []


def test_rejestr_zawiera_domyslna_modalnosc():
    assert REGISTRY[RNASEQ.id] is RNASEQ
    assert get_modality("rnaseq") is RNASEQ


def test_nieznana_modalnosc_zglasza_blad():
    with pytest.raises(KeyError):
        get_modality("nie-ma-takiej")


def test_detect_rozpoznaje_zbior_rnaseq(survival_ds):
    assert detect_modality(survival_ds) is RNASEQ


def test_detect_zwraca_none_bez_cech():
    assert detect_modality(pl.DataFrame({"sample_id": ["x"], "time": [1.0]})) is None


def test_modalnosc_jest_niemutowalna():
    with pytest.raises(Exception):
        RNASEQ.feature_prefix = "INNY"


# --- integracja: rdzeń przyjmuje modalność ------------------------------------

def test_readiness_przyjmuje_modalnosc(survival_ds):
    from src.analysis.readiness_report import ml_readiness_report

    report = ml_readiness_report(survival_ds, None, modality=RNASEQ)
    assert report["summary"]["verdict"]


def test_eksport_przyjmuje_modalnosc(survival_ds):
    from src.analysis.ml_export import prepare_ml_dataset

    result = prepare_ml_dataset(survival_ds, top_k=10, test_frac=0.25, seed=1, modality=RNASEQ)
    assert len(result["selected_genes"]) == 10


def test_eksport_bez_cech_modalnosci_zglasza_czytelny_blad(survival_ds):
    from src.analysis.ml_export import prepare_ml_dataset

    with pytest.raises(ValueError) as exc:
        prepare_ml_dataset(survival_ds, top_k=10, modality=MIRNA_LIKE)
    assert "hsa-mir" in str(exc.value)


def test_cohort_summary_przyjmuje_modalnosc(survival_ds):
    from src.analysis.survival_report import cohort_summary

    assert cohort_summary(survival_ds, modality=RNASEQ)["n_samples"] == survival_ds.height
