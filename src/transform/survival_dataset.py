"""Budowanie zbioru do analizy przeżywalności przez integrację ekspresji i danych klinicznych."""

__author__ = "Łukasz Połaski"

import polars as pl

METADATA_COLUMNS: list[str] = [
    "sample_id",
    "case_id",
    "time",
    "event",
    "age_at_index",
    "gender",
    "ajcc_pathologic_stage",
    "tissue_type",
]

CLINICAL_COVARIATES: list[str] = [
    "time",
    "event",
    "age_at_index",
    "gender",
    "ajcc_pathologic_stage",
]


class SurvivalDatasetError(Exception):
    """Zgłaszany, gdy integracja danych do zbioru przeżywalności nie może się powieść."""


def _validate_inputs(
    expression_matrix: pl.DataFrame,
    sample_sheet: pl.DataFrame,
    clinical: pl.DataFrame,
) -> None:
    """Waliduje obecność wymaganych kolumn we wszystkich danych wejściowych."""
    if "gene_id" not in expression_matrix.columns:
        raise SurvivalDatasetError("Macierz ekspresji nie zawiera kolumny gene_id")

    if expression_matrix.width < 2:
        raise SurvivalDatasetError("Macierz ekspresji nie zawiera żadnych próbek")

    sheet_required = {"sample_id", "case_id", "tissue_type", "is_tumor"}
    sheet_missing = sheet_required - set(sample_sheet.columns)
    if sheet_missing:
        raise SurvivalDatasetError(
            f"Sample sheet nie zawiera kolumn: {sorted(sheet_missing)}"
        )

    clinical_required = {"case_submitter_id"} | set(CLINICAL_COVARIATES)
    clinical_missing = clinical_required - set(clinical.columns)
    if clinical_missing:
        raise SurvivalDatasetError(
            f"Dane kliniczne nie zawierają kolumn: {sorted(clinical_missing)}"
        )
