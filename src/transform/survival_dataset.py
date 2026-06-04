"""Budowanie zbioru do analizy przeżywalności przez integrację ekspresji i danych klinicznych."""

__author__ = "Łukasz Połaski"

import sys

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


def build_survival_dataset(
    expression_matrix: pl.DataFrame,
    sample_sheet: pl.DataFrame,
    clinical: pl.DataFrame,
    tumor_only: bool = True,
    gene_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Integruje macierz ekspresji, arkusz próbek i dane kliniczne w jeden zbiór.

    Produkuje tabelę, w której każdy wiersz to jedna próbka, a kolumny zawierają:
    identyfikatory, dane przeżycia (``time``, ``event``), kowarianty kliniczne
    oraz ekspresję genów. Format jest gotowy do analizy w bibliotece ``lifelines``
    (Kaplan-Meier, model Coxa).

    Argumenty:
        expression_matrix: Macierz z ``build_expression_matrix`` (gene_id x próbki).
        sample_sheet: DataFrame z ``parse_sample_sheet`` (mapowanie próbka -> pacjent).
        clinical: DataFrame z ``parse_clinical`` (czas i status przeżycia per pacjent).
        tumor_only: Jeśli True (domyślnie), zachowuje wyłącznie próbki nowotworowe.
            Próbki tkanki prawidłowej służą tylko do kontroli jakości.
        gene_ids: Opcjonalna lista identyfikatorów genów do uwzględnienia. Jeśli None,
            zachowywane są wszystkie geny z macierzy.

    Zwraca:
        DataFrame: sample_id | case_id | time | event | <kowarianty> | <geny...>

    Zgłasza:
        SurvivalDatasetError: Jeśli brakuje wymaganych kolumn w danych wejściowych,
            wskazane gene_ids nie istnieją w macierzy, po filtrze nowotworowym
            nie pozostała żadna próbka, lub próbka nie ma dopasowania w danych
            klinicznych.
    """
    _validate_inputs(expression_matrix, sample_sheet, clinical)

    matrix = expression_matrix
    if gene_ids is not None:
        missing_genes = set(gene_ids) - set(matrix["gene_id"].to_list())
        if missing_genes:
            raise SurvivalDatasetError(
                f"Geny nieobecne w macierzy: {sorted(missing_genes)[:5]}"
                + ("..." if len(missing_genes) > 5 else "")
            )
        matrix = matrix.filter(pl.col("gene_id").is_in(gene_ids))

    genes = matrix["gene_id"].to_list()
    expr = matrix.drop("gene_id").transpose(
        include_header=True,
        header_name="sample_id",
        column_names=genes,
    )

    sample_meta = sample_sheet.select(["sample_id", "case_id", "tissue_type", "is_tumor"])
    dataset = expr.join(sample_meta, on="sample_id", how="left")

    unmatched_sheet = dataset.filter(pl.col("case_id").is_null())
    if unmatched_sheet.height > 0:
        cases = unmatched_sheet["sample_id"].head(3).to_list()
        raise SurvivalDatasetError(
            f"Próbki bez dopasowania w sample sheet: {cases}"
        )

    if tumor_only:
        dataset = dataset.filter(pl.col("is_tumor"))
        if dataset.height == 0:
            raise SurvivalDatasetError(
                "Po filtrze tumor_only nie pozostała żadna próbka nowotworowa"
            )

    dataset = dataset.drop("is_tumor")

    clinical_cols = ["case_submitter_id"] + CLINICAL_COVARIATES
    clinical_subset = clinical.select(clinical_cols)
    dataset = dataset.join(
        clinical_subset,
        left_on="case_id",
        right_on="case_submitter_id",
        how="left",
    )

    unmatched_clinical = dataset.filter(pl.col("time").is_null())
    if unmatched_clinical.height > 0:
        cases = unmatched_clinical["case_id"].unique().to_list()
        print(
            f"survival_dataset: pominięto {unmatched_clinical.height} próbek "
            f"bez dopasowania w danych klinicznych "
            f"({len(cases)} unikalnych pacjentów, przykłady: {cases[:3]})",
            file=sys.stderr,
        )
        dataset = dataset.filter(pl.col("time").is_not_null())

    if dataset.height == 0:
        raise SurvivalDatasetError(
            "Po filtrze próbek bez dopasowania klinicznego nie pozostała żadna próbka"
        )

    ordered_columns = METADATA_COLUMNS + genes
    return dataset.select(ordered_columns).sort("sample_id")


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
