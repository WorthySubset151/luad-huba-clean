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


def build_survival_dataset(
    expression_matrix: pl.DataFrame,
    sample_sheet: pl.DataFrame,
    clinical: pl.DataFrame,
    tumor_only: bool = True,
    gene_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Integruje macierz ekspresji, arkusz próbek i dane kliniczne w jeden zbiór."""
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

    return dataset


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
