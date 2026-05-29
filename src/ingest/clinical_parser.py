"""Parser pliku clinical.tsv pobieranego z portalu GDC dla projektów TCGA."""

__author__ = "Łukasz Połaski"

import re
from pathlib import Path

import polars as pl

SOURCE_COLUMNS: list[str] = [
    "cases.submitter_id",
    "cases.primary_site",
    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.age_at_index",
    "demographic.gender",
    "demographic.race",
    "diagnoses.diagnosis_is_primary_disease",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.ajcc_pathologic_stage",
]

COLUMN_RENAME_MAP: dict[str, str] = {
    "cases.submitter_id": "case_submitter_id",
    "cases.primary_site": "primary_site",
    "demographic.vital_status": "vital_status",
    "demographic.days_to_death": "days_to_death",
    "demographic.age_at_index": "age_at_index",
    "demographic.gender": "gender",
    "demographic.race": "race",
    "diagnoses.days_to_last_follow_up": "days_to_last_follow_up",
    "diagnoses.ajcc_pathologic_stage": "ajcc_pathologic_stage",
}

OUTPUT_COLUMNS: list[str] = [
    "case_submitter_id",
    "vital_status",
    "days_to_death",
    "days_to_last_follow_up",
    "age_at_index",
    "gender",
    "race",
    "ajcc_pathologic_stage",
    "primary_site",
]

VALID_VITAL_STATUSES: set[str] = {"Alive", "Dead"}
GDC_NULL_VALUES: list[str] = ["", "NA", "'--", "--", "not reported", "Not Reported"]
TCGA_CASE_ID_PATTERN = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$")


class ClinicalParserError(Exception):
    """Zgłaszany, gdy plik clinical.tsv ma nieprawidłowy format lub nie przejdzie walidacji."""


def parse_clinical(path: str | Path) -> pl.DataFrame:
    """Analizuje plik clinical.tsv z koszyka portalu GDC.

    Filtruje do diagnoz podstawowych i deduplikuje do jednego wiersza na pacjenta.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku clinical: {path}")

    df = pl.read_csv(
        path,
        separator="\t",
        has_header=True,
        null_values=GDC_NULL_VALUES,
        infer_schema_length=0,
    )

    missing = set(SOURCE_COLUMNS) - set(df.columns)
    if missing:
        raise ClinicalParserError(
            f"Brak wymaganych kolumn w {path.name}: {sorted(missing)}"
        )

    df = (
        df.select(SOURCE_COLUMNS)
        .filter(pl.col("diagnoses.diagnosis_is_primary_disease") == "true")
        .drop("diagnoses.diagnosis_is_primary_disease")
        .rename(COLUMN_RENAME_MAP)
        .unique()
        .group_by("case_submitter_id")
        .first()
    )

    if df.height == 0:
        raise ClinicalParserError(
            f"Po filtrowaniu diagnoses.diagnosis_is_primary_disease w {path.name} "
            f"nie pozostał żaden wiersz"
        )

    df = df.with_columns(
        [
            pl.col("days_to_death").cast(pl.Float64, strict=False).cast(pl.Int64, strict=False),
            pl.col("days_to_last_follow_up").cast(pl.Float64, strict=False).cast(pl.Int64, strict=False),
            pl.col("age_at_index").cast(pl.Int64, strict=False),
        ]
    )

    invalid_case = df.filter(~pl.col("case_submitter_id").str.contains(TCGA_CASE_ID_PATTERN.pattern))
    if invalid_case.height > 0:
        sample = invalid_case["case_submitter_id"].head(3).to_list()
        raise ClinicalParserError(
            f"Nieprawidłowy format TCGA Case ID w {path.name}: {sample}"
        )

    unknown_vital = (
        df.filter(~pl.col("vital_status").is_in(VALID_VITAL_STATUSES))
        ["vital_status"].drop_nulls().unique().to_list()
    )
    if unknown_vital:
        raise ClinicalParserError(
            f"Nieznane wartości vital_status w {path.name}: {unknown_vital}. "
            f"Oczekiwane: {sorted(VALID_VITAL_STATUSES)}"
        )

    return df.select(OUTPUT_COLUMNS).sort("case_submitter_id")
