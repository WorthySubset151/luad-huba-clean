"""Parser pliku gdc_sample_sheet pobieranego z portalu GDC."""

__author__ = "Łukasz Połaski"

import re
from pathlib import Path

import polars as pl

SOURCE_COLUMNS: list[str] = [
    "File ID",
    "File Name",
    "Case ID",
    "Sample ID",
    "Tissue Type",
]

COLUMN_RENAME_MAP: dict[str, str] = {
    "File ID": "file_id",
    "File Name": "file_name",
    "Case ID": "case_id",
    "Sample ID": "sample_id",
    "Tissue Type": "tissue_type",
}

OUTPUT_COLUMNS: list[str] = [
    "file_id",
    "file_name",
    "case_id",
    "sample_id",
    "tissue_type",
    "tcga_sample_code",
    "is_tumor",
]

VALID_TISSUE_TYPES: set[str] = {"Normal", "Tumor"}

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
TCGA_CASE_ID_PATTERN = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$")
TCGA_SAMPLE_ID_PATTERN = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]$")


class SampleSheetParserError(Exception):
    """Zgłaszany, gdy plik gdc_sample_sheet ma nieprawidłowy format lub nie przejdzie walidacji."""


def parse_sample_sheet(path: str | Path) -> pl.DataFrame:
    """Analizuje plik gdc_sample_sheet pobierany z koszyka portalu GDC.

    Plik zawiera mapowanie identyfikatorów plików pobranych z GDC na metadane
    próbek (pacjent TCGA, typ tkanki, kod próbki). Parser wzbogaca dane
    o dwie kolumny pochodne: ``tcga_sample_code`` (np. ``11A``) oraz
    flagę ``is_tumor`` wyznaczaną na podstawie kolumny ``Tissue Type``.

    Argumenty:
        path: Ścieżka do pliku ``gdc_sample_sheet*.tsv``.

    Zwraca:
        Obiekt polars DataFrame z kolumnami OUTPUT_COLUMNS.

    Zgłasza:
        FileNotFoundError: Jeśli plik nie istnieje.
        SampleSheetParserError: Jeśli plik nie zawiera wymaganych kolumn,
            zawiera nieprawidłowe identyfikatory UUID lub TCGA, zawiera
            niedopuszczalne wartości w kolumnie Tissue Type, lub zawiera
            zduplikowane identyfikatory plików.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku gdc_sample_sheet: {path}")

    df = pl.read_csv(
        path,
        separator="\t",
        has_header=True,
        null_values=["", "NA"],
        infer_schema_length=10000,
    )

    if df.height == 0:
        raise SampleSheetParserError(f"Plik {path.name} nie zawiera żadnych wierszy danych")

    missing = set(SOURCE_COLUMNS) - set(df.columns)
    if missing:
        raise SampleSheetParserError(
            f"Brak wymaganych kolumn w {path.name}: {sorted(missing)}"
        )

    df = df.select(SOURCE_COLUMNS).rename(COLUMN_RENAME_MAP)

    invalid_uuid = df.filter(~pl.col("file_id").str.contains(UUID_PATTERN.pattern))
    if invalid_uuid.height > 0:
        sample = invalid_uuid["file_id"].head(3).to_list()
        raise SampleSheetParserError(
            f"Nieprawidłowy format UUID w kolumnie file_id w {path.name}: {sample}"
        )

    invalid_case = df.filter(~pl.col("case_id").str.contains(TCGA_CASE_ID_PATTERN.pattern))
    if invalid_case.height > 0:
        sample = invalid_case["case_id"].head(3).to_list()
        raise SampleSheetParserError(
            f"Nieprawidłowy format TCGA Case ID w {path.name}: {sample}"
        )

    invalid_sample = df.filter(~pl.col("sample_id").str.contains(TCGA_SAMPLE_ID_PATTERN.pattern))
    if invalid_sample.height > 0:
        sample = invalid_sample["sample_id"].head(3).to_list()
        raise SampleSheetParserError(
            f"Nieprawidłowy format TCGA Sample ID w {path.name}: {sample}"
        )

    unknown_tissues = (
        df.filter(~pl.col("tissue_type").is_in(VALID_TISSUE_TYPES))
        ["tissue_type"].unique().to_list()
    )
    if unknown_tissues:
        raise SampleSheetParserError(
            f"Niedopuszczalne wartości tissue_type w {path.name}: {unknown_tissues}. "
            f"Oczekiwane: {sorted(VALID_TISSUE_TYPES)}"
        )

    if df["file_id"].n_unique() != df.height:
        raise SampleSheetParserError(f"Znaleziono zduplikowane file_id w {path.name}")

    df = df.with_columns(
        [
            pl.col("sample_id").str.slice(-3).alias("tcga_sample_code"),
            (pl.col("tissue_type") == "Tumor").alias("is_tumor"),
        ]
    ).select(OUTPUT_COLUMNS)

    return df
