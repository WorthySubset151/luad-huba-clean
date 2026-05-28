"""Parser pliku gdc_sample_sheet pobieranego z portalu GDC."""

__author__ = "Łukasz Połaski"

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
]

VALID_TISSUE_TYPES: set[str] = {"Normal", "Tumor"}


class SampleSheetParserError(Exception):
    """Zgłaszany, gdy plik gdc_sample_sheet ma nieprawidłowy format lub nie przejdzie walidacji."""


def parse_sample_sheet(path: str | Path) -> pl.DataFrame:
    """Analizuje plik gdc_sample_sheet pobierany z koszyka portalu GDC."""
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

    unknown_tissues = (
        df.filter(~pl.col("tissue_type").is_in(VALID_TISSUE_TYPES))
        ["tissue_type"].unique().to_list()
    )
    if unknown_tissues:
        raise SampleSheetParserError(
            f"Niedopuszczalne wartości tissue_type w {path.name}: {unknown_tissues}. "
            f"Oczekiwane: {sorted(VALID_TISSUE_TYPES)}"
        )

    return df.select(OUTPUT_COLUMNS)
