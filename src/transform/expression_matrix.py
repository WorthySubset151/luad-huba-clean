"""Budowanie macierzy ekspresji genów ze sparsowanych plików STAR-Counts."""

__author__ = "Łukasz Połaski"

from pathlib import Path

import polars as pl

ALLOWED_METRICS: set[str] = {
    "unstranded",
    "stranded_first",
    "stranded_second",
    "tpm_unstranded",
    "fpkm_unstranded",
    "fpkm_uq_unstranded",
}

SAMPLE_SHEET_REQUIRED_COLUMNS: set[str] = {"file_name", "sample_id"}


class ExpressionMatrixError(Exception):
    """Zgłaszany, gdy budowanie macierzy ekspresji nie może się powieść."""


def _read_and_validate_parquet(path: Path, metric: str) -> pl.DataFrame:
    """Wczytuje plik Parquet i waliduje obecność wymaganych kolumn."""
    if not path.exists():
        raise ExpressionMatrixError(f"Plik Parquet nie istnieje: {path}")

    df = pl.read_parquet(path, columns=["gene_id", metric])

    required = {"gene_id", metric}
    missing = required - set(df.columns)
    if missing:
        raise ExpressionMatrixError(
            f"Plik {path.name} nie zawiera kolumn: {sorted(missing)}"
        )

    return df
