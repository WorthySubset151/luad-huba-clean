"""Interfejs CLI dla etapów pipeline'u LUAD-HUBA."""

__author__ = "Łukasz Połaski"

from enum import Enum
from pathlib import Path

import typer

from src.ingest.file_naming import STAR_FILE_PATTERNS, STAR_FILE_SUFFIXES

app = typer.Typer(
    help=(
        "Narzędzia CLI dla pipeline'u LUAD-HUBA. "
        "Dostępne komendy: parse-star, build-matrix."
    )
)


class OutputFormat(str, Enum):
    """Dozwolone formaty zapisu wyników parsera STAR."""

    PARQUET = "parquet"
    CSV = "csv"


class ExpressionMetric(str, Enum):
    """Dozwolone metryki ekspresji do agregacji w macierzy."""

    UNSTRANDED = "unstranded"
    STRANDED_FIRST = "stranded_first"
    STRANDED_SECOND = "stranded_second"
    TPM_UNSTRANDED = "tpm_unstranded"
    FPKM_UNSTRANDED = "fpkm_unstranded"
    FPKM_UQ_UNSTRANDED = "fpkm_uq_unstranded"


def _default_raw_dir() -> Path:
    return Path("data/raw")


def _default_interim_dir() -> Path:
    return Path("data/interim/star_counts")


def _default_processed_dir() -> Path:
    return Path("data/processed")


def _discover_star_files(input_dir: Path) -> list[Path]:
    """Wyszukuje rekurencyjnie pliki STAR-Counts dla obu konwencji nazewniczych."""
    found: set[Path] = set()
    for pattern in STAR_FILE_PATTERNS:
        found.update(input_dir.rglob(pattern))
    return sorted(found)


def _output_stem(path: Path) -> str:
    """Buduje nazwę pliku wyjściowego niezależnie od konwencji wejściowej."""
    name = path.name
    for suffix in STAR_FILE_SUFFIXES:
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return path.stem


def _find_sample_sheet(directory: Path) -> Path:
    """Wyszukuje plik gdc_sample_sheet w katalogu."""
    candidates = sorted(directory.glob("gdc_sample_sheet*.tsv"))
    if not candidates:
        raise FileNotFoundError(
            f"Nie znaleziono pliku gdc_sample_sheet*.tsv w {directory}"
        )
    return candidates[0]


@app.callback()
def main() -> None:
    """Punkt wejścia CLI wymagający jawnego wyboru podkomendy."""


if __name__ == "__main__":
    app()
