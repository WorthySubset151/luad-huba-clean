"""Runner warstwy validate - spina reguły kontroli jakości w jeden raport."""

__author__ = "Łukasz Połaski"

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from src.validate.cohort_checks import (
    check_cases_have_clinical,
    check_duplicate_samples,
    check_orphan_star_files,
    check_samples_have_star_files,
)
from src.validate.qc_result import QCReport

QC_LOG_FILENAME_TEMPLATE = "qc_report_{timestamp}.json"


def run_cohort_qc(
    sample_sheet: pl.DataFrame,
    clinical: pl.DataFrame,
    available_stems: set[str],
) -> QCReport:
    """Uruchamia wszystkie reguły walidacji kohorty i agreguje wyniki.

    Argumenty:
        sample_sheet: DataFrame z ``parse_sample_sheet``.
        clinical: DataFrame z ``parse_clinical``.
        available_stems: Zbiór stemów plików dostępnych po etapie ingest
            (np. z ``data/interim/star_counts/``).

    Zwraca:
        Raport ``QCReport`` ze wszystkimi wykrytymi problemami.
    """
    report = QCReport()
    report.extend(check_samples_have_star_files(sample_sheet, available_stems))
    report.extend(check_orphan_star_files(sample_sheet, available_stems))
    report.extend(check_cases_have_clinical(sample_sheet, clinical))
    report.extend(check_duplicate_samples(sample_sheet))
    return report


def discover_stems(directory: Path, suffix: str = ".parquet") -> set[str]:
    """Skanuje katalog w poszukiwaniu plików o danym rozszerzeniu i zwraca ich stemy.

    Argumenty:
        directory: Katalog do przeskanowania (np. ``data/interim/star_counts/``).
        suffix: Rozszerzenie plików (domyślnie ``.parquet``).

    Zwraca:
        Zbiór stemów (nazw bez rozszerzenia) znalezionych plików.
        Pusty zbiór, jeśli katalog nie istnieje.
    """
    if not directory.exists():
        return set()
    return {p.stem for p in directory.glob(f"*{suffix}")}


def save_qc_report(
    report: QCReport,
    output_dir: Path,
    filename: str | None = None,
) -> Path:
    """Zapisuje raport QC jako sformatowany JSON i zwraca ścieżkę zapisanego pliku.

    Argumenty:
        report: Raport do zapisu.
        output_dir: Katalog wyjściowy (zostanie utworzony, jeśli nie istnieje).
        filename: Opcjonalna nazwa pliku. Jeśli pominięta, generowana jest
            nazwa ze stemplem czasowym UTC.

    Zwraca:
        Ścieżka do zapisanego pliku JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = QC_LOG_FILENAME_TEMPLATE.format(timestamp=timestamp)
    out_path = output_dir / filename
    out_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path
