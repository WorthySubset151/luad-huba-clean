"""Reguły walidacji spójności kohorty dla pipeline'u LUAD-HUBA.

Każda funkcja jest czystą regułą: przyjmuje dane i zwraca listę wykrytych
problemów (``QCIssue``), bez efektów ubocznych ani operacji wejścia/wyjścia.
Orchestracją i logowaniem zajmuje się osobny moduł (runner).
"""

__author__ = "Łukasz Połaski"

import polars as pl

from src.ingest.file_naming import extract_star_file_stem
from src.validate.qc_result import QCCategory, QCIssue, Severity


def check_samples_have_star_files(
    sample_sheet: pl.DataFrame,
    available_stems: set[str],
) -> list[QCIssue]:
    """Sprawdza, czy każda próbka z arkusza ma odpowiadający plik danych.

    Próbka zadeklarowana w sample sheet, dla której brakuje pliku STAR
    (lub przetworzonego Parquet) nie może zostać użyta w analizie - błąd.

    Argumenty:
        sample_sheet: DataFrame z ``parse_sample_sheet`` (kolumny: file_name, sample_id).
        available_stems: Zbiór stemów plików dostępnych na dysku.

    Zwraca:
        Lista problemów o kategorii MISSING_STAR_FILE i istotności ERROR.
    """
    issues: list[QCIssue] = []
    for row in sample_sheet.select(["file_name", "sample_id"]).iter_rows(named=True):
        stem = extract_star_file_stem(row["file_name"])
        if stem not in available_stems:
            issues.append(
                QCIssue(
                    severity=Severity.ERROR,
                    category=QCCategory.MISSING_STAR_FILE,
                    message=f"Próbka {row['sample_id']} nie ma pliku danych (stem: {stem})",
                    context={"sample_id": row["sample_id"], "file_stem": stem},
                )
            )
    return issues


def check_orphan_star_files(
    sample_sheet: pl.DataFrame,
    available_stems: set[str],
) -> list[QCIssue]:
    """Sprawdza, czy nie ma plików danych bez wpisu w arkuszu próbek.

    Plik obecny na dysku, ale niewystępujący w sample sheet, jest sierotą -
    nie wiadomo, do której próbki należy. Ostrzeżenie (pomijamy go w analizie).

    Argumenty:
        sample_sheet: DataFrame z ``parse_sample_sheet``.
        available_stems: Zbiór stemów plików dostępnych na dysku.

    Zwraca:
        Lista problemów o kategorii ORPHAN_STAR_FILE i istotności WARNING.
    """
    expected_stems = {
        extract_star_file_stem(name)
        for name in sample_sheet["file_name"].to_list()
    }
    orphans = available_stems - expected_stems

    return [
        QCIssue(
            severity=Severity.WARNING,
            category=QCCategory.ORPHAN_STAR_FILE,
            message=f"Plik danych bez wpisu w sample sheet (stem: {stem})",
            context={"file_stem": stem},
        )
        for stem in sorted(orphans)
    ]


def check_cases_have_clinical(
    sample_sheet: pl.DataFrame,
    clinical: pl.DataFrame,
) -> list[QCIssue]:
    """Sprawdza, czy każdy pacjent z arkusza ma dane kliniczne.

    Próbka, której pacjent nie występuje w danych klinicznych, nie ma czasu
    przeżycia ani statusu - jest bezużyteczna do analizy przeżywalności. Błąd.

    Argumenty:
        sample_sheet: DataFrame z ``parse_sample_sheet`` (kolumny: case_id, sample_id).
        clinical: DataFrame z ``parse_clinical`` (kolumna: case_submitter_id).

    Zwraca:
        Lista problemów o kategorii MISSING_CLINICAL i istotności ERROR.
    """
    clinical_cases = set(clinical["case_submitter_id"].to_list())

    issues: list[QCIssue] = []
    for row in sample_sheet.select(["case_id", "sample_id"]).iter_rows(named=True):
        if row["case_id"] not in clinical_cases:
            issues.append(
                QCIssue(
                    severity=Severity.ERROR,
                    category=QCCategory.MISSING_CLINICAL,
                    message=(
                        f"Pacjent {row['case_id']} (próbka {row['sample_id']}) "
                        f"nie ma danych klinicznych"
                    ),
                    context={"case_id": row["case_id"], "sample_id": row["sample_id"]},
                )
            )
    return issues


def check_duplicate_samples(sample_sheet: pl.DataFrame) -> list[QCIssue]:
    """Sprawdza, czy ten sam identyfikator próbki nie występuje wielokrotnie.

    W TCGA występowanie tej samej próbki w wielu plikach jest scenariuszem
    legalnym (różne aliquoty, ponowna analiza), obsługiwanym przez parametr
    ``duplicate_strategy`` w ``build_expression_matrix``. Ostrzeżenie, nie błąd.

    Argumenty:
        sample_sheet: DataFrame z ``parse_sample_sheet`` (kolumna: sample_id).

    Zwraca:
        Lista problemów o kategorii DUPLICATE_SAMPLE i istotności WARNING.
    """
    counts = (
        sample_sheet.group_by("sample_id")
        .len()
        .filter(pl.col("len") > 1)
        .sort("sample_id")
    )

    return [
        QCIssue(
            severity=Severity.WARNING,
            category=QCCategory.DUPLICATE_SAMPLE,
            message=f"Próbka {row['sample_id']} występuje {row['len']} razy",
            context={"sample_id": row["sample_id"], "count": row["len"]},
        )
        for row in counts.iter_rows(named=True)
    ]
