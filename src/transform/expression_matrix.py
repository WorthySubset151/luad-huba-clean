"""Budowanie macierzy ekspresji genów ze sparsowanych plików STAR-Counts."""

__author__ = "Łukasz Połaski"

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from src.ingest.file_naming import extract_star_file_stem

ALLOWED_METRICS: set[str] = {
    "unstranded",
    "stranded_first",
    "stranded_second",
    "tpm_unstranded",
    "fpkm_unstranded",
    "fpkm_uq_unstranded",
}

VALID_DUPLICATE_STRATEGIES: set[str] = {"fail", "deepest", "first"}

SAMPLE_SHEET_REQUIRED_COLUMNS: set[str] = {"file_name", "sample_id"}


class ExpressionMatrixError(Exception):
    """Zgłaszany, gdy budowanie macierzy ekspresji nie może się powieść."""


def build_expression_matrix(
    parquet_paths: list[Path],
    sample_sheet: pl.DataFrame,
    metric: str = "unstranded",
    duplicate_strategy: str = "fail",
) -> pl.DataFrame:
    """Łączy pliki Parquet z parsowanych STAR-Counts w jedną macierz ekspresji.

    Każdy plik Parquet zawiera dane jednej próbki (60 660 genów x 9 kolumn).
    Funkcja wyciąga z każdej próbki wybraną metrykę i tworzy macierz
    o wymiarach (n_genów x n_próbek+1), gdzie pierwsza kolumna to ``gene_id``,
    a kolejne kolumny noszą nazwy ``sample_id`` z arkusza próbek.

    Argumenty:
        parquet_paths: Lista ścieżek do plików Parquet z ``data/interim/star_counts/``.
        sample_sheet: DataFrame z parserem ``parse_sample_sheet``. Musi zawierać
            kolumny ``file_name`` i ``sample_id``.
        metric: Nazwa kolumny ekspresji do wyciągnięcia. Dozwolone wartości
            w ``ALLOWED_METRICS``.
        duplicate_strategy: Strategia obsługi duplikatów sample_id (jeden sample_id
            mający kilka plików, np. wielokrotne aliquoty w TCGA):

            - ``fail`` (domyślnie): rzuca wyjątek przy wykryciu duplikatów
            - ``deepest``: wybiera plik z największą sumą wartości metryki
            - ``first``: wybiera pierwszy plik alfabetycznie po ścieżce

    Zwraca:
        DataFrame o strukturze:
            gene_id | <sample_id_1> | <sample_id_2> | ...

    Zgłasza:
        ExpressionMatrixError: Jeśli lista plików jest pusta, metric jest
            nieobsługiwana, sample_sheet nie ma wymaganych kolumn, brakuje
            mapowania file -> sample_id, kolejność/zawartość gene_id różni się
            między plikami, lub wynik ma duplikaty sample_id (przy strategy=fail).
    """
    if not parquet_paths:
        raise ExpressionMatrixError("Lista plików parquet jest pusta")

    if metric not in ALLOWED_METRICS:
        raise ExpressionMatrixError(
            f"Niedozwolona metryka: {metric!r}. Dozwolone: {sorted(ALLOWED_METRICS)}"
        )

    if duplicate_strategy not in VALID_DUPLICATE_STRATEGIES:
        raise ExpressionMatrixError(
            f"Niedozwolona strategia deduplikacji: {duplicate_strategy!r}. "
            f"Dozwolone: {sorted(VALID_DUPLICATE_STRATEGIES)}"
        )

    missing_cols = SAMPLE_SHEET_REQUIRED_COLUMNS - set(sample_sheet.columns)
    if missing_cols:
        raise ExpressionMatrixError(
            f"Sample sheet nie zawiera wymaganych kolumn: {sorted(missing_cols)}"
        )

    stem_to_sample = _build_stem_to_sample_map(sample_sheet)

    sample_ids: list[str] = []
    for path in parquet_paths:
        stem = path.stem
        if stem not in stem_to_sample:
            raise ExpressionMatrixError(
                f"Brak mapowania w sample sheet dla pliku {path.name} (stem: {stem!r})"
            )
        sample_ids.append(stem_to_sample[stem])

    if len(set(sample_ids)) != len(sample_ids):
        if duplicate_strategy == "fail":
            duplicates = sorted({s for s in sample_ids if sample_ids.count(s) > 1})
            raise ExpressionMatrixError(
                f"Duplikaty sample_id w wyniku: {duplicates}"
            )
        parquet_paths, sample_ids = _deduplicate(
            parquet_paths, sample_ids, strategy=duplicate_strategy, metric=metric
        )

    first_df = _read_and_validate_parquet(parquet_paths[0], metric)
    reference_genes = first_df["gene_id"]
    matrix = first_df.rename({metric: sample_ids[0]})

    for path, sample_id in zip(parquet_paths[1:], sample_ids[1:]):
        df = _read_and_validate_parquet(path, metric)
        if not df["gene_id"].equals(reference_genes):
            raise ExpressionMatrixError(
                f"Plik {path.name} ma inne gene_id niż {parquet_paths[0].name} "
                f"(różna kolejność lub zawartość genów)"
            )
        matrix = matrix.with_columns(df[metric].alias(sample_id))

    null_columns = [c for c in matrix.columns if matrix[c].null_count() > 0]
    if null_columns:
        raise ExpressionMatrixError(
            f"Macierz zawiera wartości null w kolumnach: {null_columns}"
        )

    return matrix


def build_manifest(
    matrix: pl.DataFrame,
    parquet_paths: list[Path],
    metric: str,
) -> dict:
    """Reekspr z src.export.manifest dla backward compatibility."""
    from src.export.manifest import build_manifest as _impl
    return _impl(matrix, parquet_paths, metric)


def _build_stem_to_sample_map(sample_sheet: pl.DataFrame) -> dict[str, str]:
    """Buduje mapowanie stem_pliku -> sample_id z arkusza próbek."""
    mapping: dict[str, str] = {}
    for row in sample_sheet.select(["file_name", "sample_id"]).iter_rows():
        file_name, sample_id = row
        stem = extract_star_file_stem(file_name)
        mapping[stem] = sample_id
    return mapping


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


def save_manifest(manifest: dict, output_path: Path) -> None:
    """Reekspr z src.export.manifest dla backward compatibility."""
    from src.export.manifest import save_manifest as _impl
    _impl(manifest, output_path)


def _deduplicate(
    paths: list[Path],
    sample_ids: list[str],
    strategy: str,
    metric: str,
) -> tuple[list[Path], list[str]]:
    """Wybiera jeden plik per sample_id zgodnie ze strategią.

    - ``deepest``: dla każdego sample_id wybiera plik z największą sumą metryki
    - ``first``: dla każdego sample_id wybiera plik z najmniejszą ścieżką (sort alfabetyczny)
    """
    groups: dict[str, list[Path]] = {}
    for path, sid in zip(paths, sample_ids):
        groups.setdefault(sid, []).append(path)

    chosen_paths: list[Path] = []
    chosen_sample_ids: list[str] = []
    for sid, group in groups.items():
        if len(group) == 1:
            chosen_paths.append(group[0])
        elif strategy == "first":
            chosen_paths.append(sorted(group)[0])
        elif strategy == "deepest":
            chosen_paths.append(max(group, key=lambda p: _read_metric_sum(p, metric)))
        chosen_sample_ids.append(sid)

    return chosen_paths, chosen_sample_ids


def _read_metric_sum(path: Path, metric: str) -> float:
    """Wczytuje tylko kolumnę metryki z pliku Parquet i zwraca jej sumę."""
    df = pl.read_parquet(path, columns=[metric])
    return float(df[metric].sum())
