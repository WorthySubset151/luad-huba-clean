"""Interfejs CLI dla etapów pipeline'u LUAD-HUBA."""

__author__ = "Łukasz Połaski"

from enum import Enum
from pathlib import Path

import typer

from src.ingest.file_naming import STAR_FILE_PATTERNS, STAR_FILE_SUFFIXES
from src.ingest.sample_sheet_parser import SampleSheetParserError, parse_sample_sheet
from src.ingest.star_parser import StarParserError, parse_star_counts
from src.transform.expression_matrix import (
    ALLOWED_METRICS,
    ExpressionMatrixError,
    build_expression_matrix,
    build_manifest,
    save_manifest,
)

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


@app.command("parse-star")
def parse_star(
    input_dir: Path = typer.Option(
        _default_raw_dir(),
        "--input-dir",
        help="Katalog wejściowy z pobranymi plikami GDC STAR-Counts.",
    ),
    output_dir: Path = typer.Option(
        _default_interim_dir(),
        "--output-dir",
        help="Katalog wyjściowy na przetworzone pliki po parsowaniu.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.PARQUET,
        "--format",
        help="Format zapisu wynikowego dla każdego pliku wejściowego.",
    ),
) -> None:
    """Parsuje wszystkie pliki STAR-Counts i zapisuje je do katalogu wyjściowego."""
    if not input_dir.exists():
        typer.secho(f"Nie znaleziono katalogu wejściowego: {input_dir}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    star_files = _discover_star_files(input_dir)
    if not star_files:
        typer.secho(
            f"Nie znaleziono plików STAR-Counts w {input_dir}. "
            f"Oczekiwane wzorce: {', '.join(STAR_FILE_PATTERNS)}",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Znaleziono {len(star_files)} plik(i) STAR-Counts do przetworzenia.")

    processed = 0
    for path in star_files:
        try:
            df = parse_star_counts(path)
        except (StarParserError, FileNotFoundError) as exc:
            typer.secho(f"Błąd parsowania {path}: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc

        stem = _output_stem(path)
        if output_format == OutputFormat.PARQUET:
            out_path = output_dir / f"{stem}.parquet"
            df.write_parquet(out_path)
        else:
            out_path = output_dir / f"{stem}.csv"
            df.write_csv(out_path)

        processed += 1
        typer.secho(
            f"[{processed}/{len(star_files)}] Zapisano: {out_path}",
            fg=typer.colors.GREEN,
        )

    typer.secho(
        f"Zakończono. Przetworzono {processed} plik(i). Wyniki w: {output_dir}",
        fg=typer.colors.BRIGHT_GREEN,
    )


@app.command("build-matrix")
def build_matrix(
    input_dir: Path = typer.Option(
        _default_interim_dir(),
        "--input-dir",
        help="Katalog z plikami Parquet z etapu parse-star.",
    ),
    sample_sheet: Path = typer.Option(
        None,
        "--sample-sheet",
        help="Ścieżka do pliku gdc_sample_sheet. Jeśli pominięta, plik jest "
             "wyszukiwany automatycznie w data/raw/.",
    ),
    output_dir: Path = typer.Option(
        _default_processed_dir(),
        "--output-dir",
        help="Katalog wyjściowy na macierz ekspresji i manifest.",
    ),
    metric: ExpressionMetric = typer.Option(
        ExpressionMetric.UNSTRANDED,
        "--metric",
        help="Metryka ekspresji do wyciągnięcia z plików parquet.",
    ),
) -> None:
    """Buduje macierz ekspresji genów z plików Parquet po parse-star."""
    if not input_dir.exists():
        typer.secho(f"Nie znaleziono katalogu wejściowego: {input_dir}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    parquet_paths = sorted(input_dir.glob("*.parquet"))
    if not parquet_paths:
        typer.secho(f"Brak plików Parquet w {input_dir}", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    if sample_sheet is None:
        try:
            sample_sheet = _find_sample_sheet(_default_raw_dir())
        except FileNotFoundError as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc

    typer.echo(f"Wczytuję sample sheet: {sample_sheet}")
    try:
        sheet_df = parse_sample_sheet(sample_sheet)
    except (SampleSheetParserError, FileNotFoundError) as exc:
        typer.secho(f"Błąd wczytywania sample sheet: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"Buduję macierz z {len(parquet_paths)} plik(ów) parquet, metryka: {metric.value}"
    )
    try:
        matrix = build_expression_matrix(
            parquet_paths=parquet_paths,
            sample_sheet=sheet_df,
            metric=metric.value,
        )
    except ExpressionMatrixError as exc:
        typer.secho(f"Błąd budowania macierzy: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "expression_matrix.parquet"
    manifest_path = output_dir / "expression_matrix_manifest.json"

    matrix.write_parquet(matrix_path)
    manifest = build_manifest(matrix, parquet_paths, metric.value)
    save_manifest(manifest, manifest_path)

    typer.secho(
        f"Macierz zapisana: {matrix_path} ({matrix.height} genów x {matrix.width - 1} próbek)",
        fg=typer.colors.GREEN,
    )
    typer.secho(f"Manifest zapisany: {manifest_path}", fg=typer.colors.GREEN)
    typer.secho(
        f"Zakończono. Rozmiar pliku: {matrix_path.stat().st_size / 1024:.1f} KB",
        fg=typer.colors.BRIGHT_GREEN,
    )


if __name__ == "__main__":
    app()
