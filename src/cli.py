"""Interfejs CLI dla etapów pipeline'u LUAD-HUBA."""

__author__ = "Łukasz Połaski"

from enum import Enum
from pathlib import Path

import typer

from src.ingest.file_naming import STAR_FILE_PATTERNS, STAR_FILE_SUFFIXES
from src.ingest.sample_sheet_parser import SampleSheetParserError, parse_sample_sheet
from src.ingest.star_parser import StarParserError, parse_star_counts
from src.ingest.clinical_parser import ClinicalParserError, parse_clinical
from src.validate.runner import discover_stems, run_cohort_qc, save_qc_report
from src.transform.expression_matrix import (
    ALLOWED_METRICS,
    ExpressionMatrixError,
    build_expression_matrix,
    build_manifest,
    save_manifest,
)
from src.transform.survival_dataset import (
    SurvivalDatasetError,
    build_survival_dataset,
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


class DuplicateStrategy(str, Enum):
    """Strategia obsługi duplikatów sample_id w macierzy ekspresji."""

    FAIL = "fail"
    DEEPEST = "deepest"
    FIRST = "first"


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
    duplicate_strategy: DuplicateStrategy = typer.Option(
        DuplicateStrategy.FAIL,
        "--duplicate-strategy",
        help=(
            "Strategia obsługi duplikatów sample_id (zdarza się w TCGA przy "
            "wielokrotnych aliquotach): fail (domyślnie), deepest, first."
        ),
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
            duplicate_strategy=duplicate_strategy.value,
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


@app.command("build-survival")
def build_survival(
    matrix_path: Path = typer.Option(
        Path("data/processed/expression_matrix.parquet"),
        "--matrix",
        help="Ścieżka do macierzy ekspresji z komendy build-matrix.",
    ),
    sample_sheet: Path = typer.Option(
        None,
        "--sample-sheet",
        help="Ścieżka do gdc_sample_sheet. Domyślnie wyszukiwana w data/raw/.",
    ),
    clinical_path: Path = typer.Option(
        None,
        "--clinical",
        help="Ścieżka do clinical.tsv. Domyślnie wyszukiwana w data/raw/.",
    ),
    output_dir: Path = typer.Option(
        _default_processed_dir(),
        "--output-dir",
        help="Katalog wyjściowy na zbiór do analizy przeżywalności.",
    ),
    tumor_only: bool = typer.Option(
        True,
        "--tumor-only/--all-samples",
        help="Zachowaj wyłącznie próbki nowotworowe (domyślnie) lub wszystkie.",
    ),
) -> None:
    """Buduje zbiór do analizy przeżywalności z macierzy ekspresji i danych klinicznych."""
    if not matrix_path.exists():
        typer.secho(f"Nie znaleziono macierzy ekspresji: {matrix_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if sample_sheet is None:
        try:
            sample_sheet = _find_sample_sheet(_default_raw_dir())
        except FileNotFoundError as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc

    if clinical_path is None:
        candidates = sorted(_default_raw_dir().glob("clinical*.tsv"))
        if not candidates:
            typer.secho(
                f"Nie znaleziono pliku clinical*.tsv w {_default_raw_dir()}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        clinical_path = candidates[0]

    import polars as pl

    typer.echo(f"Wczytuję macierz: {matrix_path}")
    matrix = pl.read_parquet(matrix_path)

    typer.echo(f"Wczytuję sample sheet: {sample_sheet}")
    typer.echo(f"Wczytuję dane kliniczne: {clinical_path}")
    try:
        sheet_df = parse_sample_sheet(sample_sheet)
        clinical_df = parse_clinical(clinical_path)
    except (SampleSheetParserError, ClinicalParserError, FileNotFoundError) as exc:
        typer.secho(f"Błąd wczytywania metadanych: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    mode = "tylko nowotworowe" if tumor_only else "wszystkie"
    typer.echo(f"Buduję zbiór przeżywalności (próbki: {mode})")
    try:
        dataset = build_survival_dataset(
            expression_matrix=matrix,
            sample_sheet=sheet_df,
            clinical=clinical_df,
            tumor_only=tumor_only,
        )
    except SurvivalDatasetError as exc:
        typer.secho(f"Błąd budowania zbioru: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "survival_dataset.parquet"
    dataset.write_parquet(out_path)

    n_genes = len([c for c in dataset.columns if c.startswith("ENSG")])
    n_events = int(dataset["event"].sum())
    typer.secho(
        f"Zbiór zapisany: {out_path} "
        f"({dataset.height} próbek x {n_genes} genów, zdarzenia: {n_events})",
        fg=typer.colors.GREEN,
    )
    typer.secho(
        f"Zakończono. Rozmiar pliku: {out_path.stat().st_size / 1024:.1f} KB",
        fg=typer.colors.BRIGHT_GREEN,
    )


@app.command("validate-cohort")
def validate_cohort(
    sample_sheet: Path = typer.Option(
        None,
        "--sample-sheet",
        help="Ścieżka do gdc_sample_sheet. Domyślnie wyszukiwana w data/raw/.",
    ),
    clinical_path: Path = typer.Option(
        None,
        "--clinical",
        help="Ścieżka do clinical.tsv. Domyślnie wyszukiwana w data/raw/.",
    ),
    interim_dir: Path = typer.Option(
        _default_interim_dir(),
        "--interim-dir",
        help="Katalog z plikami Parquet po parse-star (źródło stemów plików).",
    ),
    log_dir: Path = typer.Option(
        Path("logs/qc"),
        "--log-dir",
        help="Katalog na strukturyzowane raporty QC w formacie JSON.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Jeśli ustawione, kończy z kodem 1 przy jakimkolwiek błędzie ERROR.",
    ),
) -> None:
    """Uruchamia kontrolę spójności kohorty i zapisuje raport QC."""
    if sample_sheet is None:
        try:
            sample_sheet = _find_sample_sheet(_default_raw_dir())
        except FileNotFoundError as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc

    if clinical_path is None:
        candidates = sorted(_default_raw_dir().glob("clinical*.tsv"))
        if not candidates:
            typer.secho(
                f"Nie znaleziono pliku clinical*.tsv w {_default_raw_dir()}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        clinical_path = candidates[0]

    typer.echo(f"Wczytuję sample sheet: {sample_sheet}")
    typer.echo(f"Wczytuję dane kliniczne: {clinical_path}")
    typer.echo(f"Skanuję katalog plików: {interim_dir}")

    try:
        sheet_df = parse_sample_sheet(sample_sheet)
        clinical_df = parse_clinical(clinical_path)
    except (SampleSheetParserError, ClinicalParserError, FileNotFoundError) as exc:
        typer.secho(f"Błąd wczytywania metadanych: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    available_stems = discover_stems(interim_dir)
    typer.echo(f"Znaleziono {len(available_stems)} plik(ów) w {interim_dir}")

    report = run_cohort_qc(sheet_df, clinical_df, available_stems)
    log_path = save_qc_report(report, log_dir)

    summary = report.summary()
    typer.echo("")
    typer.secho("=== Podsumowanie QC ===", fg=typer.colors.CYAN)
    typer.echo(f"  Wszystkie problemy: {summary['total']}")
    typer.secho(
        f"  ERROR:    {summary['errors']}",
        fg=typer.colors.RED if summary["errors"] else typer.colors.GREEN,
    )
    typer.secho(
        f"  WARNING:  {summary['warnings']}",
        fg=typer.colors.YELLOW if summary["warnings"] else typer.colors.GREEN,
    )
    typer.echo(f"  INFO:     {summary['info']}")
    typer.echo("")
    typer.secho(f"Raport zapisany: {log_path}", fg=typer.colors.GREEN)

    if strict and report.has_errors:
        typer.secho(
            f"Tryb --strict: zakończono z błędem ({report.n_errors} ERROR)",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
