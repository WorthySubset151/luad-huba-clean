"""Interfejs CLI dla etapów pipeline'u LUAD-HUBA."""

__author__ = "Łukasz Połaski"

from enum import Enum
from pathlib import Path
from typing import Optional

import polars as pl
import typer

from src.cli_config import (
    ConfigError,
    get_nested,
    load_config,
    resolve_metric,
)
from src.ingest.cases_client import (
    CasesClientError,
    parse_cases_response,
    query_cases,
)
from src.ingest.file_naming import STAR_FILE_PATTERNS, STAR_FILE_SUFFIXES
from src.ingest.gdc_client import (
    GDCClientError,
    build_files_filter,
    download_files,
    parse_files_response,
    query_files,
)
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
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Plik YAML z parametrami pipeline'u. Wczytywany informacyjnie - "
             "obecnie parse-star nie używa pól z YAML, ale przyszłe rozszerzenia będą.",
    ),
) -> None:
    """Parsuje wszystkie pliki STAR-Counts i zapisuje je do katalogu wyjściowego."""
    if config is not None:
        try:
            load_config(config)
            typer.secho(f"Załadowano config: {config}", fg=typer.colors.CYAN)
        except ConfigError as exc:
            typer.secho(f"Błąd configu: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc

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
    metric: Optional[ExpressionMetric] = typer.Option(
        None,
        "--metric",
        help="Metryka ekspresji do wyciągnięcia z plików parquet. "
             "Pierwszeństwo: flaga CLI > config YAML > domyślnie 'unstranded'.",
    ),
    duplicate_strategy: DuplicateStrategy = typer.Option(
        DuplicateStrategy.FAIL,
        "--duplicate-strategy",
        help=(
            "Strategia obsługi duplikatów sample_id (zdarza się w TCGA przy "
            "wielokrotnych aliquotach): fail (domyślnie), deepest, first."
        ),
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Plik YAML z parametrami pipeline'u (np. configs/default.yaml). "
             "Używane pola: normalization.method (alias metryki), "
             "normalization.biotype_filter (filtr po gene_type, np. 'protein_coding'). "
             "Wartości z YAML są używane jako domyślne dla nieustawionych flag CLI.",
    ),
) -> None:
    """Buduje macierz ekspresji genów z plików Parquet po parse-star."""
    cfg: dict = {}
    if config is not None:
        try:
            cfg = load_config(config)
            typer.secho(f"Załadowano config: {config}", fg=typer.colors.CYAN)
        except ConfigError as exc:
            typer.secho(f"Błąd configu: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc

    if metric is None:
        cfg_metric = get_nested(cfg, "normalization", "method")
        if cfg_metric is not None:
            try:
                resolved = resolve_metric(cfg_metric)
                metric = ExpressionMetric(resolved)
                typer.secho(
                    f"Metryka z configu: '{cfg_metric}' -> '{resolved}'",
                    fg=typer.colors.CYAN,
                )
            except ConfigError as exc:
                typer.secho(f"Błąd configu: {exc}", fg=typer.colors.RED)
                raise typer.Exit(code=1) from exc
        else:
            metric = ExpressionMetric.UNSTRANDED

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

    biotype_filter = get_nested(cfg, "normalization", "biotype_filter")
    if biotype_filter is not None:
        if not isinstance(biotype_filter, str) or not biotype_filter.strip():
            typer.secho(
                f"Niepoprawna wartość normalization.biotype_filter w configu: "
                f"{biotype_filter!r} (oczekiwany niepusty string)",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        typer.secho(
            f"Filtr biotype z configu: {biotype_filter!r}",
            fg=typer.colors.CYAN,
        )

    try:
        matrix = build_expression_matrix(
            parquet_paths=parquet_paths,
            sample_sheet=sheet_df,
            metric=metric.value,
            duplicate_strategy=duplicate_strategy.value,
            biotype_filter=biotype_filter,
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
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Plik YAML z parametrami pipeline'u. "
             "Używane pola: survival.min_follow_up_days (filtr próbek o krótkim "
             "czasie obserwacji).",
    ),
) -> None:
    """Buduje zbiór do analizy przeżywalności z macierzy ekspresji i danych klinicznych."""
    cfg: dict = {}
    if config is not None:
        try:
            cfg = load_config(config)
            typer.secho(f"Załadowano config: {config}", fg=typer.colors.CYAN)
        except ConfigError as exc:
            typer.secho(f"Błąd configu: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc

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

    min_follow_up_days = 0
    cfg_min_followup = get_nested(cfg, "survival", "min_follow_up_days")
    if cfg_min_followup is not None:
        try:
            min_follow_up_days = int(cfg_min_followup)
            if min_follow_up_days < 0:
                raise ValueError("musi być >= 0")
            typer.secho(
                f"Filtr min_follow_up_days z configu: {min_follow_up_days} dni",
                fg=typer.colors.CYAN,
            )
        except (ValueError, TypeError) as exc:
            typer.secho(
                f"Niepoprawna wartość survival.min_follow_up_days w configu: "
                f"{cfg_min_followup!r} ({exc})",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1) from exc

    try:
        dataset = build_survival_dataset(
            expression_matrix=matrix,
            sample_sheet=sheet_df,
            clinical=clinical_df,
            tumor_only=tumor_only,
            min_follow_up_days=min_follow_up_days,
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
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Plik YAML z parametrami pipeline'u. Wczytywany informacyjnie - "
             "obecnie validate-cohort nie używa pól z YAML, ale w przyszłości "
             "stąd będą pobierane np. qc.min_mapped_reads.",
    ),
) -> None:
    """Uruchamia kontrolę spójności kohorty i zapisuje raport QC."""
    if config is not None:
        try:
            load_config(config)
            typer.secho(f"Załadowano config: {config}", fg=typer.colors.CYAN)
        except ConfigError as exc:
            typer.secho(f"Błąd configu: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc

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


@app.command("download")
def download(
    output_dir: Path = typer.Option(
        _default_raw_dir(),
        "--output-dir",
        help="Katalog docelowy na pobrane pliki (domyślnie data/raw/).",
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project",
        help="Identyfikator projektu GDC. Pierwszeństwo: flaga > config > 'TCGA-LUAD'.",
    ),
    workflow: Optional[str] = typer.Option(
        None,
        "--workflow",
        help="Workflow GDC. Pierwszeństwo: flaga > config > 'STAR - Counts'.",
    ),
    size: Optional[int] = typer.Option(
        None,
        "--size",
        help="Limit liczby plików do pobrania (do testów). Domyślnie wszystkie.",
    ),
    skip_files: bool = typer.Option(
        False,
        "--skip-files",
        help="Tylko metadane (sample_sheet, clinical, metadata.cart.json), "
             "bez plików STAR.",
    ),
    skip_clinical: bool = typer.Option(
        False,
        "--skip-clinical",
        help="Pomiń pobieranie clinical.tsv (np. jeśli już masz lokalnie).",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Plik YAML z parametrami pipeline'u. Używane pola: "
             "pipeline.project_id, pipeline.workflow_type.",
    ),
) -> None:
    """Pobiera pełną kohortę z GDC: pliki STAR-Counts, sample sheet, clinical.tsv, metadata.cart.json.

    Po wykonaniu w output_dir znajdują się 4 typy plików - dokładnie te same,
    które można pobrać ręcznie z portalu GDC. Pipeline jest dalej w pełni
    samowystarczalny, bez ręcznych kroków w przeglądarce.

    Przykład: luad-huba download --project TCGA-LUAD --size 5
    """
    cfg: dict = {}
    if config is not None:
        try:
            cfg = load_config(config)
            typer.secho(f"Załadowano config: {config}", fg=typer.colors.CYAN)
        except ConfigError as exc:
            typer.secho(f"Błąd configu: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc

    if project is None:
        project = get_nested(cfg, "pipeline", "project_id", default="TCGA-LUAD")
    if workflow is None:
        workflow = get_nested(cfg, "pipeline", "workflow_type", default="STAR - Counts")

    output_dir.mkdir(parents=True, exist_ok=True)

    typer.secho(
        f"=== Pobieranie kohorty z GDC: project={project}, workflow='{workflow}' ===",
        fg=typer.colors.CYAN,
    )
    if size is not None:
        typer.echo(f"Limit liczby plików: {size}")

    typer.echo("")
    typer.echo("[1/4] Zapytanie o metadane plików...")
    try:
        filt = build_files_filter(project_id=project, workflow_type=workflow)
        page_size = size if size is not None else 10000
        response = query_files(filters=filt, size=page_size)
    except GDCClientError as exc:
        typer.secho(f"Błąd zapytania /files: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    files_metadata = parse_files_response(response)
    total_available = response.get("data", {}).get("pagination", {}).get("total", 0)
    typer.secho(
        f"  Otrzymano metadane dla {files_metadata.height} plików "
        f"(z {total_available} dostępnych)",
        fg=typer.colors.GREEN,
    )

    typer.echo("")
    typer.echo("[2/4] Zapis sample_sheet.tsv i metadata.cart.json...")
    sheet_path = output_dir / "gdc_sample_sheet.tsv"
    _write_sample_sheet(files_metadata, sheet_path)
    typer.secho(f"  Zapisano: {sheet_path}", fg=typer.colors.GREEN)

    metadata_path = output_dir / "metadata.cart.json"
    _write_metadata_cart(response, metadata_path)
    typer.secho(f"  Zapisano: {metadata_path}", fg=typer.colors.GREEN)

    if skip_clinical:
        typer.echo("")
        typer.secho("[3/4] Pomijam clinical.tsv (--skip-clinical)", fg=typer.colors.YELLOW)
    else:
        typer.echo("")
        typer.echo("[3/4] Zapytanie o dane kliniczne (/cases)...")
        try:
            response_cases = query_cases(size=10000)
        except CasesClientError as exc:
            typer.secho(f"Błąd zapytania /cases: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc

        cases_df = parse_cases_response(response_cases)
        clinical_path = output_dir / "clinical.tsv"
        cases_df.write_csv(clinical_path, separator="\t", quote_style="never")
        typer.secho(
            f"  Zapisano: {clinical_path} ({cases_df.height} wierszy, "
            f"{cases_df['cases.submitter_id'].n_unique()} pacjentów)",
            fg=typer.colors.GREEN,
        )

    if skip_files:
        typer.echo("")
        typer.secho("[4/4] Pomijam pobieranie plików STAR (--skip-files)", fg=typer.colors.YELLOW)
    else:
        typer.echo("")
        total_mb = files_metadata["file_size"].sum() / 1024**2
        typer.echo(
            f"[4/4] Pobieranie {files_metadata.height} plików STAR-Counts "
            f"(~{total_mb:.0f} MB)..."
        )
        download_result = download_files(
            metadata=files_metadata,
            output_dir=output_dir,
            show_progress=True,
        )
        n_verified = download_result.filter(pl.col("verified")).height
        n_failed = download_result.filter(~pl.col("verified")).height
        typer.secho(
            f"  Pobrano: {n_verified}/{download_result.height} plików zweryfikowanych",
            fg=typer.colors.GREEN if n_failed == 0 else typer.colors.YELLOW,
        )
        if n_failed > 0:
            typer.secho(f"  BŁĘDY: {n_failed} plików nie zweryfikowanych", fg=typer.colors.RED)

    typer.echo("")
    typer.secho("=== Kohorta gotowa w " + str(output_dir) + " ===", fg=typer.colors.BRIGHT_GREEN)


def _write_sample_sheet(files_metadata, output_path: Path) -> None:
    """Zapisuje gdc_sample_sheet.tsv w formacie z portalu GDC.

    Format zawiera dwie podobne kolumny dotyczące rodzaju próbki:
    - 'Sample Type' - szczegółowy typ (Primary Tumor, Solid Tissue Normal, ...)
    - 'Tissue Type' - binarna klasyfikacja (Tumor/Normal), używana przez
      sample_sheet_parser do obliczania flagi is_tumor w pipeline'ie
    """
    sheet = files_metadata.select([
        pl.col("file_id").alias("File ID"),
        pl.col("file_name").alias("File Name"),
        pl.col("data_type").alias("Data Type"),
        pl.col("experimental_strategy").alias("Data Category"),
        pl.lit("TCGA-LUAD").alias("Project ID"),
        pl.col("case_submitter_id").alias("Case ID"),
        pl.col("sample_id").alias("Sample ID"),
        pl.col("sample_id")
            .str.slice(-3, 3)
            .map_elements(_tcga_code_to_tissue_type, return_dtype=pl.Utf8)
            .alias("Tissue Type"),
        pl.col("sample_id")
            .str.slice(-3, 3)
            .map_elements(_tcga_code_to_sample_type, return_dtype=pl.Utf8)
            .alias("Sample Type"),
    ])
    sheet.write_csv(output_path, separator="\t", quote_style="never")


def _tcga_code_to_tissue_type(code: str) -> str:
    """Mapuje TCGA sample code (np. '01A', '11A') na nazwę Tissue Type.

    Binarna klasyfikacja używana w pipeline'ie do flagi is_tumor:
    - kody 01-09 -> Tumor
    - kody 10-19 -> Normal
    """
    if not code or len(code) < 2:
        return "Unknown"
    try:
        num = int(code[:2])
    except ValueError:
        return "Unknown"
    if 1 <= num <= 9:
        return "Tumor"
    if 10 <= num <= 19:
        return "Normal"
    return "Unknown"


def _tcga_code_to_sample_type(code: str) -> str:
    """Mapuje TCGA sample code (np. '01A', '11A') na nazwę Sample Type."""
    if not code or len(code) < 2:
        return "Unknown"
    try:
        num = int(code[:2])
    except ValueError:
        return "Unknown"
    if 1 <= num <= 9:
        return "Primary Tumor" if num == 1 else "Recurrent Tumor" if num == 2 else "Tumor"
    if 10 <= num <= 19:
        return "Solid Tissue Normal"
    return "Other"


def _write_metadata_cart(response: dict, output_path: Path) -> None:
    """Zapisuje metadata.cart.json w formacie identycznym z eksportem portalu."""
    import json as json_lib
    hits = response.get("data", {}).get("hits", [])
    with output_path.open("w", encoding="utf-8") as fh:
        json_lib.dump(hits, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    app()
