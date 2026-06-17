"""LUAD-HUBA — terminal w stylu z/OS / ISPF (Textual).

Menu główne (primary option menu) + ekrany raportów konsumujące warstwę headless
``src/analysis``. Te same liczby co dashboard Streamlit — terminal renderuje je
jako zielono-czarne tabele zamiast wykresów Plotly.

Uruchomienie:  ``python -m tui``  albo console script ``luad-huba-tui``.
Nawigacja: strzałki + Enter lub wpisanie numeru opcji w linii ``Opcja ===>``.
Klawisze PF: PF1 pomoc, PF3 koniec/powrót, PF5 odśwież.
"""

__author__ = "Łukasz Połaski"

import sys
from pathlib import Path

# Repo root na ścieżce — by `python -m tui` działało z dowolnego cwd (jak app/main.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl  # noqa: E402
from rich.console import Group  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402
from textual import work  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual.containers import Horizontal, Vertical, VerticalScroll  # noqa: E402
from textual.screen import Screen  # noqa: E402
from textual.widgets import (  # noqa: E402
    Footer, Input, Label, ListItem, ListView, ProgressBar, Static,
)

from src.analysis import survival_report as sr  # noqa: E402
from src.analysis.expression_report import expression_summary, LUAD_MARKERS  # noqa: E402
from src.pipeline_status import pipeline_status  # noqa: E402
from src.validate.runner import run_cohort_qc, discover_stems  # noqa: E402
from src.validate.report_view import classify_qc  # noqa: E402
from src.ingest.sample_sheet_parser import parse_sample_sheet  # noqa: E402
from src.ingest.clinical_parser import parse_clinical  # noqa: E402
from src.ingest.star_parser import parse_star_counts, StarParserError  # noqa: E402
from src.ingest.file_naming import STAR_FILE_PATTERNS, extract_star_file_stem  # noqa: E402
from src.transform.expression_matrix import build_expression_matrix  # noqa: E402
from src.transform.survival_dataset import build_survival_dataset  # noqa: E402
from src.ingest.gdc_client import (  # noqa: E402
    build_files_filter, query_files, parse_files_response, download_files, GDCClientError,
)
from src.ingest.cases_client import query_cases, parse_cases_response, CasesClientError  # noqa: E402
from src.cli_config import resolve_metric, ConfigError  # noqa: E402
from src.manage.data_ops import (  # noqa: E402
    MANAGE_SCOPES, scope_stats, fmt_size, delete_scope, build_archive_to_path,
)
import yaml  # noqa: E402
from tui import render  # noqa: E402

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim" / "star_counts"
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
SURVIVAL_PARQUET = DATA_PROCESSED / "survival_dataset.parquet"
EXPRESSION_MATRIX = DATA_PROCESSED / "expression_matrix.parquet"
VERSION = "v0.1"

# (kod opcji, nazwa, opis, akcja)
MENU = [
    ("1", "DOWNLOAD", "Pobieranie z GDC API (metadane + pliki STAR + clinical)", "download"),
    ("2", "INGEST", "Parsowanie STAR-Counts po ścieżce → data/interim", "ingest"),
    ("3", "MATRIX", "Budowa macierzy ekspresji (interim → expression_matrix)", "matrix"),
    ("4", "DATASET", "Budowa zbioru przeżywalności (macierz + clinical → parquet)", "dataset"),
    ("5", "PIPELINE", "Status etapów ETL (zbiory danych, kompletność)", "pipeline"),
    ("6", "VALIDATE", "Walidacja kohorty (QC — spójność próbek/klinika)", "validate"),
    ("7", "STATUS", "Przegląd kohorty (próbki, zdarzenia, geny, stadia)", "status"),
    ("8", "EXPRESSION", "Macierz ekspresji (rozkład, batch TSS, PCA)", "expression"),
    ("9", "SURVIVAL", "Analiza przeżywalności (Kaplan-Meier + Cox HR + rygor stat.)", "survival"),
    ("10", "CONFIG", "Konfiguracja pipeline'u (podgląd configs/default.yaml)", "config"),
    ("11", "ZARZĄDZANIE", "Archiwizacja ZIP + bezpieczne kasowanie danych (z potwierdzeniem)", "manage"),
    ("X", "KONIEC", "Wyjście z programu", "exit"),
]
_ACTION_BY_CODE = {code: action for code, _name, _desc, action in MENU}


def load_dataset():
    """Zwraca (ds, None) albo (None, komunikat_błędu) gdy brak/uszkodzony parquet."""
    if not SURVIVAL_PARQUET.exists():
        return None, (
            "BRAK ZBIORU PRZEŻYWALNOŚCI\n\n"
            f"Nie znaleziono pliku:\n  {SURVIVAL_PARQUET}\n\n"
            "Uruchom najpierw pipeline (CLI lub dashboard) i wygeneruj etap\n"
            "'Zbiór przeżywalności' (survival_dataset.parquet)."
        )
    try:
        return pl.read_parquet(SURVIVAL_PARQUET), None
    except Exception as exc:  # noqa: BLE001
        return None, f"BŁĄD ODCZYTU PARQUET:\n  {exc}"


class PanelHeader(Static):
    """Pasek panel-ID w stylu ISPF: lewy panel-id, środek tytuł, prawy wersja."""

    def __init__(self, panel_id: str, title: str) -> None:
        super().__init__()
        self._panel_id = panel_id
        self._title = title

    def render(self):
        grid = Table.grid(expand=True)
        grid.add_column(justify="left")
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="right")
        grid.add_row(
            Text(self._panel_id, style=render.HEAD),
            Text(self._title, style=f"bold {render.GREEN}"),
            Text(f"LUAD-HUBA {VERSION}", style=render.DIM),
        )
        return grid


class ReportScreen(Screen):
    """Bazowy ekran raportu: panel-header + przewijalna treść + pasek PF."""

    BINDINGS = [
        Binding("f3,escape", "app.pop_screen", "PF3 Koniec"),
        Binding("f5", "refresh", "PF5 Odśwież"),
    ]
    PANEL_ID = "LUADHUB"
    TITLE_TXT = ""

    def compose(self) -> ComposeResult:
        yield PanelHeader(self.PANEL_ID, self.TITLE_TXT)
        with VerticalScroll(id="body"):
            yield Static(id="content")
        yield Footer()

    def on_mount(self) -> None:
        self.action_refresh()

    def action_refresh(self) -> None:
        content = self.query_one("#content", Static)
        ds, err = load_dataset()
        if err:
            content.update(Text(err, style=f"bold {render.RISK}"))
            return
        content.update(self.build_report(ds))

    def build_report(self, ds):  # nadpisywane
        raise NotImplementedError


class StatusScreen(ReportScreen):
    PANEL_ID = "LUADHUB.STAT"
    TITLE_TXT = "STATUS — PRZEGLĄD KOHORTY"

    def build_report(self, ds):
        return render.cohort_report(sr.cohort_summary(ds))


class SurvivalScreen(ReportScreen):
    PANEL_ID = "LUADHUB.SURV"
    TITLE_TXT = "SURVIVAL — KAPLAN-MEIER + COX"

    # Katalog genów i widoki domyślne — zgodne z GUI (Dashboard analityczny)
    GENE_CATALOG = {**LUAD_MARKERS,
                    **{sym: ensg for sym, (ensg, _s) in sr.SIGNATURE_PANEL.items()}}
    DEFAULT_SINGLE_GENE = "ALK"                      # index 0 posortowanego katalogu (GUI)
    DEFAULT_MULTI_GENES = ["NKX2-1", "MKI67", "BIRC5"]  # default multiselect (GUI)

    def build_report(self, ds):
        km = sr.km_summary(ds)
        cox_clin = sr.cox_clinical_report(ds)
        signature = sr.signature_km_report(ds)
        single = sr.single_gene_km_report(
            ds, self.GENE_CATALOG.get(self.DEFAULT_SINGLE_GENE, ""), self.DEFAULT_SINGLE_GENE)
        multi_pairs = [(g, self.GENE_CATALOG[g])
                       for g in self.DEFAULT_MULTI_GENES if g in self.GENE_CATALOG]
        multi = sr.multi_gene_km_report(ds, multi_pairs)
        cox_genes = sr.cox_genes_report(ds)
        stat = sr.statistical_rigor_report(ds)
        return render.survival_report(km, cox_clin, signature, single, multi, cox_genes, stat)


class PipelineScreen(ReportScreen):
    """Status etapów ETL — inspekcja dysku, działa też gdy artefaktów brak."""

    PANEL_ID = "LUADHUB.PIPE"
    TITLE_TXT = "PIPELINE — STATUS ETL"

    def action_refresh(self) -> None:
        # Nie wymaga survival_dataset — sensem panelu jest pokazać, czego brak.
        content = self.query_one("#content", Static)
        content.update(render.pipeline_report(pipeline_status(PROJECT_ROOT)))


class ExpressionScreen(ReportScreen):
    """Podsumowanie macierzy ekspresji (rozkład, batch TSS, PCA, markery)."""

    PANEL_ID = "LUADHUB.EXPR"
    TITLE_TXT = "EXPRESSION — MACIERZ EKSPRESJI"

    def action_refresh(self) -> None:
        # expression_summary sam zgłasza brak macierzy jako {'error': ...}.
        content = self.query_one("#content", Static)
        content.update(render.expression_report(expression_summary(EXPRESSION_MATRIX)))


def _find_sample_sheet():
    sheets = sorted(DATA_RAW.glob("gdc_sample_sheet*.tsv"))
    return sheets[0] if sheets else None


def _run_qc() -> dict:
    """Uruchamia QC kohorty (read-only). Zwraca dane dla render.qc_report
    albo {'error': ...} gdy brak wymaganych metadanych."""
    sheet = _find_sample_sheet()
    clinical = DATA_RAW / "clinical.tsv"
    stems = discover_stems(DATA_INTERIM)
    missing = []
    if not stems:
        missing.append(f"sparsowane parquety STAR w {DATA_INTERIM}")
    if sheet is None:
        missing.append(f"sample sheet (gdc_sample_sheet*.tsv) w {DATA_RAW}")
    if not clinical.exists():
        missing.append(f"clinical.tsv w {DATA_RAW}")
    if missing:
        return {"error": "BRAK DANYCH DO WALIDACJI\n\n  - " + "\n  - ".join(missing)
                + "\n\nUruchom najpierw etapy Pobieranie + Parsowanie (CLI lub dashboard)."}
    try:
        sheet_df = parse_sample_sheet(sheet)
        clinical_df = parse_clinical(clinical)
        report = run_cohort_qc(sheet_df, clinical_df, stems)
        issues = [i.to_dict() for i in report.issues]
        return {"classified": classify_qc(report.summary(), issues), "n_parsed": len(stems)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"BŁĄD WALIDACJI:\n  {type(exc).__name__}: {exc}"}


def _read_config() -> dict:
    """Czyta configs/default.yaml (read-only). Zwraca dane dla render.config_report."""
    if not CONFIG_PATH.exists():
        return {"error": f"BRAK PLIKU KONFIGURACYJNEGO:\n  {CONFIG_PATH}"}
    try:
        raw = CONFIG_PATH.read_text(encoding="utf-8")
        cfg = yaml.safe_load(raw) or {}
        return {"cfg": cfg, "raw": raw, "path": str(CONFIG_PATH.relative_to(PROJECT_ROOT))}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"BŁĄD ODCZYTU YAML:\n  {type(exc).__name__}: {exc}"}


class ValidateScreen(ReportScreen):
    """Walidacja kohorty (QC) — read-only; działa też gdy brak metadanych."""

    PANEL_ID = "LUADHUB.VALD"
    TITLE_TXT = "VALIDATE — WALIDACJA KOHORTY"

    def action_refresh(self) -> None:
        content = self.query_one("#content", Static)
        content.update(render.qc_report(_run_qc()))


class ConfigScreen(ReportScreen):
    """Konfiguracja pipeline'u (read-only) — wartości + surowy YAML."""

    PANEL_ID = "LUADHUB.CONF"
    TITLE_TXT = "CONFIG — KONFIGURACJA"

    def action_refresh(self) -> None:
        content = self.query_one("#content", Static)
        content.update(render.config_report(_read_config()))


def _discover_star_files_at(path: Path) -> list:
    """Pliki STAR pod ścieżką: pojedynczy plik albo rekurencyjnie z katalogu."""
    if path.is_file():
        return [path]
    if path.is_dir():
        found = set()
        for pattern in STAR_FILE_PATTERNS:
            found.update(path.rglob(pattern))
        return sorted(found)
    return []


class IngestScreen(Screen):
    """Parsowanie STAR-Counts po ścieżce → data/interim (worker w tle + progress)."""

    PANEL_ID = "LUADHUB.INGE"
    TITLE_TXT = "INGEST — PARSOWANIE STAR PO ŚCIEŻCE"
    BINDINGS = [
        Binding("f3,escape", "app.pop_screen", "PF3 Koniec"),
    ]

    def compose(self) -> ComposeResult:
        yield PanelHeader(self.PANEL_ID, self.TITLE_TXT)
        with Vertical(id="ingest-wrap"):
            yield Static(self._intro(), id="ingest-intro")
            with Horizontal(id="ingest-cmdline"):
                yield Label("Ścieżka ===> ", id="ingest-label")
                yield Input(
                    id="ingest-path",
                    placeholder="katalog (rekurencyjnie) lub plik STAR .tsv — Enter parsuje",
                )
            yield ProgressBar(id="ingest-progress", show_eta=False)
            yield Static(id="ingest-status")
        yield Footer()

    def _intro(self):
        return Text.assemble(
            ("Surowe STAR-Counts (TSV) → data/interim/star_counts/*.parquet\n", render.HEAD),
            ("ZAPIS na dysk: istniejące parquety o tej samej nazwie zostaną nadpisane.\n", render.DIM),
            (f"Wzorce: {'  '.join(STAR_FILE_PATTERNS)}", render.DIM),
        )

    def on_mount(self) -> None:
        self.query_one("#ingest-progress", ProgressBar).display = False
        self.query_one("#ingest-path", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        raw = event.value.strip()
        if not raw:
            return
        self._start(Path(raw).expanduser())

    def _start(self, path: Path) -> None:
        status = self.query_one("#ingest-status", Static)
        files = _discover_star_files_at(path)
        if not files:
            status.update(Text(
                f"Brak plików STAR pod:\n  {path}\nWzorce: {'  '.join(STAR_FILE_PATTERNS)}",
                style=render.RISK))
            return
        pbar = self.query_one("#ingest-progress", ProgressBar)
        pbar.display = True
        pbar.update(total=len(files), progress=0)
        status.update(Text(f"Znaleziono {len(files)} plików — parsowanie…", style=render.GREEN))
        self.query_one("#ingest-path", Input).disabled = True
        self._parse_worker(files)

    @work(thread=True, exclusive=True)
    def _parse_worker(self, files: list) -> None:
        DATA_INTERIM.mkdir(parents=True, exist_ok=True)
        errors = []
        for done, path in enumerate(files, start=1):
            try:
                df = parse_star_counts(path)
                df.write_parquet(DATA_INTERIM / f"{extract_star_file_stem(path)}.parquet")
            except (StarParserError, FileNotFoundError, OSError) as exc:
                errors.append(f"{path.name}: {exc}")
            self.app.call_from_thread(self._tick, done, path.name)
        self.app.call_from_thread(self._done, len(files), errors)

    def _tick(self, done: int, name: str) -> None:
        self.query_one("#ingest-progress", ProgressBar).progress = done
        self.query_one("#ingest-status", Static).update(
            Text(f"Przetworzono {done} — ostatni: {name}", style=render.DIM))

    def _done(self, total: int, errors: list) -> None:
        self.query_one("#ingest-path", Input).disabled = False
        status = self.query_one("#ingest-status", Static)
        ok = total - len(errors)
        if errors:
            lines = "\n".join(f"  • {e}" for e in errors[:15])
            more = f"\n  … oraz {len(errors) - 15} więcej" if len(errors) > 15 else ""
            status.update(Text(
                f"Zakończono: {ok}/{total} OK, {len(errors)} błędów\n{lines}{more}",
                style=render.WARN if ok else render.RISK))
        else:
            status.update(Text(
                f"✓ Sparsowano {total} plików → data/interim/star_counts/",
                style=render.GREEN))


class BuildMatrixScreen(Screen):
    """Budowa macierzy ekspresji: interim parquety → expression_matrix.parquet (worker)."""

    PANEL_ID = "LUADHUB.MTRX"
    TITLE_TXT = "BUILD MATRIX — MACIERZ EKSPRESJI"
    BINDINGS = [
        Binding("f4", "build", "PF4 Buduj"),
        Binding("f3,escape", "app.pop_screen", "PF3 Koniec"),
    ]
    METRIC = "tpm"            # domyślnie jak w GUI (selectbox index 0)
    DUP_STRATEGY = "deepest"  # zalecane dla TCGA

    def __init__(self) -> None:
        super().__init__()
        self._busy = False

    def compose(self) -> ComposeResult:
        yield PanelHeader(self.PANEL_ID, self.TITLE_TXT)
        with Vertical(id="mtrx-wrap"):
            yield Static(id="mtrx-plan")
            yield ProgressBar(id="mtrx-progress", show_eta=False)
            yield Static(id="mtrx-status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#mtrx-progress", ProgressBar).display = False
        self._render_plan()

    def _gather(self):
        parquets = sorted(DATA_INTERIM.glob("*.parquet"))
        sheet = _find_sample_sheet()
        cfgdata = _read_config()
        cfg = cfgdata.get("cfg", {}) if "cfg" in cfgdata else {}
        biotype = (cfg.get("normalization", {}) or {}).get("biotype_filter", "protein_coding")
        return parquets, sheet, biotype

    def _render_plan(self) -> None:
        parquets, sheet, biotype = self._gather()
        t = Table(box=render._BOX, border_style=render.DIM, show_header=False)
        t.add_column(style=render.DIM)
        t.add_column(style=render.GREEN, justify="right")
        t.add_row("Parquetów do połączenia", str(len(parquets)))
        t.add_row("Metryka", self.METRIC)
        t.add_row("Strategia duplikatów", self.DUP_STRATEGY)
        t.add_row("Filtr biotype", biotype if biotype else "(brak — wszystkie geny)")
        t.add_row("Wyjście", "data/processed/expression_matrix.parquet")
        head = Text("Interim parquety → macierz geny × próbki (filtr biotype, deduplikacja)",
                    style=render.HEAD)
        self.query_one("#mtrx-plan", Static).update(Group(head, Text(""), t))

        status = self.query_one("#mtrx-status", Static)
        problems = []
        if not parquets:
            problems.append(f"brak parquetów w {DATA_INTERIM} — najpierw INGEST/Parsowanie")
        if sheet is None:
            problems.append(f"brak sample sheet (gdc_sample_sheet*.tsv) w {DATA_RAW}")
        if problems:
            status.update(Text("BRAK DANYCH:\n  - " + "\n  - ".join(problems), style=render.RISK))
        else:
            status.update(Text("Gotowe. Naciśnij PF4 aby zbudować macierz.", style=render.GREEN))

    def action_build(self) -> None:
        if self._busy:
            return
        parquets, sheet, biotype = self._gather()
        if not parquets or sheet is None:
            self._render_plan()
            return
        self._busy = True
        pbar = self.query_one("#mtrx-progress", ProgressBar)
        pbar.display = True
        pbar.update(total=len(parquets), progress=0)
        self.query_one("#mtrx-status", Static).update(
            Text(f"Budowanie macierzy z {len(parquets)} parquetów…", style=render.GREEN))
        self._build_worker(parquets, sheet, biotype)

    @work(thread=True, exclusive=True)
    def _build_worker(self, parquets, sheet, biotype) -> None:
        try:
            sheet_df = parse_sample_sheet(sheet)
            metric_resolved = resolve_metric(self.METRIC)
        except (ConfigError, OSError, ValueError) as exc:
            self.app.call_from_thread(self._fail, f"Przygotowanie: {exc}")
            return

        def cb(done: int, total: int) -> None:
            self.app.call_from_thread(self._tick, done, total)

        try:
            matrix = build_expression_matrix(
                parquets, sheet_df, metric=metric_resolved,
                duplicate_strategy=self.DUP_STRATEGY,
                biotype_filter=biotype if (biotype and biotype.strip()) else None,
                progress_callback=cb,
            )
            DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
            matrix.write_parquet(DATA_PROCESSED / "expression_matrix.parquet")
            self.app.call_from_thread(self._done, matrix.height, matrix.width - 1)
        except Exception as exc:  # noqa: BLE001
            self.app.call_from_thread(self._fail, f"{type(exc).__name__}: {exc}")

    def _tick(self, done: int, total: int) -> None:
        self.query_one("#mtrx-progress", ProgressBar).update(total=total, progress=done)
        self.query_one("#mtrx-status", Static).update(
            Text(f"Połączono {done}/{total} próbek…", style=render.DIM))

    def _done(self, n_genes: int, n_samples: int) -> None:
        self._busy = False
        self.query_one("#mtrx-status", Static).update(Text(
            f"✓ Macierz: {n_genes} genów × {n_samples} próbek "
            "→ data/processed/expression_matrix.parquet", style=render.GREEN))

    def _fail(self, msg: str) -> None:
        self._busy = False
        self.query_one("#mtrx-status", Static).update(
            Text(f"Błąd budowy macierzy:\n  {msg}", style=render.RISK))


class BuildSurvivalScreen(Screen):
    """Budowa zbioru przeżywalności: macierz + clinical → survival_dataset.parquet (worker)."""

    PANEL_ID = "LUADHUB.SURB"
    TITLE_TXT = "BUILD DATASET — ZBIÓR PRZEŻYWALNOŚCI"
    BINDINGS = [
        Binding("f4", "build", "PF4 Buduj"),
        Binding("f3,escape", "app.pop_screen", "PF3 Koniec"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._busy = False

    def compose(self) -> ComposeResult:
        yield PanelHeader(self.PANEL_ID, self.TITLE_TXT)
        with Vertical(id="surv-wrap"):
            yield Static(id="surv-plan")
            yield ProgressBar(id="surv-progress", show_eta=False)
            yield Static(id="surv-status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#surv-progress", ProgressBar).display = False
        self._render_plan()

    def _opts(self) -> dict:
        cfgdata = _read_config()
        cfg = cfgdata.get("cfg", {}) if "cfg" in cfgdata else {}
        surv = cfg.get("survival", {}) or {}
        return {
            "tumor_only": True,  # domyślnie jak w GUI
            "drop_zero_time": bool(surv.get("drop_zero_time", True)),
            "min_follow_up_days": int(surv.get("min_follow_up_days", 30)),
        }

    def _prereqs(self):
        matrix = DATA_PROCESSED / "expression_matrix.parquet"
        sheet = _find_sample_sheet()
        clinical = DATA_RAW / "clinical.tsv"
        problems = []
        if not matrix.exists():
            problems.append(f"brak macierzy ({matrix.name}) — najpierw MATRIX")
        if sheet is None:
            problems.append(f"brak sample sheet (gdc_sample_sheet*.tsv) w {DATA_RAW}")
        if not clinical.exists():
            problems.append(f"brak clinical.tsv w {DATA_RAW}")
        return matrix, sheet, clinical, problems

    def _render_plan(self) -> None:
        matrix, sheet, clinical, problems = self._prereqs()
        o = self._opts()
        t = Table(box=render._BOX, border_style=render.DIM, show_header=False)
        t.add_column(style=render.DIM)
        t.add_column(style=render.GREEN, justify="right")
        t.add_row("Tylko nowotworowe", "tak" if o["tumor_only"] else "nie")
        t.add_row("Usuń artefakty time<=0", "tak" if o["drop_zero_time"] else "nie")
        t.add_row("Min follow-up [dni]", str(o["min_follow_up_days"]))
        t.add_row("Wyjście", "data/processed/survival_dataset.parquet")
        head = Text("Macierz + sample sheet + clinical → zbiór przeżywalności (filtry warunkowe)",
                    style=render.HEAD)
        self.query_one("#surv-plan", Static).update(Group(head, Text(""), t))

        status = self.query_one("#surv-status", Static)
        if problems:
            status.update(Text("BRAK DANYCH:\n  - " + "\n  - ".join(problems), style=render.RISK))
        else:
            status.update(Text("Gotowe. Naciśnij PF4 aby zbudować zbiór.", style=render.GREEN))

    def action_build(self) -> None:
        if self._busy:
            return
        matrix, sheet, clinical, problems = self._prereqs()
        if problems:
            self._render_plan()
            return
        self._busy = True
        pbar = self.query_one("#surv-progress", ProgressBar)
        pbar.display = True
        pbar.update(total=None)  # nieokreślony — build nie raportuje kroków
        self.query_one("#surv-status", Static).update(
            Text("Budowanie zbioru (wczytywanie + integracja + filtry)…", style=render.GREEN))
        self._build_worker(matrix, sheet, clinical, self._opts())

    @work(thread=True, exclusive=True)
    def _build_worker(self, matrix_path, sheet, clinical, opts) -> None:
        try:
            matrix = pl.read_parquet(matrix_path)
            sheet_df = parse_sample_sheet(sheet)
            clinical_df = parse_clinical(clinical)
            dataset = build_survival_dataset(
                matrix, sheet_df, clinical_df,
                tumor_only=opts["tumor_only"],
                min_follow_up_days=opts["min_follow_up_days"],
                drop_zero_time=opts["drop_zero_time"],
            )
            DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
            dataset.write_parquet(DATA_PROCESSED / "survival_dataset.parquet")
            n = dataset.height
            ev = int(dataset["event"].sum()) if "event" in dataset.columns else 0
            cens = (1 - ev / n) * 100 if n else 0.0
            self.app.call_from_thread(self._done, n, ev, cens)
        except Exception as exc:  # noqa: BLE001
            self.app.call_from_thread(self._fail, f"{type(exc).__name__}: {exc}")

    def _done(self, n: int, ev: int, cens: float) -> None:
        self._busy = False
        self.query_one("#surv-progress", ProgressBar).update(total=1, progress=1)
        self.query_one("#surv-status", Static).update(Text(
            f"✓ Zbiór: {n} próbek, zdarzenia {ev}, cenzura {cens:.1f}% "
            "→ data/processed/survival_dataset.parquet", style=render.GREEN))

    def _fail(self, msg: str) -> None:
        self._busy = False
        self.query_one("#surv-progress", ProgressBar).display = False
        self.query_one("#surv-status", Static).update(
            Text(f"Błąd budowy zbioru:\n  {msg}", style=render.RISK))


class GDCDownloadScreen(Screen):
    """Pobieranie z GDC API: metadane (PF4) + pliki STAR + sheet + clinical (PF5) w workerze."""

    PANEL_ID = "LUADHUB.GDC"
    TITLE_TXT = "DOWNLOAD — POBIERANIE Z GDC"
    BINDINGS = [
        Binding("f4", "check", "PF4 Sprawdź"),
        Binding("f5", "download", "PF5 Pobierz"),
        Binding("f3,escape", "app.pop_screen", "PF3 Koniec"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._busy = False

    def compose(self) -> ComposeResult:
        yield PanelHeader(self.PANEL_ID, self.TITLE_TXT)
        with Vertical(id="gdc-wrap"):
            yield Static(self._intro(), id="gdc-intro")
            with Horizontal(classes="gdc-row"):
                yield Label("Projekt   ===> ", classes="gdc-lab")
                yield Input(value="TCGA-LUAD", id="gdc-project")
            with Horizontal(classes="gdc-row"):
                yield Label("Workflow  ===> ", classes="gdc-lab")
                yield Input(value="STAR - Counts", id="gdc-workflow")
            with Horizontal(classes="gdc-row"):
                yield Label("Plików    ===> ", classes="gdc-lab")
                yield Input(value="10", id="gdc-nfiles")
            yield ProgressBar(id="gdc-progress", show_eta=False)
            yield Static(id="gdc-status")
        yield Footer()

    def _intro(self):
        return Text.assemble(
            ("Pobieranie TCGA z Genomic Data Commons (podzbiory; pełna kohorta → CLI).\n",
             render.HEAD),
            ("PF4 sprawdza metadane (szybkie). PF5 pobiera pliki STAR + sample sheet + clinical.\n",
             render.DIM),
            ("ZAPIS: data/raw/ (uploaded_star/, gdc_sample_sheet.tsv, clinical.tsv).", render.DIM),
        )

    def on_mount(self) -> None:
        self.query_one("#gdc-progress", ProgressBar).display = False
        self.query_one("#gdc-project", Input).focus()

    def _params(self):
        project = self.query_one("#gdc-project", Input).value.strip() or "TCGA-LUAD"
        workflow = self.query_one("#gdc-workflow", Input).value.strip() or "STAR - Counts"
        raw_n = self.query_one("#gdc-nfiles", Input).value.strip()
        try:
            n = max(1, int(raw_n))
        except ValueError:
            n = 10
        return project, workflow, n

    # --- Krok 1: sprawdzenie dostępności (metadane) ---
    def action_check(self) -> None:
        if self._busy:
            return
        project, workflow, _ = self._params()
        self._busy = True
        self.query_one("#gdc-status", Static).update(
            Text(f"Zapytanie do GDC o metadane ({project})…", style=render.GREEN))
        self._check_worker(project, workflow)

    @work(thread=True, exclusive=True)
    def _check_worker(self, project, workflow) -> None:
        try:
            filt = build_files_filter(project_id=project, workflow_type=workflow)
            response = query_files(filters=filt, size=10000)
            meta = parse_files_response(response)
            total = response.get("data", {}).get("pagination", {}).get("total", 0)
            total_mb = meta["file_size"].sum() / 1024 ** 2 if "file_size" in meta.columns else 0
            self.app.call_from_thread(self._check_done, meta.height, total, total_mb)
        except Exception as exc:  # noqa: BLE001
            self.app.call_from_thread(self._fail, f"{type(exc).__name__}: {exc}")

    def _check_done(self, n_in_resp: int, total: int, total_mb: float) -> None:
        self._busy = False
        self.query_one("#gdc-status", Static).update(Text(
            f"Dostępnych plików: {total}  (w odpowiedzi: {n_in_resp})  ·  ≈ {total_mb:.0f} MB\n"
            "Ustaw liczbę i naciśnij PF5 aby pobrać.", style=render.GREEN))

    # --- Krok 2: pobieranie ---
    def action_download(self) -> None:
        if self._busy:
            return
        project, workflow, n = self._params()
        self._busy = True
        pbar = self.query_one("#gdc-progress", ProgressBar)
        pbar.display = True
        pbar.update(total=n, progress=0)
        warn = "  (uwaga: dużo plików przez sieć bywa zawodne — dla pełnej kohorty CLI)" if n > 50 else ""
        self.query_one("#gdc-status", Static).update(
            Text(f"Pobieranie {n} plików STAR + sheet + clinical…{warn}", style=render.GREEN))
        self._download_worker(project, workflow, n)

    @work(thread=True, exclusive=True)
    def _download_worker(self, project, workflow, n) -> None:
        msgs = []
        try:
            from src.cli import _write_sample_sheet, _write_metadata_cart
            DATA_RAW.mkdir(parents=True, exist_ok=True)
            filt = build_files_filter(project_id=project, workflow_type=workflow)
            response = query_files(filters=filt, size=n)
            files_meta = parse_files_response(response).head(n)
            msgs.append(f"metadane: {files_meta.height} plików")

            _write_sample_sheet(files_meta, DATA_RAW / "gdc_sample_sheet.tsv")
            _write_metadata_cart(response, DATA_RAW / "metadata.cart.json")
            msgs.append("zapisano gdc_sample_sheet.tsv")

            try:
                resp_cases = query_cases(size=10000)
                cases_df = parse_cases_response(resp_cases)
                cases_df.write_csv(DATA_RAW / "clinical.tsv", separator="\t", quote_style="never")
                msgs.append(f"zapisano clinical.tsv ({cases_df.height} wierszy)")
            except CasesClientError as exc:
                msgs.append(f"clinical pominięte: {exc}")

            def on_progress(idx, total, name):
                self.app.call_from_thread(self._tick, idx, total, name)

            report = download_files(
                metadata=files_meta, output_dir=DATA_RAW / "uploaded_star",
                show_progress=False, progress_callback=on_progress,
            )
            n_ok = report.filter(pl.col("verified")).height
            msgs.append(f"pobrano {n_ok}/{report.height} STAR (MD5 OK)")
            self.app.call_from_thread(self._download_done, msgs)
        except Exception as exc:  # noqa: BLE001
            self.app.call_from_thread(self._fail, f"{type(exc).__name__}: {exc}\n  " + " · ".join(msgs))

    def _tick(self, idx: int, total: int, name: str) -> None:
        self.query_one("#gdc-progress", ProgressBar).update(total=total, progress=idx)
        self.query_one("#gdc-status", Static).update(
            Text(f"Pobieranie {idx}/{total}: {name}", style=render.DIM))

    def _download_done(self, msgs: list) -> None:
        self._busy = False
        self.query_one("#gdc-status", Static).update(
            Text("✓ " + "\n  ".join(msgs) + "\n→ data/raw/", style=render.GREEN))

    def _fail(self, msg: str) -> None:
        self._busy = False
        self.query_one("#gdc-progress", ProgressBar).display = False
        self.query_one("#gdc-status", Static).update(
            Text(f"Błąd GDC:\n  {msg}", style=render.RISK))


class ManageScreen(Screen):
    """Zarządzanie danymi: archiwizacja ZIP (PF4) i bezpieczne kasowanie (PF5, z potwierdzeniem USUŃ)."""

    PANEL_ID = "LUADHUB.MNG"
    TITLE_TXT = "ZARZĄDZANIE — ARCHIWIZACJA I KASOWANIE"
    BINDINGS = [
        Binding("f4", "archive", "PF4 Archiwizuj"),
        Binding("f5", "delete", "PF5 Skasuj"),
        Binding("f6", "refresh", "PF6 Odśwież"),
        Binding("f3,escape", "app.pop_screen", "PF3 Koniec"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._busy = False

    def compose(self) -> ComposeResult:
        yield PanelHeader(self.PANEL_ID, self.TITLE_TXT)
        with Vertical(id="mng-wrap"):
            yield Static(self._intro(), id="mng-intro")
            yield Static(id="mng-stats")
            with Horizontal(classes="gdc-row"):
                yield Label("Zakresy        ===> ", classes="gdc-lab")
                yield Input(value="all", id="mng-scopes",
                            placeholder="star,metadata,interim,processed lub all")
            with Horizontal(classes="gdc-row"):
                yield Label("Potwierdź USUŃ ===> ", classes="gdc-lab")
                yield Input(id="mng-confirm", placeholder="wpisz USUŃ aby odblokować PF5")
            yield ProgressBar(id="mng-progress", show_eta=False)
            yield Static(id="mng-status")
        yield Footer()

    def _intro(self):
        return Text.assemble(
            ("Backup i czyszczenie danych pipeline'u. Cztery kategorie — STAR (gigabajty) "
             "osobno od lekkich metadanych.\n", render.HEAD),
            ("PF4 pakuje wybrane zakresy do ZIP (backups/). PF5 kasuje TRWALE — dopiero po wpisaniu USUŃ.\n",
             render.DIM),
            ("Kasowanie działa wyłącznie wewnątrz data/ (twardy guard bezpieczeństwa).", render.WARN),
        )

    def on_mount(self) -> None:
        self.query_one("#mng-progress", ProgressBar).display = False
        self._refresh_stats()
        self.query_one("#mng-scopes", Input).focus()

    # --- statystyki zakresów ---
    def _refresh_stats(self) -> None:
        body = Text()
        body.append("Stan danych (klucz · kategoria · pliki · rozmiar):\n", style=render.HEAD)
        for key, (label, path, mode, _desc) in MANAGE_SCOPES.items():
            n, b = scope_stats(path, mode)
            style = render.DIM if n == 0 else render.GREEN
            body.append(f"  {key:<10}{label:<24}{n:>5} plików{fmt_size(b):>12}\n", style=style)
        self.query_one("#mng-stats", Static).update(body)

    def action_refresh(self) -> None:
        if self._busy:
            return
        self._refresh_stats()
        self.query_one("#mng-status", Static).update(
            Text("Odświeżono stan danych.", style=render.DIM))

    # --- parsowanie wyboru zakresów ---
    def _parse_scopes(self) -> list[str]:
        raw = self.query_one("#mng-scopes", Input).value.strip().lower()
        if raw in ("", "all", "*"):
            return list(MANAGE_SCOPES.keys())
        keys = [k.strip() for k in raw.replace(";", ",").split(",") if k.strip()]
        return [k for k in keys if k in MANAGE_SCOPES]

    def _selected_total(self, keys: list[str]) -> int:
        return sum(scope_stats(MANAGE_SCOPES[k][1], MANAGE_SCOPES[k][2])[0] for k in keys)

    def _warn_scopes(self) -> None:
        self.app.bell()
        self.query_one("#mng-status", Static).update(
            Text("Brak poprawnych zakresów. Użyj: star, metadata, interim, processed lub all.",
                 style=render.WARN))

    # --- PF4: archiwizacja ---
    def action_archive(self) -> None:
        if self._busy:
            return
        keys = self._parse_scopes()
        if not keys:
            self._warn_scopes()
            return
        n_total = self._selected_total(keys)
        if n_total == 0:
            self.query_one("#mng-status", Static).update(
                Text("Wybrane zakresy są puste — nie ma czego archiwizować.", style=render.DIM))
            return
        self._busy = True
        pbar = self.query_one("#mng-progress", ProgressBar)
        pbar.display = True
        pbar.update(total=n_total, progress=0)
        self.query_one("#mng-status", Static).update(
            Text(f"Pakowanie {n_total} plików z: {', '.join(keys)}…", style=render.GREEN))
        self._archive_worker(keys, n_total)

    @work(thread=True, exclusive=True)
    def _archive_worker(self, keys, n_total) -> None:
        try:
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = PROJECT_ROOT / "backups" / f"luad-huba-backup_{ts}.zip"
            targets = [(MANAGE_SCOPES[k][0].replace(" ", "_"),
                        MANAGE_SCOPES[k][1], MANAGE_SCOPES[k][2]) for k in keys]

            def on_progress(idx, total, name):
                self.app.call_from_thread(self._tick, idx, total, name)

            count, size = build_archive_to_path(targets, out_path, progress_callback=on_progress)
            self.app.call_from_thread(self._archive_done, out_path, count, size)
        except Exception as exc:  # noqa: BLE001
            self.app.call_from_thread(self._fail, f"{type(exc).__name__}: {exc}")

    def _archive_done(self, out_path, count, size) -> None:
        self._busy = False
        self.query_one("#mng-progress", ProgressBar).display = False
        rel = out_path.relative_to(PROJECT_ROOT)
        self.query_one("#mng-status", Static).update(Text(
            f"✓ Zarchiwizowano {count} plików ({fmt_size(size)})\n→ {rel}", style=render.GREEN))

    # --- PF5: kasowanie (wymaga potwierdzenia USUŃ) ---
    def action_delete(self) -> None:
        if self._busy:
            return
        keys = self._parse_scopes()
        if not keys:
            self._warn_scopes()
            return
        confirm = self.query_one("#mng-confirm", Input).value.strip().upper()
        if confirm != "USUŃ":
            self.app.bell()
            self.query_one("#mng-status", Static).update(Text.assemble(
                ("KASOWANIE ZABLOKOWANE. ", render.RISK),
                (f"Wpisz USUŃ w polu potwierdzenia, aby trwale skasować: {', '.join(keys)}.",
                 render.WARN)))
            return
        n_total = self._selected_total(keys)
        if n_total == 0:
            self.query_one("#mng-status", Static).update(
                Text("Wybrane zakresy są już puste — nie ma czego kasować.", style=render.DIM))
            return
        self._busy = True
        pbar = self.query_one("#mng-progress", ProgressBar)
        pbar.display = True
        pbar.update(total=n_total, progress=0)
        self.query_one("#mng-status", Static).update(
            Text(f"Kasowanie {n_total} plików z: {', '.join(keys)}…", style=render.RISK))
        self._delete_worker(keys, n_total)

    @work(thread=True, exclusive=True)
    def _delete_worker(self, keys, n_total) -> None:
        msgs = []
        done = {"n": 0}
        try:
            for k in keys:
                label, path, mode, _ = MANAGE_SCOPES[k]

                def on_progress(idx, total, name, _label=label):
                    done["n"] += 1
                    self.app.call_from_thread(self._tick, done["n"], n_total, f"{_label}: {name}")

                deleted, errors = delete_scope(path, mode, progress_callback=on_progress)
                msgs.append(f"{label}: usunięto {deleted}")
                for e in errors:
                    msgs.append(f"  ⚠ {e}")
            self.app.call_from_thread(self._delete_done, msgs)
        except Exception as exc:  # noqa: BLE001
            self.app.call_from_thread(
                self._fail, f"{type(exc).__name__}: {exc}\n  " + " · ".join(msgs))

    def _delete_done(self, msgs) -> None:
        self._busy = False
        self.query_one("#mng-progress", ProgressBar).display = False
        self.query_one("#mng-confirm", Input).value = ""
        self._refresh_stats()
        self.query_one("#mng-status", Static).update(
            Text("✓ Kasowanie zakończone:\n  " + "\n  ".join(msgs), style=render.GREEN))

    # --- wspólne ---
    def _tick(self, idx, total, name) -> None:
        self.query_one("#mng-progress", ProgressBar).update(total=total, progress=idx)
        self.query_one("#mng-status", Static).update(
            Text(f"{idx}/{total}: {name}", style=render.DIM))

    def _fail(self, msg: str) -> None:
        self._busy = False
        self.query_one("#mng-progress", ProgressBar).display = False
        self.query_one("#mng-status", Static).update(
            Text(f"Błąd zarządzania:\n  {msg}", style=render.RISK))


class PrimaryMenu(Screen):
    """Menu główne w stylu ISPF (primary option menu)."""

    BINDINGS = [
        Binding("f1", "help", "PF1 Pomoc"),
        Binding("f3,escape", "app.quit", "PF3 Koniec"),
    ]

    def compose(self) -> ComposeResult:
        yield PanelHeader("LUADHUB.PRIM", "MENU GŁÓWNE — PRIMARY OPTION MENU")
        with Vertical(id="menu-wrap"):
            yield Static(self._intro(), id="intro")
            yield ListView(id="options")
            with Horizontal(id="cmdline"):
                yield Label("Opcja ===> ", id="cmdlabel")
                yield Input(id="cmd", placeholder="wpisz numer opcji i Enter")
        yield Footer()

    def on_mount(self) -> None:
        options = self.query_one("#options", ListView)
        for code, name, desc, _action in MENU:
            label = Text.assemble(
                (f"  {code}  ", f"bold {render.GREEN}"),
                (f"{name:<10}", render.HEAD),
                (desc, render.DIM),
            )
            options.append(ListItem(Static(label), id=f"opt-{code}"))
        self.query_one("#cmd", Input).focus()

    def _intro(self) -> Text:
        return Text.assemble(
            ("TCGA-LUAD · pipeline przeżywalności\n", render.HEAD),
            ("Wybierz opcję strzałkami i Enter, albo wpisz jej numer poniżej.", render.DIM),
        )

    # -- routing ----------------------------------------------------------
    def _route(self, code: str) -> None:
        action = _ACTION_BY_CODE.get(code.strip().upper())
        if action == "exit":
            self.app.exit()
        elif action == "status":
            self.app.push_screen(StatusScreen())
        elif action == "survival":
            self.app.push_screen(SurvivalScreen())
        elif action == "pipeline":
            self.app.push_screen(PipelineScreen())
        elif action == "expression":
            self.app.push_screen(ExpressionScreen())
        elif action == "validate":
            self.app.push_screen(ValidateScreen())
        elif action == "config":
            self.app.push_screen(ConfigScreen())
        elif action == "ingest":
            self.app.push_screen(IngestScreen())
        elif action == "matrix":
            self.app.push_screen(BuildMatrixScreen())
        elif action == "dataset":
            self.app.push_screen(BuildSurvivalScreen())
        elif action == "download":
            self.app.push_screen(GDCDownloadScreen())
        elif action == "manage":
            self.app.push_screen(ManageScreen())
        else:
            self.app.bell()
            self.notify(f"Nieznana opcja: {code!r}", severity="warning", title="LUADHUB")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        code = (event.item.id or "opt-").split("-", 1)[1]
        self._route(code)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value
        event.input.value = ""
        self._route(value)

    def action_help(self) -> None:
        self.notify(
            "Nawigacja: strzałki + Enter lub numer opcji w linii 'Opcja ===>'.\n"
            "PF3 = wyjście/powrót, PF5 = odśwież raport.",
            title="LUADHUB — pomoc",
        )


class LuadHubaTUI(App):
    """Aplikacja terminalowa LUAD-HUBA."""

    CSS_PATH = "theme.tcss"
    TITLE = "LUAD-HUBA"

    def on_mount(self) -> None:
        self.push_screen(PrimaryMenu())


def main() -> None:
    LuadHubaTUI().run()


if __name__ == "__main__":
    main()
