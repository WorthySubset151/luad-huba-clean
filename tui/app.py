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
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual.containers import Horizontal, Vertical, VerticalScroll  # noqa: E402
from textual.screen import Screen  # noqa: E402
from textual.widgets import Footer, Input, Label, ListItem, ListView, Static  # noqa: E402

from src.analysis import survival_report as sr  # noqa: E402
from src.analysis.expression_report import expression_summary, LUAD_MARKERS  # noqa: E402
from src.pipeline_status import pipeline_status  # noqa: E402
from src.validate.runner import run_cohort_qc, discover_stems  # noqa: E402
from src.validate.report_view import classify_qc  # noqa: E402
from src.ingest.sample_sheet_parser import parse_sample_sheet  # noqa: E402
from src.ingest.clinical_parser import parse_clinical  # noqa: E402
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
    ("0", "STATUS", "Przegląd kohorty (próbki, zdarzenia, geny, stadia)", "status"),
    ("1", "SURVIVAL", "Analiza przeżywalności (Kaplan-Meier + Cox HR)", "survival"),
    ("2", "PIPELINE", "Status etapów ETL (zbiory danych, kompletność)", "pipeline"),
    ("3", "EXPRESSION", "Macierz ekspresji (rozkład, batch TSS, PCA)", "expression"),
    ("4", "VALIDATE", "Walidacja kohorty (QC — spójność próbek/klinika)", "validate"),
    ("5", "CONFIG", "Konfiguracja pipeline'u (podgląd configs/default.yaml)", "config"),
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
        return render.survival_report(km, cox_clin, signature, single, multi, cox_genes)


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
