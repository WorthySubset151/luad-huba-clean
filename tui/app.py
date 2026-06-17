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
from src.pipeline_status import pipeline_status  # noqa: E402
from tui import render  # noqa: E402

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SURVIVAL_PARQUET = DATA_PROCESSED / "survival_dataset.parquet"
VERSION = "v0.1"

# (kod opcji, nazwa, opis, akcja)
MENU = [
    ("0", "STATUS", "Przegląd kohorty (próbki, zdarzenia, geny, stadia)", "status"),
    ("1", "SURVIVAL", "Analiza przeżywalności (Kaplan-Meier + Cox HR)", "survival"),
    ("2", "PIPELINE", "Status etapów ETL (zbiory danych, kompletność)", "pipeline"),
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

    def build_report(self, ds):
        km = sr.km_summary(ds)
        cox_clin = sr.cox_clinical_report(ds)
        cox_genes = sr.cox_genes_report(ds)
        return render.survival_report(km, cox_clin, cox_genes)


class PipelineScreen(ReportScreen):
    """Status etapów ETL — inspekcja dysku, działa też gdy artefaktów brak."""

    PANEL_ID = "LUADHUB.PIPE"
    TITLE_TXT = "PIPELINE — STATUS ETL"

    def action_refresh(self) -> None:
        # Nie wymaga survival_dataset — sensem panelu jest pokazać, czego brak.
        content = self.query_one("#content", Static)
        content.update(render.pipeline_report(pipeline_status(PROJECT_ROOT)))


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
