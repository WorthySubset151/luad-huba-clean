"""Renderowanie raportów survival jako tabele Rich — styl z/OS, zielono na czarnym.

Każda funkcja przyjmuje słownik danych z ``src/analysis/survival_report`` i zwraca
renderable Rich (Group/Table/Text). Logika prezentacji wyłącznie tutaj — żadnych
obliczeń. HR < 1 zielono (ochronne), HR ≥ 1 czerwono (ryzyko), jak w forest plot
dashboardu, żeby oba frontendy czytały się spójnie.
"""

__author__ = "Łukasz Połaski"

from rich import box
from rich.console import Group
from rich.table import Table
from rich.text import Text

GREEN = "#33ff66"   # fosfor — wartości
DIM = "#1f9d4d"     # przygaszona zieleń — etykiety/ramki
RISK = "#ff5f56"    # czerwień — ryzyko / brak / istotny brak
HEAD = "bold #8affb0"  # jasna zieleń — nagłówki sekcji

_BOX = box.SQUARE


def _fmt_p(p) -> str:
    if p is None:
        return "—"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def _fmt_pct(x) -> str:
    return f"{x * 100:.0f}%" if x is not None else "n/d"


# ---------------------------------------------------------------------------
#  STATUS — przegląd kohorty
# ---------------------------------------------------------------------------
def cohort_report(s: dict) -> Group:
    metrics = Table(box=_BOX, show_header=False, border_style=DIM, pad_edge=False)
    metrics.add_column(style=DIM)
    metrics.add_column(style=GREEN, justify="right")
    metrics.add_row("Próbki (tumor)", str(s["n_samples"]))
    metrics.add_row("Zdarzenia (zgony)", str(s["n_events"]))
    metrics.add_row("Cenzurowane", f'{s["n_censored"]}  ({s["censoring_pct"]:.1f}%)')
    metrics.add_row("Geny w macierzy", str(s["n_genes"]))
    metrics.add_row("Mediana obserwacji", f'{s["median_followup_years"]:.2f} lat')

    stages = Table(title="ROZKŁAD STADIÓW (AJCC)", box=_BOX,
                   border_style=DIM, title_style=HEAD)
    stages.add_column("Stadium", style=DIM)
    stages.add_column("n", style=GREEN, justify="right")
    for k in ("I", "II", "III", "IV", "Unknown"):
        if k in s["stage_counts"]:
            stages.add_row(k, str(s["stage_counts"][k]))

    return Group(
        Text("PRZEGLĄD KOHORTY TCGA-LUAD", style=HEAD),
        Text(""),
        metrics,
        Text(""),
        stages,
    )


# ---------------------------------------------------------------------------
#  SURVIVAL — Kaplan-Meier + Cox
# ---------------------------------------------------------------------------
def _km_block(km: dict) -> Group:
    head = Text("KAPLAN-MEIER — PRZEŻYCIE CAŁKOWITE (OS)", style=HEAD)
    med = f'{km["median_os"]:.2f} lat' if km["median_os"] is not None else "nieosiągnięta"
    line = Text.assemble(
        ("  Mediana OS: ", DIM), (med, GREEN),
        ("      1 / 3 / 5-letnie: ", DIM),
        (f'{_fmt_pct(km["surv_1y"])} / {_fmt_pct(km["surv_3y"])} / {_fmt_pct(km["surv_5y"])}', GREEN),
    )
    t = Table(box=_BOX, border_style=DIM)
    t.add_column("Stadium", style=DIM)
    t.add_column("n", justify="right", style=GREEN)
    t.add_column("Mediana OS [lat]", justify="right", style=GREEN)
    for r in km["per_stage"]:
        med_s = f'{r["median_os"]:.2f}' if r["median_os"] is not None else "—"
        t.add_row(r["stage"], str(r["n"]), med_s)

    p = km["logrank_p"]
    sig = p is not None and p < 0.05
    p_line = Text.assemble(
        ("  Log-rank (I/II/III/IV): p = ", DIM),
        (_fmt_p(p), GREEN if sig else RISK),
        ("   → stadium " + ("RÓŻNICUJE przeżycie" if sig else "nie różnicuje"), DIM),
    )
    return Group(head, line, Text(""), t, p_line)


def _cox_clinical_block(rep: dict) -> Group:
    head = Text("COX — KOWARIANTY KLINICZNE (wiek, płeć, stadium)", style=HEAD)
    if "error" in rep:
        return Group(head, Text("  " + rep["error"], style=RISK))
    t = Table(box=_BOX, border_style=DIM)
    t.add_column("Kowariant", style=DIM)
    t.add_column("HR", justify="right")
    t.add_column("95% CI", justify="right", style=GREEN)
    t.add_column("p", justify="right", style=GREEN)
    t.add_column("", style=DIM)
    for r in rep["rows"]:
        col = GREEN if r["hr"] < 1 else RISK
        t.add_row(
            r["label"],
            Text(f'{r["hr"]:.2f}', style=col),
            f'{r["ci_lower"]:.2f}–{r["ci_upper"]:.2f}',
            _fmt_p(r["p"]),
            "istotny" if r["p"] < 0.05 else "",
        )
    cidx = Text.assemble(
        ("  C-index: ", DIM), (f'{rep["c_index"]:.3f}', f"bold {GREEN}"),
        (f'      (n = {rep["n"]}, zdarzeń = {rep["n_events"]})', DIM),
    )
    return Group(head, t, cidx)


def _cox_genes_block(rep: dict) -> Group:
    head = Text("COX — KLINIKA + PANEL GENÓW (z-score log2 TPM)", style=HEAD)
    if "error" in rep:
        return Group(head, Text("  " + rep["error"], style=RISK))
    t = Table(box=_BOX, border_style=DIM)
    t.add_column("Gen", style=DIM)
    t.add_column("HR", justify="right")
    t.add_column("95% CI", justify="right", style=GREEN)
    t.add_column("p", justify="right", style=GREEN)
    t.add_column("Kierunek", style=DIM)
    for r in rep["rows"]:
        protective = r["hr"] < 1
        col = GREEN if protective else RISK
        t.add_row(
            r["symbol"],
            Text(f'{r["hr"]:.2f}', style=col),
            f'{r["ci_lower"]:.2f}–{r["ci_upper"]:.2f}',
            _fmt_p(r["p"]),
            Text("ochronny" if protective else "ryzyko", style=col),
        )
    d = rep["delta"]
    cmp_line = Text.assemble(
        ("  C-index  klinika: ", DIM), (f'{rep["c_index_clinical"]:.3f}', GREEN),
        ("    klinika+geny: ", DIM), (f'{rep["c_index_genes"]:.3f}', f"bold {GREEN}"),
        ("    Δ = ", DIM), (f'{d:+.3f}', f"bold {GREEN if d > 0 else RISK}"),
    )
    if d > 0.01:
        verdict = "  → panel genów dodaje sygnał prognostyczny ponad klinikę."
    elif d > 0:
        verdict = "  → panel nieznacznie poprawia predykcję."
    else:
        verdict = "  → panel nie poprawia C-index w tej kohorcie."
    parts = [head, t, cmp_line, Text(verdict, style=DIM)]
    if rep.get("missing"):
        parts.append(Text(f'  Geny pominięte (brak w macierzy): {", ".join(rep["missing"])}',
                          style=RISK))
    return Group(*parts)


def survival_report(km: dict, cox_clinical: dict, cox_genes: dict) -> Group:
    sep = Text("─" * 66, style=DIM)
    return Group(
        _km_block(km),
        Text(""), sep, Text(""),
        _cox_clinical_block(cox_clinical),
        Text(""), sep, Text(""),
        _cox_genes_block(cox_genes),
    )
