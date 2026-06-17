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
    head = Text("Kaplan-Meier — cała kohorta", style=HEAD)
    med = f'{km["median_os"]:.2f} lat' if km["median_os"] is not None else "nieosiągnięta"
    line = Text.assemble(
        ("  Mediana OS: ", DIM), (med, GREEN),
        ("      1 / 3 / 5-letnie: ", DIM),
        (f'{_fmt_pct(km["surv_1y"])} / {_fmt_pct(km["surv_3y"])} / {_fmt_pct(km["surv_5y"])}', GREEN),
    )
    t = Table(box=_BOX, border_style=DIM, title="Kaplan-Meier — per stadium",
              title_style=HEAD)
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
    head = Text("Model Coxa — kowarianty kliniczne", style=HEAD)
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
    head = Text("Model Coxa — klinika + panel genów", style=HEAD)
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


def _signature_block(rep: dict) -> Group:
    head = Text("Kaplan-Meier — sygnatura wielogenowa", style=HEAD)
    if "error" in rep:
        return Group(head, Text("  " + rep["error"], style=RISK))
    t = Table(box=_BOX, border_style=DIM)
    t.add_column("Profil", style=DIM)
    t.add_column("n", justify="right", style=GREEN)
    t.add_column("Mediana OS [lat]", justify="right", style=GREEN)
    for key, color in (("high", RISK), ("low", GREEN)):
        g = rep[key]
        med = f'{g["median_os"]:.2f}' if g["median_os"] is not None else "—"
        t.add_row(Text(g["label"], style=color), str(g["n"]), med)
    p = rep["logrank_p"]
    sig = p is not None and p < 0.05
    p_line = Text.assemble(
        ("  Log-rank (high vs low): p = ", DIM),
        (_fmt_p(p), GREEN if sig else RISK),
        (f'   panel {rep["n_genes"]} genów   → ', DIM),
        ("różnicuje przeżycie" if sig else "nie różnicuje", DIM),
    )
    return Group(head, t, p_line)


def survival_report(km: dict, cox_clinical: dict, signature: dict, cox_genes: dict) -> Group:
    sep = Text("─" * 66, style=DIM)
    return Group(
        _km_block(km),
        Text(""), sep, Text(""),
        _cox_clinical_block(cox_clinical),
        Text(""), sep, Text(""),
        _signature_block(signature),
        Text(""), sep, Text(""),
        _cox_genes_block(cox_genes),
    )


# ---------------------------------------------------------------------------
#  PIPELINE — status etapów ETL (lista zbiorów w stylu z/OS)
# ---------------------------------------------------------------------------
def _fmt_size(n) -> str:
    if n is None:
        return "—"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def pipeline_report(status: dict) -> Group:
    done, total = status["stages_done"], status["stages_total"]
    head = Text.assemble(
        ("STATUS PIPELINE'U ETL — TCGA-LUAD", HEAD),
        (f"      etapy gotowe: {done}/{total}", DIM),
    )

    stages = Table(box=_BOX, border_style=DIM, title="ETAPY", title_style=HEAD)
    stages.add_column("Status", justify="left")
    stages.add_column("Etap", style=DIM)
    stages.add_column("Szczegóły", style=DIM)
    for s in status["stages"]:
        if s["status"] == "ok":
            badge = Text("✓ GOTOWY", style=f"bold {GREEN}")
        elif s["status"] == "missing":
            badge = Text("✗ BRAK", style=f"bold {RISK}")
        else:
            badge = Text(f"⊘ czeka: {s['blocked_by']}", style=DIM)
        stages.add_row(badge, s["label"], s["detail"])

    datasets = Table(box=_BOX, border_style=DIM, title="ZBIORY DANYCH", title_style=HEAD)
    datasets.add_column("Nazwa zbioru", style=DIM)
    datasets.add_column("", justify="center")
    datasets.add_column("Rekordy", justify="right", style=GREEN)
    datasets.add_column("Rozmiar", justify="right", style=GREEN)
    datasets.add_column("Zmodyfikowano", justify="right", style=GREEN)
    for a in status["artifacts"]:
        present = Text("✓", style=GREEN) if a["exists"] else Text("✗", style=RISK)
        if a.get("note") == "nieczytelny parquet":
            rec = Text("nieczytelny", style=RISK)
        elif a.get("rows"):
            rec = f'{a["rows"]:,}×{a["cols"]}'
        elif "count" in a:
            rec = f'{a["count"]} plik.' if a["exists"] else "—"
        elif a["exists"]:
            rec = "plik"
        else:
            rec = "—"
        datasets.add_row(a["name"], present, rec, _fmt_size(a["size"]), a["mtime"] or "—")

    root_line = Text(
        f'  root: {status["root"]}     config: ' + ("✓" if status["config_exists"] else "✗"),
        style=DIM,
    )
    return Group(head, Text(""), stages, Text(""), datasets, Text(""), root_line)


# ---------------------------------------------------------------------------
#  EXPRESSION — podsumowanie macierzy ekspresji
# ---------------------------------------------------------------------------
def expression_report(s: dict) -> Group:
    if "error" in s:
        return Group(Text("MACIERZ EKSPRESJI", style=HEAD), Text("  " + s["error"], style=RISK))

    metric = s["distribution"]["metric"]
    biotyp = (f"filtr biotypów zastosowany (pełny GENCODE ≈ {s['gencode_total']:,})"
              if s["biotype_filtered"]
              else f"bez filtra biotypów (≈ pełny GENCODE {s['gencode_total']:,})")
    head = Text.assemble(
        ("MACIERZ EKSPRESJI — TCGA-LUAD", HEAD),
        (f"      {s['n_genes']:,} genów × {s['n_samples']} próbek", DIM),
    )
    sub = Text(f"  {biotyp}     metryka wykryta z danych: {metric}", style=DIM)

    d = s["distribution"]
    dist = Table(box=_BOX, border_style=DIM, title="ROZKŁAD WARTOŚCI",
                 title_style=HEAD, show_header=False)
    dist.add_column(style=DIM)
    dist.add_column(style=GREEN, justify="right")
    dist.add_row("Mediana", f"{d['median']:.2f}")
    dist.add_row("IQR (p25–p75)", f"{d['p25']:.2f} – {d['p75']:.2f}")
    dist.add_row("Maksimum", f"{d['max']:,.0f}")
    dist.add_row("Wartości zerowe", f"{d['zero_pct']:.1f}%")
    dist.add_row("Mediana głębokości (suma/próbkę)", f"{d['median_depth']:,.0f}")

    tss = Table(box=_BOX, border_style=DIM, title_style=HEAD,
                title=f"BATCH — OŚRODKI TSS ({s['parsed_tss']}/{s['n_tss']} z barkodu)")
    tss.add_column("TSS", style=DIM)
    tss.add_column("Próbki", justify="right", style=GREEN)
    tss.add_column("%", justify="right", style=GREEN)
    rows = s["tss_rows"]
    cap = 20
    for r in rows[:cap]:
        tss.add_row(r["tss"], str(r["n"]), f"{r['pct']:.1f}%")
    if len(rows) > cap:
        rest = sum(r["n"] for r in rows[cap:])
        tss.add_row(f"… pozostałe ({len(rows) - cap})", str(rest),
                    f"{100.0 * rest / s['n_samples']:.1f}%")

    pca = Table(box=_BOX, border_style=DIM, title_style=HEAD,
                title=f"PCA (top {s['pca_top_n']} zmiennych genów, log2 + z-score)")
    pca.add_column("Składowa", style=DIM)
    pca.add_column("% wariancji", justify="right", style=GREEN)
    if s["pcs"]:
        for p in s["pcs"]:
            pca.add_row(f"PC{p['pc']}", f"{p['var_pct']:.1f}%")
    else:
        pca.add_row("—", "nie policzono")

    mk = Table(box=_BOX, border_style=DIM, title="MARKERY LUAD", title_style=HEAD)
    mk.add_column("Gen", style=DIM)
    mk.add_column(f"Mediana ({metric})", justify="right", style=GREEN)
    mk.add_column("Mediana log2", justify="right", style=GREEN)
    for m in s["markers"]:
        if m["found"]:
            mk.add_row(m["symbol"], f"{m['median']:.2f}", f"{m['median_log2']:.2f}")
        else:
            mk.add_row(m["symbol"], Text("brak w macierzy", style=RISK), "—")

    return Group(head, sub, Text(""), dist, Text(""), tss, Text(""), pca, Text(""), mk)
