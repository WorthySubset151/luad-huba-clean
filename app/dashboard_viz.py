"""Dashboard analityczny LUAD-HUBA — wizualizacje Plotly.

Funkcje budujące wykresy z survival_dataset i expression_matrix.
Adaptacja wizualizacji z notebooków 05 (EDA) i 06 (survival) do Plotly.
"""

__author__ = "Łukasz Połaski"

import numpy as np
import plotly.graph_objects as go
import polars as pl
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Rdzeń analityczny (jedno źródło prawdy metodologii Cox/KM oraz danych panelu).
from src.analysis import survival_report as sr
from src.analysis.survival_report import SIGNATURE_PANEL, collapse_stage, find_gene_col

# Paleta spójna z notebookami (beż/brąz/zieleń)
PALETTE = {
    "primary": "#5a8b3c",    # zieleń
    "secondary": "#8b5a3c",  # brąz
    "accent": "#c4a484",     # beż
    "dark": "#2a4a1c",       # ciemna zieleń
    "stage": ["#5a8b3c", "#a8a030", "#a86a30", "#8b3c2a"],  # I->IV gradient
}

# SIGNATURE_PANEL importowany z src/analysis (jedno źródło prawdy panelu genów).

LUAD_MARKERS = {
    "EGFR":   "ENSG00000146648",
    "KRAS":   "ENSG00000133703",
    "TP53":   "ENSG00000141510",
    "ALK":    "ENSG00000171094",
    "ROS1":   "ENSG00000047936",
    "NKX2-1": "ENSG00000136352",
    "SFTPC":  "ENSG00000168484",
}

# Krótkie, precyzyjne charakterystyki genów (rola w LUAD)
GENE_INFO = {
    "EGFR": "Receptor naskórkowego czynnika wzrostu. Mutacje aktywujące (del19, L858R) "
            "częste w LUAD, zwłaszcza u niepalących — cel terapii inhibitorami TKI (gefitynib, ozymertynib).",
    "KRAS": "Onkogen szlaku RAS/MAPK. Mutacje (G12C i in.) to najczęstszy driver LUAD; "
            "wzajemnie wykluczające się z EGFR. Cel nowych inhibitorów KRAS-G12C (sotorasib).",
    "TP53": "Gen supresorowy „strażnik genomu”. Mutacje bardzo częste w LUAD, zwykle współistnieją "
            "z innymi driverami; związane z gorszym rokowaniem i niestabilnością genomową.",
    "ALK":  "Kinaza receptorowa. Onkogenna przez REARANŻACJE/FUZJE (EML4-ALK), nie przez nadekspresję "
            "— dlatego sam poziom mRNA bywa niski. Cel inhibitorów ALK (kryzotynib, alektynib).",
    "ROS1": "Kinaza receptorowa. Onkogenna przez REARANŻACJE/FUZJE (~1-2% LUAD). Mechanizm fuzyjny "
            "niewidoczny w samej ekspresji. Cel inhibitorów ROS1 (kryzotynib, entrektynib).",
    "NKX2-1": "Czynnik transkrypcyjny (TTF-1) różnicowania pęcherzykowego płuc. Wysoka ekspresja = "
              "lepiej zróżnicowany guz i LEPSZE rokowanie. Kluczowy marker diagnostyczny LUAD.",
    "SFTPC": "Białko surfaktantu C, marker komórek pęcherzykowych typu II. Wysoka ekspresja wskazuje "
             "na zróżnicowanie pęcherzykowe — zwykle korzystne rokowniczo.",
    "NAPSA": "Napsyna A, proteaza aspartylowa pneumocytów. Marker różnicowania pęcherzykowego, "
             "wspiera diagnostykę LUAD (vs rak płaskonabłonkowy).",
    "MKI67": "Ki-67, marker proliferacji (frakcja dzielących się komórek). Wysoka ekspresja = "
             "agresywniejszy, szybko rosnący guz — zwykle gorsze rokowanie.",
    "TOP2A": "Topoizomeraza II alfa, enzym replikacji DNA. Marker proliferacji; wysoka ekspresja "
             "wiąże się z agresywnością guza.",
    "BIRC5": "Surwiwina, inhibitor apoptozy z rodziny IAP. Wysoka ekspresja = unikanie śmierci "
             "komórkowej, proliferacja — niekorzystne rokowniczo.",
    "SPP1": "Osteopontyna, glikoproteina macierzy. Związana z inwazją, przerzutowaniem i angiogenezą; "
            "wysoka ekspresja zwykle niekorzystna rokowniczo.",
}

# Etykiety kolumn klinicznych (po polsku) do panelu próbki
CLINICAL_LABELS = {
    "case_id": "Pacjent (case_id)",
    "sample_id": "Próbka (sample_id)",
    "time": "Czas obserwacji (dni)",
    "event": "Status (zgon)",
    "age_at_index": "Wiek przy diagnozie",
    "gender": "Płeć",
    "ajcc_pathologic_stage": "Stadium (AJCC)",
    "tissue_type": "Typ tkanki",
}


def _ensure_time_years(pdf):
    """Zapewnia kolumnę time_years (survival_dataset ma 'time' w dniach)."""
    pdf = pdf.copy()
    if "time_years" not in pdf.columns:
        if "time" in pdf.columns:
            pdf["time_years"] = pdf["time"] / 365.25
        else:
            raise KeyError("Brak kolumny 'time' ani 'time_years' w danych przeżywalności")
    return pdf


# collapse_stage i find_gene_col importowane z src/analysis (jedno źródło prawdy).


# (find_gene_col powyżej — importowany z src/analysis)


def _layout(fig: go.Figure, title: str, xlabel: str, ylabel: str) -> go.Figure:
    """Wspólny styl layoutu wykresów."""
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_white",
        font=dict(family="sans-serif", size=13, color="#2a2a26"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(bgcolor="rgba(250,248,244,0.8)"),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.08)")
    return fig


def _km_trace(pdf, mask, label, color, dash=None):
    """Buduje ślad KM (schodkowa krzywa przeżycia + CI) dla podgrupy."""
    kmf = KaplanMeierFitter()
    kmf.fit(pdf["time_years"][mask], pdf["event"][mask], label=label)
    sf = kmf.survival_function_
    ci = kmf.confidence_interval_
    times = sf.index.values
    surv = sf.iloc[:, 0].values
    lower = ci.iloc[:, 0].values
    upper = ci.iloc[:, 1].values

    n = int(mask.sum())
    traces = []
    # Pasmo CI
    traces.append(go.Scatter(
        x=np.concatenate([times, times[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself", fillcolor=color.replace(")", ",0.12)").replace("rgb", "rgba")
            if color.startswith("rgb") else _hex_to_rgba(color, 0.12),
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))
    # Krzywa schodkowa
    traces.append(go.Scatter(
        x=times, y=surv, mode="lines", name=f"{label} (n={n})",
        line=dict(color=color, width=2.5, shape="hv", dash=dash),
        hovertemplate="%{y:.2f}<extra></extra>",
    ))
    return traces, kmf


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# =====================================================================
#  WYKRESY PRZEŻYWALNOŚCI
# =====================================================================
def km_overall(pdf) -> tuple[go.Figure, dict]:
    """KM dla całej kohorty + statystyki (mediana OS, przeżycie 1/3/5-letnie)."""
    pdf = _ensure_time_years(pdf)
    fig = go.Figure()
    traces, kmf = _km_trace(pdf, np.ones(len(pdf), dtype=bool),
                            "Cała kohorta", PALETTE["primary"])
    for t in traces:
        fig.add_trace(t)

    median_os = kmf.median_survival_time_
    stats = {"median_os": float(median_os) if not np.isnan(median_os) else None}
    for years in [1, 3, 5]:
        try:
            stats[f"surv_{years}y"] = float(kmf.survival_function_at_times(years).iloc[0])
        except Exception:
            stats[f"surv_{years}y"] = None

    # Linia mediany
    if stats["median_os"]:
        fig.add_hline(y=0.5, line_dash="dash", line_color=PALETTE["secondary"], opacity=0.4)
        fig.add_vline(x=median_os, line_dash="dash", line_color=PALETTE["secondary"], opacity=0.4)
        fig.add_annotation(x=median_os, y=0.55, text=f"Mediana OS: {median_os:.2f} lat",
                           showarrow=False, font=dict(color=PALETTE["dark"], size=12),
                           xanchor="left", xshift=8)

    _layout(fig, "Kaplan-Meier — cała kohorta TCGA-LUAD", "Czas (lata)",
            "Prawdopodobieństwo przeżycia")
    fig.update_yaxes(range=[0, 1.02])
    return fig, stats


def km_per_stage(pdf) -> tuple[go.Figure, dict]:
    """KM rozbite per stadium (collapse do 4 grup) + log-rank."""
    pdf = _ensure_time_years(pdf)
    pdf["stage_group"] = pdf["ajcc_pathologic_stage"].apply(collapse_stage)

    fig = go.Figure()
    stage_order = ["I", "II", "III", "IV"]
    present = [s for s in stage_order if (pdf["stage_group"] == s).any()]

    for i, stage in enumerate(present):
        mask = (pdf["stage_group"] == stage).values
        if mask.sum() < 2:
            continue
        traces, _ = _km_trace(pdf, mask, f"Stage {stage}", PALETTE["stage"][i])
        for t in traces:
            fig.add_trace(t)

    # Log-rank test (multivariate przez pary - tu uproszczone: I vs IV jako sygnał)
    from lifelines.statistics import multivariate_logrank_test
    mask_known = pdf["stage_group"].isin(stage_order).values
    p_value = None
    if mask_known.sum() > 10:
        try:
            res = multivariate_logrank_test(
                pdf["time_years"][mask_known],
                pdf["stage_group"][mask_known],
                pdf["event"][mask_known],
            )
            p_value = float(res.p_value)
        except Exception:
            p_value = None

    _layout(fig, "Kaplan-Meier per stadium (grupy główne)", "Czas (lata)",
            "Prawdopodobieństwo przeżycia")
    fig.update_yaxes(range=[0, 1.02])
    return fig, {"p_value": p_value, "stages": present}


def signature_score(ds, pdf) -> np.ndarray:
    """Liczy signature score (znakowane z-score log2 TPM panelu) - jak nb 06."""
    gene_cols = [c for c in ds.columns if c.startswith("ENSG")]
    score = np.zeros(ds.height)
    found = 0
    for symbol, (ensg, sign) in SIGNATURE_PANEL.items():
        col = find_gene_col(ensg, gene_cols)
        if col is None:
            continue
        expr = ds[col].to_numpy().astype(float)
        expr_log = np.log2(expr + 1)
        std = expr_log.std()
        if std > 0:
            z = (expr_log - expr_log.mean()) / std
            score += sign * z
            found += 1
    return score, found


def km_signature(ds, pdf) -> tuple[go.Figure, dict]:
    """KM dla sygnatury wielogenowej (high/low względem mediany score)."""
    score, n_genes = signature_score(ds, pdf)
    pdf = _ensure_time_years(pdf)
    pdf["sig_score"] = score
    median = np.median(score)
    pdf["sig_group"] = np.where(score >= median, "high", "low")

    fig = go.Figure()
    high_mask = (pdf["sig_group"] == "high").values
    low_mask = (pdf["sig_group"] == "low").values

    traces_low, _ = _km_trace(pdf, low_mask, "Profil łagodny (low)", PALETTE["primary"])
    traces_high, _ = _km_trace(pdf, high_mask, "Profil agresywny (high)", PALETTE["secondary"])
    for t in traces_low + traces_high:
        fig.add_trace(t)

    p_value = None
    try:
        res = logrank_test(pdf["time_years"][high_mask], pdf["time_years"][low_mask],
                           pdf["event"][high_mask], pdf["event"][low_mask])
        p_value = float(res.p_value)
    except Exception:
        pass

    _layout(fig, "Kaplan-Meier — sygnatura wielogenowa (podział: mediana)",
            "Czas (lata)", "Prawdopodobieństwo przeżycia")
    fig.update_yaxes(range=[0, 1.02])
    return fig, {"p_value": p_value, "n_genes": n_genes}


def km_single_gene(ds, pdf, gene_symbol: str, ensg: str) -> tuple[go.Figure, dict]:
    """KM dla pojedynczego genu (high/low względem mediany ekspresji)."""
    gene_cols = [c for c in ds.columns if c.startswith("ENSG")]
    col = find_gene_col(ensg, gene_cols)
    if col is None:
        return None, {"error": f"Gen {gene_symbol} ({ensg}) nie znaleziony w macierzy"}

    expr = ds[col].to_numpy().astype(float)
    pdf = _ensure_time_years(pdf)
    pdf["expr"] = expr
    median = np.median(expr)
    pdf["expr_group"] = np.where(expr >= median, "high", "low")

    fig = go.Figure()
    high_mask = (pdf["expr_group"] == "high").values
    low_mask = (pdf["expr_group"] == "low").values

    traces_high, _ = _km_trace(pdf, high_mask, f"{gene_symbol} high", PALETTE["primary"])
    traces_low, _ = _km_trace(pdf, low_mask, f"{gene_symbol} low", PALETTE["secondary"])
    for t in traces_high + traces_low:
        fig.add_trace(t)

    p_value = None
    try:
        res = logrank_test(pdf["time_years"][high_mask], pdf["time_years"][low_mask],
                           pdf["event"][high_mask], pdf["event"][low_mask])
        p_value = float(res.p_value)
    except Exception:
        pass

    _layout(fig, f"Kaplan-Meier — {gene_symbol} high vs low (podział: mediana)",
            "Czas (lata)", "Prawdopodobieństwo przeżycia")
    fig.update_yaxes(range=[0, 1.02])
    return fig, {"p_value": p_value, "median_expr": float(median)}


def km_multi_gene(ds, pdf, genes: list[tuple[str, str]]) -> tuple[go.Figure, list]:
    """Porównuje wiele genów na jednym wykresie KM.

    Dla każdego genu rysuje krzywą grupy "high" (ekspresja >= mediana),
    pozwalając wizualnie porównać prognostyczny efekt różnych genów.
    Zwraca też tabelę log-rank (high vs low) per gen.

    Argumenty:
        genes: lista par (symbol, ensg) genów do porównania.
    """
    pdf = _ensure_time_years(pdf)
    gene_cols = [c for c in ds.columns if c.startswith("ENSG")]

    # Paleta dla wielu genów (cyklicznie)
    multi_colors = [
        PALETTE["primary"], PALETTE["secondary"], "#a8a030",
        "#3c6a8b", "#8b3c6a", "#6a8b3c", "#c4844a",
    ]

    fig = go.Figure()
    results = []
    color_idx = 0

    for symbol, ensg in genes:
        col = find_gene_col(ensg, gene_cols)
        if col is None:
            results.append({"gene": symbol, "p_value": None, "note": "brak w macierzy"})
            continue

        expr = ds[col].to_numpy().astype(float)
        median = np.median(expr)
        high_mask = (expr >= median)
        low_mask = (expr < median)

        color = multi_colors[color_idx % len(multi_colors)]
        color_idx += 1

        # Rysujemy tylko krzywą "high" (porównanie efektu między genami)
        kmf = KaplanMeierFitter()
        kmf.fit(pdf["time_years"][high_mask], pdf["event"][high_mask], label=f"{symbol} high")
        sf = kmf.survival_function_
        n_high = int(high_mask.sum())
        fig.add_trace(go.Scatter(
            x=sf.index.values, y=sf.iloc[:, 0].values, mode="lines",
            name=f"{symbol} high (n={n_high})",
            line=dict(color=color, width=2.5, shape="hv"),
            hovertemplate=f"{symbol}: %{{y:.2f}}<extra></extra>",
        ))

        # Log-rank high vs low dla tego genu
        p_value = None
        try:
            res = logrank_test(pdf["time_years"][high_mask], pdf["time_years"][low_mask],
                               pdf["event"][high_mask], pdf["event"][low_mask])
            p_value = float(res.p_value)
        except Exception:
            pass
        results.append({"gene": symbol, "p_value": p_value, "note": ""})

    _layout(fig, "Kaplan-Meier — porównanie genów (krzywe grup „high”)",
            "Czas (lata)", "Prawdopodobieństwo przeżycia")
    fig.update_yaxes(range=[0, 1.02])
    return fig, results


# =====================================================================
#  MODEL COXA (proportional hazards) — tylko render Plotly
# =====================================================================
# Obliczenia (encoding, fitowanie, C-index) w src/analysis/survival_report.py.
# Tu zostaje wyłącznie forest plot Plotly + formatowanie tabel do st.dataframe.


def _forest_plot(rows, title) -> go.Figure:
    """Wspólny forest plot hazard ratios (oś log, CI, linia odniesienia HR=1).

    Argumenty:
        rows: lista krotek (label, hr, ci_lower, ci_upper, p_value).
    Kolor po kierunku efektu: HR<1 zieleń (ochronne), HR>=1 brąz (ryzyko).
    Oś X w skali logarytmicznej - standard dla ilorazów (CI symetryczne w log).
    """
    fig = go.Figure()
    for label, hr, lo, hi, p in rows:
        color = PALETTE["primary"] if hr < 1 else PALETTE["secondary"]
        # Pasmo CI jako pozioma linia
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[label, label], mode="lines",
            line=dict(color=color, width=2),
            hoverinfo="skip", showlegend=False,
        ))
        # Punkt HR
        p_txt = f"{p:.3g}" if p is not None else "—"
        fig.add_trace(go.Scatter(
            x=[hr], y=[label], mode="markers",
            marker=dict(color=color, size=11, line=dict(color="white", width=1)),
            hovertemplate=(f"{label}<br>HR = {hr:.2f} "
                           f"(95% CI {lo:.2f}–{hi:.2f})<br>p = {p_txt}<extra></extra>"),
            showlegend=False,
        ))
    # Linia odniesienia HR=1 (brak efektu)
    fig.add_vline(x=1.0, line_dash="dash", line_color="grey", opacity=0.6)
    # Atrapy do legendy (objaśnienie kolorów)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(color=PALETTE["primary"], size=10),
                             name="HR < 1 (ochronne)"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(color=PALETTE["secondary"], size=10),
                             name="HR ≥ 1 (ryzyko)"))

    _layout(fig, title, "Hazard Ratio (95% CI, skala log)", "")
    fig.update_xaxes(type="log")
    fig.update_yaxes(autorange="reversed")  # pierwszy wiersz u góry
    fig.update_layout(hovermode="closest")
    return fig


def cox_clinical(ds) -> tuple[go.Figure, dict]:
    """Model Coxa na kowariantach klinicznych: wiek + płeć + stadium.

    Liczby pochodzą z src/analysis (jedno źródło prawdy); tu tylko forest plot
    Plotly + tabela do st.dataframe. Wejście: polars DataFrame survival_dataset.
    Zwraca (fig, info) gdzie info = {c_index, n, n_events, table},
    lub (None, {"error": ...}).
    """
    rep = sr.cox_clinical_report(ds)
    if "error" in rep:
        return None, rep

    rows_fp, table = [], []
    for r in rep["rows"]:
        rows_fp.append((r["label"], r["hr"], r["ci_lower"], r["ci_upper"], r["p"]))
        table.append({
            "Kowariant": r["label"],
            "HR": f"{r['hr']:.2f}",
            "95% CI": f"{r['ci_lower']:.2f}–{r['ci_upper']:.2f}",
            "p": f"{r['p']:.3g}",
            "Istotność": "istotny" if r["p"] < 0.05 else "nieistotny",
        })

    fig = _forest_plot(rows_fp, "Cox kliniczny — hazard ratios (wiek, płeć, stadium)")
    return fig, {
        "c_index": rep["c_index"],
        "n": rep["n"],
        "n_events": rep["n_events"],
        "table": table,
    }


def cox_clinical_genes(ds) -> tuple[go.Figure, dict]:
    """Multivariate Cox: kowarianty kliniczne + panel genów (z-score log2 TPM).

    Liczby z src/analysis (dwa modele na tej samej kohorcie -> uczciwe
    porównanie C-index). Tu tylko forest plot genów + tabela. Wejście: polars
    DataFrame survival_dataset. Zwraca (fig, info) gdzie info =
    {c_index_clinical, c_index_genes, delta, n, gene_table, missing},
    lub (None, {"error": ...}).
    """
    rep = sr.cox_genes_report(ds)
    if "error" in rep:
        return None, rep

    rows_fp, table = [], []
    for r in rep["rows"]:
        rows_fp.append((r["symbol"], r["hr"], r["ci_lower"], r["ci_upper"], r["p"]))
        table.append({
            "Gen": r["symbol"],
            "HR": f"{r['hr']:.2f}",
            "95% CI": f"{r['ci_lower']:.2f}–{r['ci_upper']:.2f}",
            "p": f"{r['p']:.3g}",
            "Kierunek": "ochronny (HR<1)" if r["hr"] < 1 else "ryzyko (HR>1)",
        })

    fig = _forest_plot(
        rows_fp, "Multivariate Cox — hazard ratios genów panelu (po korekcie o klinikę)")
    return fig, {
        "c_index_clinical": rep["c_index_clinical"],
        "c_index_genes": rep["c_index_genes"],
        "delta": rep["delta"],
        "n": rep["n"],
        "gene_table": table,
        "missing": rep["missing"],
    }


# =====================================================================
#  WYKRESY EKSPRESJI
# =====================================================================
def histogram_tpm(matrix: pl.DataFrame, sample_col: str) -> go.Figure:
    """Histogram log2(TPM+1) dla jednej próbki - rozkład ekspresji."""
    expr = matrix[sample_col].to_numpy().astype(float)
    log_expr = np.log2(expr + 1)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=log_expr, nbinsx=80,
                               marker_color=PALETTE["primary"], opacity=0.85,
                               name="log2(TPM+1)"))
    _layout(fig, f"Rozkład ekspresji log2(TPM+1) — próbka {sample_col[:20]}",
            "log2(TPM+1)", "Liczba genów")
    fig.update_layout(hovermode="x")
    return fig


def markers_expression(matrix: pl.DataFrame, sample_cols: list) -> go.Figure:
    """Ekspresja markerów LUAD uśredniona po próbkach (boxplot per marker)."""
    gene_ids = matrix["gene_id"].to_list()
    fig = go.Figure()

    for symbol, ensg in LUAD_MARKERS.items():
        col_idx = None
        for i, gid in enumerate(gene_ids):
            if gid.startswith(ensg + ".") or gid == ensg:
                col_idx = i
                break
        if col_idx is None:
            continue
        # Ekspresja tego genu po wszystkich próbkach
        row = matrix.row(col_idx)
        # kolumny próbek (pomijamy gene_id na idx 0)
        vals = np.array([row[matrix.columns.index(c)] for c in sample_cols], dtype=float)
        log_vals = np.log2(vals + 1)
        fig.add_trace(go.Box(y=log_vals, name=symbol, marker_color=PALETTE["primary"],
                             line_color=PALETTE["dark"], boxpoints=False))

    _layout(fig, "Ekspresja markerów LUAD (log2 TPM, rozkład po próbkach)",
            "Marker", "log2(TPM+1)")
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig
