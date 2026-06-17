"""Headless analiza przeżywalności LUAD-HUBA — czyste dane (bez wizualizacji).

Warstwa wspólna dla frontendów: ``app/dashboard_viz.py`` owija te dane w wykresy
Plotly, a ``tui/`` renderuje je jako tabele w stylu z/OS. Cała metodologia
Kaplan-Meier / Cox proportional hazards mieszka tutaj — jedno źródło prawdy.

To warstwa pandas + lifelines (statystyka), w odróżnieniu od polars ETL w
``src/transform``. Wejście wszędzie to polars DataFrame ``survival_dataset``
(kolumny: ``sample_id``, ``case_id``, ``time`` [dni], ``event``, ``age_at_index``,
``gender``, ``ajcc_pathologic_stage`` oraz geny ``ENSG...``). Konwersja do pandas
odbywa się wewnątrz funkcji.

Funkcje raportujące zwracają słownik z surowymi liczbami (do dowolnego
renderowania) albo ``{"error": "..."}`` gdy danych jest za mało lub model się nie
dopasował — frontend decyduje jak to pokazać.

Metodologia (zgodna z notebookiem 06):
- endpoint OS, czas w latach (z dni / 365.25),
- collapse stadium do 4 grup I/II/III/IV (NOS/Unknown odrzucone w modelu Coxa,
  bo stadium wchodzi jako zmienna uporządkowana),
- kowarianty kliniczne: wiek (per rok), płeć (męska = 1), stadium (per poziom),
- panel genów a priori jako z-score log2(TPM+1), każdy gen osobnym predyktorem
  (kierunek HR uczy się model — znak panelu służy tylko sygnaturze KM, nie tu),
- bez penalizera (artefakty kolinearności pokazywane świadomie).
"""

__author__ = "Łukasz Połaski"

import numpy as np
import polars as pl
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

# Główne stadia użyte w modelu Coxa (uporządkowalne). NOS/Unknown odrzucone.
MAIN_STAGES = ["I", "II", "III", "IV"]
STAGE_NUMERIC_MAP = {"I": 1, "II": 2, "III": 3, "IV": 4}

# Czytelne etykiety kowariantów klinicznych.
COX_CLINICAL_LABELS = {
    "age_at_index": "Wiek (per rok)",
    "gender_male": "Płeć męska",
    "stage_numeric": "Stadium (per poziom)",
}

# Panel ekspresyjny a priori: różnicowanie pęcherzykowe (znak -1, oczekiwane
# ochronne) vs proliferacja/inwazja (znak +1, oczekiwane ryzyko). Znak używany
# tylko w sygnaturze KM; w Coxie każdy gen wchodzi jako surowy z-score.
SIGNATURE_PANEL = {
    "NKX2-1": ("ENSG00000136352", -1),
    "NAPSA":  ("ENSG00000131400", -1),
    "SFTPC":  ("ENSG00000168484", -1),
    "MKI67":  ("ENSG00000148773", +1),
    "TOP2A":  ("ENSG00000131747", +1),
    "BIRC5":  ("ENSG00000089685", +1),
    "SPP1":   ("ENSG00000118785", +1),
}


# ---------------------------------------------------------------------------
#  Pomocnicze
# ---------------------------------------------------------------------------
def collapse_stage(stage) -> str:
    """Collapse szczegółowych stadiów AJCC do 4 grup głównych + Unknown."""
    if stage is None:
        return "Unknown"
    s = str(stage).replace("Stage ", "")
    if s in ("IV", "IVA", "IVB"):
        return "IV"
    if s.startswith("III"):
        return "III"
    if s.startswith("II"):
        return "II"
    if s.startswith("I"):
        return "I"
    return "Unknown"


def find_gene_col(ensg_base: str, gene_cols: list) -> str | None:
    """Znajduje kolumnę genu po prefiksie ENSG (pliki GDC mają sufiks wersji)."""
    for c in gene_cols:
        if c.startswith(ensg_base + ".") or c == ensg_base:
            return c
    return None


def _encode_clinical(pdf):
    """Dokłada time_years + zakodowane kowarianty kliniczne (bez filtrowania).

    Zwraca pełną ramkę pandas z dodatkowymi kolumnami: time_years, stage_group,
    gender_male, stage_numeric. Indeks 0..n-1 (zgodny z porządkiem wierszy ds),
    co pozwala dokładać kolumny genów pozycyjnie.
    """
    out = pdf.copy()
    if "time_years" not in out.columns:
        out["time_years"] = out["time"] / 365.25
    out["stage_group"] = out["ajcc_pathologic_stage"].apply(collapse_stage)
    out["gender_male"] = (out["gender"] == "male").astype(int)
    out["stage_numeric"] = out["stage_group"].map(STAGE_NUMERIC_MAP)
    return out


# ---------------------------------------------------------------------------
#  Raporty (czyste dane)
# ---------------------------------------------------------------------------
def cohort_summary(ds: pl.DataFrame) -> dict:
    """Podstawowe liczby kohorty: próbki, zdarzenia, cenzura, geny, stadia."""
    n = ds.height
    n_events = int(ds["event"].sum()) if "event" in ds.columns else 0
    gene_cols = [c for c in ds.columns if c.startswith("ENSG")]
    pdf = _encode_clinical(ds.to_pandas())
    stage_counts = {k: int(v) for k, v in pdf["stage_group"].value_counts().to_dict().items()}
    median_days = float(pdf["time"].median()) if n else 0.0
    return {
        "n_samples": n,
        "n_events": n_events,
        "n_censored": n - n_events,
        "censoring_pct": (100.0 * (n - n_events) / n) if n else 0.0,
        "n_genes": len(gene_cols),
        "median_followup_years": median_days / 365.25,
        "stage_counts": stage_counts,
    }


def _km_fit(time_years, event) -> tuple[dict, dict]:
    """Dopasowuje KM raz; zwraca (statystyki, krzywą z CI) z tego samego fitu."""
    kmf = KaplanMeierFitter()
    kmf.fit(time_years, event)
    sf = kmf.survival_function_
    ci = kmf.confidence_interval_
    curve = {
        "timeline": [float(x) for x in sf.index.values],
        "surv": [float(x) for x in sf.iloc[:, 0].values],
        "ci_lower": [float(x) for x in ci.iloc[:, 0].values],
        "ci_upper": [float(x) for x in ci.iloc[:, 1].values],
    }
    median = kmf.median_survival_time_
    stats = {"median_os": float(median) if not np.isnan(median) else None}
    for y in (1, 3, 5):
        try:
            stats[f"surv_{y}y"] = float(kmf.survival_function_at_times(y).iloc[0])
        except Exception:
            stats[f"surv_{y}y"] = None
    return stats, curve


def km_report(ds: pl.DataFrame) -> dict:
    """Pełny raport KM: statystyki + punkty krzywych (z CI) + log-rank.

    Jedno źródło dopasowań KM dla overall i per-stadium — GUI rysuje z tego
    krzywe (Plotly), terminal czyta podsumowanie przez km_summary. Zwraca:
    {overall: {stats, curve}, per_stage: [{stage, n, median_os, curve}],
    logrank_p, stages_present}.
    """
    pdf = _encode_clinical(ds.to_pandas())
    overall_stats, overall_curve = _km_fit(pdf["time_years"], pdf["event"])

    per_stage = []
    for s in MAIN_STAGES:
        mask = pdf["stage_group"] == s
        if mask.sum() < 2:
            continue
        st_stats, st_curve = _km_fit(pdf.loc[mask, "time_years"], pdf.loc[mask, "event"])
        per_stage.append({
            "stage": s,
            "n": int(mask.sum()),
            "median_os": st_stats["median_os"],
            "curve": st_curve,
        })

    logrank_p = None
    mask_known = pdf["stage_group"].isin(MAIN_STAGES)
    if mask_known.sum() > 10:
        try:
            res = multivariate_logrank_test(
                pdf.loc[mask_known, "time_years"],
                pdf.loc[mask_known, "stage_group"],
                pdf.loc[mask_known, "event"],
            )
            logrank_p = float(res.p_value)
        except Exception:
            logrank_p = None

    return {
        "overall": {"stats": overall_stats, "curve": overall_curve},
        "per_stage": per_stage,
        "logrank_p": logrank_p,
        "stages_present": [p["stage"] for p in per_stage],
    }


def km_summary(ds: pl.DataFrame) -> dict:
    """Podsumowanie KM (widok km_report) — kontrakt dla terminala.

    {median_os, surv_1y/3y/5y, per_stage[{stage,n,median_os}], logrank_p}.
    Liczby pochodzą z tego samego dopasowania co krzywe w GUI (km_report).
    """
    rep = km_report(ds)
    out = dict(rep["overall"]["stats"])
    out["per_stage"] = [
        {"stage": p["stage"], "n": p["n"], "median_os": p["median_os"]}
        for p in rep["per_stage"]
    ]
    out["logrank_p"] = rep["logrank_p"]
    return out


def signature_score(ds: pl.DataFrame) -> tuple[np.ndarray, int]:
    """Signature score: znakowany z-score log2(TPM+1) panelu (jak nb 06).

    Różnicowanie (znak -1) odejmuje, proliferacja/inwazja (znak +1) dodaje —
    wyższy score = profil bardziej agresywny. Zwraca (score, liczba_genów).
    """
    gene_cols = [c for c in ds.columns if c.startswith("ENSG")]
    score = np.zeros(ds.height)
    found = 0
    for _symbol, (ensg, sign) in SIGNATURE_PANEL.items():
        col = find_gene_col(ensg, gene_cols)
        if col is None:
            continue
        expr_log = np.log2(ds[col].to_numpy().astype(float) + 1)
        std = expr_log.std()
        if std > 0:
            z = (expr_log - expr_log.mean()) / std
            score += sign * z
            found += 1
    return score, found


def _km_split_report(pdf, score, label_high: str, label_low: str) -> dict:
    """KM dla podziału high/low względem mediany score + log-rank.

    Jedno dopasowanie KM na grupę (z krzywą i statystykami), więc GUI i terminal
    czytają to samo. Zwraca {median_split, logrank_p, high:{...}, low:{...}}.
    """
    median = float(np.median(score))
    high = score >= median
    low = ~high
    high_stats, high_curve = _km_fit(pdf.loc[high, "time_years"], pdf.loc[high, "event"])
    low_stats, low_curve = _km_fit(pdf.loc[low, "time_years"], pdf.loc[low, "event"])
    logrank_p = None
    try:
        res = logrank_test(pdf.loc[high, "time_years"], pdf.loc[low, "time_years"],
                           pdf.loc[high, "event"], pdf.loc[low, "event"])
        logrank_p = float(res.p_value)
    except Exception:
        logrank_p = None
    return {
        "median_split": median,
        "logrank_p": logrank_p,
        "high": {"label": label_high, "n": int(high.sum()),
                 "median_os": high_stats["median_os"], "stats": high_stats, "curve": high_curve},
        "low": {"label": label_low, "n": int(low.sum()),
                "median_os": low_stats["median_os"], "stats": low_stats, "curve": low_curve},
    }


def signature_km_report(ds: pl.DataFrame) -> dict:
    """KM sygnatury wielogenowej (high/low względem mediany score).

    Zwraca raport _km_split_report + n_genes, albo {'error': ...} gdy panelu
    nie ma w macierzy.
    """
    score, n_genes = signature_score(ds)
    if n_genes == 0:
        return {"error": "Żaden gen panelu nie został znaleziony w macierzy."}
    pdf = _encode_clinical(ds.to_pandas())
    rep = _km_split_report(pdf, score, "Profil agresywny (high)", "Profil łagodny (low)")
    rep["n_genes"] = n_genes
    return rep


def single_gene_km_report(ds: pl.DataFrame, ensg: str, symbol: str) -> dict:
    """KM pojedynczego genu (high/low względem mediany ekspresji) + log-rank."""
    gene_cols = [c for c in ds.columns if c.startswith("ENSG")]
    col = find_gene_col(ensg, gene_cols)
    if col is None:
        return {"error": f"Gen {symbol} ({ensg}) nie znaleziony w macierzy", "symbol": symbol}
    expr = ds[col].to_numpy().astype(float)
    pdf = _encode_clinical(ds.to_pandas())
    rep = _km_split_report(pdf, expr, f"{symbol} high", f"{symbol} low")
    rep["symbol"] = symbol
    rep["median_expr"] = float(np.median(expr))
    return rep


def multi_gene_km_report(ds: pl.DataFrame, genes: list) -> list:
    """Porównanie wielu genów: log-rank high vs low + krzywa grupy high per gen.

    ``genes`` to lista par (symbol, ensg). Zwraca listę słowników
    {symbol, p_value, note, n_high, median_os_high, high_curve} — GUI rysuje
    krzywe high, terminal pokazuje tabelę log-rank.
    """
    gene_cols = [c for c in ds.columns if c.startswith("ENSG")]
    pdf = _encode_clinical(ds.to_pandas())
    results = []
    for symbol, ensg in genes:
        col = find_gene_col(ensg, gene_cols)
        if col is None:
            results.append({"symbol": symbol, "p_value": None, "note": "brak w macierzy",
                            "n_high": 0, "median_os_high": None, "high_curve": None})
            continue
        expr = ds[col].to_numpy().astype(float)
        median = float(np.median(expr))
        high = expr >= median
        low = ~high
        high_stats, high_curve = _km_fit(pdf.loc[high, "time_years"], pdf.loc[high, "event"])
        p_value = None
        try:
            res = logrank_test(pdf.loc[high, "time_years"], pdf.loc[low, "time_years"],
                               pdf.loc[high, "event"], pdf.loc[low, "event"])
            p_value = float(res.p_value)
        except Exception:
            p_value = None
        results.append({"symbol": symbol, "p_value": p_value, "note": "",
                        "n_high": int(high.sum()), "median_os_high": high_stats["median_os"],
                        "high_curve": high_curve})
    return results


def cox_clinical_report(ds: pl.DataFrame) -> dict:
    """Cox kliniczny (wiek + płeć + stadium). Zwraca rows + c_index + n.

    ``rows`` to lista słowników {name, label, hr, ci_lower, ci_upper, p} —
    surowe liczby do dowolnego renderowania (forest plot albo tabela).
    """
    pdf = _encode_clinical(ds.to_pandas())
    cox_df = pdf[pdf["stage_group"].isin(MAIN_STAGES)]
    cox_input = cox_df[["time_years", "event", "age_at_index",
                        "gender_male", "stage_numeric"]].dropna()
    if cox_input.shape[0] < 20:
        return {"error": "Za mało kompletnych obserwacji do modelu Coxa (min. 20)."}

    try:
        cph = CoxPHFitter()
        cph.fit(cox_input, duration_col="time_years", event_col="event")
    except Exception as exc:  # ConvergenceError, kolinearność itp.
        return {"error": f"Model Coxa się nie dopasował: {exc}"}

    s = cph.summary
    rows = []
    for idx in s.index:
        rows.append({
            "name": idx,
            "label": COX_CLINICAL_LABELS.get(idx, idx),
            "hr": float(s.loc[idx, "exp(coef)"]),
            "ci_lower": float(s.loc[idx, "exp(coef) lower 95%"]),
            "ci_upper": float(s.loc[idx, "exp(coef) upper 95%"]),
            "p": float(s.loc[idx, "p"]),
        })
    return {
        "rows": rows,
        "c_index": float(cph.concordance_index_),
        "n": int(cox_input.shape[0]),
        "n_events": int(cox_input["event"].sum()),
    }


def cox_genes_report(ds: pl.DataFrame) -> dict:
    """Multivariate Cox: klinika + panel genów (z-score log2 TPM).

    Fituje DWA modele na IDENTYCZNej kohorcie (klinika vs klinika+geny), by
    porównanie C-index było uczciwe (ten sam N). Zwraca rows (geny) + oba
    C-index + delta (wkład genów) + lista pominiętych genów.
    """
    gene_cols = [c for c in ds.columns if c.startswith("ENSG")]
    pdf = _encode_clinical(ds.to_pandas())

    feats, missing = [], []
    for symbol, (ensg, _sign) in SIGNATURE_PANEL.items():
        col = find_gene_col(ensg, gene_cols)
        if col is None:
            missing.append(symbol)
            continue
        expr_log = np.log2(ds[col].to_numpy().astype(float) + 1)
        std = expr_log.std()
        if std == 0:
            missing.append(symbol)
            continue
        z = (expr_log - expr_log.mean()) / std
        feat = f"g_{symbol.replace('-', '_')}"
        pdf[feat] = z  # przypisanie pozycyjne: len(z) == ds.height == len(pdf)
        feats.append((feat, symbol))

    if not feats:
        return {"error": "Żaden gen panelu nie został znaleziony w macierzy."}

    cox_df = pdf[pdf["stage_group"].isin(MAIN_STAGES)]
    base = ["time_years", "event", "age_at_index", "gender_male", "stage_numeric"]
    feat_cols = [f for f, _ in feats]
    combined = cox_df[base + feat_cols].dropna()
    if combined.shape[0] < 20:
        return {"error": "Za mało kompletnych obserwacji do modelu Coxa (min. 20)."}

    try:
        cph_clin = CoxPHFitter()
        cph_clin.fit(combined[base], duration_col="time_years", event_col="event")
        cph_genes = CoxPHFitter()
        cph_genes.fit(combined, duration_col="time_years", event_col="event")
    except Exception as exc:
        return {"error": f"Model Coxa się nie dopasował: {exc}"}

    s = cph_genes.summary
    rows = []
    for feat, symbol in feats:
        if feat not in s.index:
            continue
        rows.append({
            "symbol": symbol,
            "hr": float(s.loc[feat, "exp(coef)"]),
            "ci_lower": float(s.loc[feat, "exp(coef) lower 95%"]),
            "ci_upper": float(s.loc[feat, "exp(coef) upper 95%"]),
            "p": float(s.loc[feat, "p"]),
        })

    c_clin = float(cph_clin.concordance_index_)
    c_genes = float(cph_genes.concordance_index_)
    return {
        "rows": rows,
        "c_index_clinical": c_clin,
        "c_index_genes": c_genes,
        "delta": c_genes - c_clin,
        "n": int(combined.shape[0]),
        "missing": missing,
    }


# ---------------------------------------------------------------------------
#  Rygor statystyczny: poziom istotności, korekcja FDR, sensitivity analysis
# ---------------------------------------------------------------------------
ALPHA = 0.05  # jawny próg istotności dla całej analizy (dwustronnie)


def benjamini_hochberg(pvalues: list) -> list:
    """Korekcja Benjamini-Hochberg (FDR). Zwraca q-wartości w oryginalnej kolejności.

    Mniej konserwatywna niż Bonferroni — kontroluje odsetek fałszywych odkryć
    (FDR), nie rodzinny błąd I rodzaju (FWER). Wartości None są pomijane w
    korekcji i zwracane jako None (np. gen nieobecny w macierzy).
    """
    valid_idx = [i for i, p in enumerate(pvalues)
                 if p is not None and not (isinstance(p, float) and np.isnan(p))]
    out: list = [None] * len(pvalues)
    if not valid_idx:
        return out
    p = np.array([pvalues[i] for i in valid_idx], dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]   # monotoniczność od największej w dół
    q = np.clip(q, 0.0, 1.0)
    q_in_order = np.empty(m)
    q_in_order[order] = q
    for j, i in enumerate(valid_idx):
        out[i] = float(q_in_order[j])
    return out


def multiple_testing_report(named_pvalues: list, alpha: float = ALPHA) -> dict:
    """Korekcja wielokrotnego testowania (BH-FDR) dla nazwanych testów.

    named_pvalues: lista (nazwa, p_value). Zwraca tabelę rows posortowaną po p
    rosnąco z kolumnami name/p/q/reject_raw/reject_fdr oraz licznikami: ile
    istotnych surowo vs po korekcji. Pokazuje, które wyniki przeżywają FDR.
    """
    pvals = [p for _, p in named_pvalues]
    qvals = benjamini_hochberg(pvals)
    rows = []
    for (name, p), q in zip(named_pvalues, qvals):
        rows.append({
            "name": name,
            "p": p,
            "q": q,
            "reject_raw": (p is not None and p < alpha),
            "reject_fdr": (q is not None and q < alpha),
        })
    rows.sort(key=lambda r: (r["p"] is None, r["p"] if r["p"] is not None else 1.0))
    return {
        "alpha": alpha,
        "method": "Benjamini-Hochberg (FDR)",
        "rows": rows,
        "n_tested": sum(1 for p in pvals if p is not None),
        "n_sig_raw": sum(1 for r in rows if r["reject_raw"]),
        "n_sig_fdr": sum(1 for r in rows if r["reject_fdr"]),
    }


def gene_panel_fdr_report(ds: pl.DataFrame, source: str = "single_gene",
                          alpha: float = ALPHA) -> dict:
    """Korekcja FDR (BH) dla panelu 7 genów sygnatury.

    source='single_gene' — p z log-rank KM (high/low per gen);
    source='cox' — p z multivariate Cox (klinika + geny). Zwraca
    multiple_testing_report na p-wartościach panelu, plus 'source'.
    """
    named = []
    if source == "cox":
        rep = cox_genes_report(ds)
        if "error" in rep:
            return {"error": rep["error"], "source": source}
        named = [(r["symbol"], r.get("p")) for r in rep["rows"]]
    else:
        for symbol, (ensg, _sign) in SIGNATURE_PANEL.items():
            rep = single_gene_km_report(ds, ensg, symbol)
            named.append((symbol, rep.get("logrank_p") if "error" not in rep else None))
    if not any(p is not None for _, p in named):
        return {"error": "Brak wartości p do korekcji (panel nieobecny w macierzy).",
                "source": source}
    out = multiple_testing_report(named, alpha=alpha)
    out["source"] = source
    return out


def schoenfeld_min_hr(n_events: int, alpha: float = ALPHA, power: float = 0.80,
                      allocation: float = 0.5) -> float:
    """Minimalny wykrywalny HR przy danej liczbie ZDARZEŃ (wzór Schoenfelda 1983).

    Test log-rank/Coxa: ln(HR) = (z_{1-α/2} + z_power) / sqrt(d·p1·p2), gdzie
    d = liczba zdarzeń, p1/p2 = proporcje grup (allocation=0.5 → median split).
    Zwraca HR > 1 (kierunek ryzyka; odwrotność = kierunek ochronny).
    """
    from scipy.stats import norm
    import math
    if n_events <= 0:
        return float("inf")
    p1, p2 = allocation, 1.0 - allocation
    za = norm.ppf(1.0 - alpha / 2.0)
    zb = norm.ppf(power)
    ln_hr = (za + zb) / math.sqrt(n_events * p1 * p2)
    return float(math.exp(ln_hr))


def schoenfeld_power(n_events: int, hr: float, alpha: float = ALPHA,
                     allocation: float = 0.5) -> float:
    """Moc testu log-rank/Coxa dla danego HR i liczby ZDARZEŃ (wzór Schoenfelda)."""
    from scipy.stats import norm
    import math
    if n_events <= 0 or hr <= 0:
        return 0.0
    p1, p2 = allocation, 1.0 - allocation
    za = norm.ppf(1.0 - alpha / 2.0)
    zb = math.sqrt(n_events * p1 * p2) * abs(math.log(hr)) - za
    return float(norm.cdf(zb))


def sensitivity_report(n_events: int, alpha: float = ALPHA, allocation: float = 0.5,
                       powers: tuple = (0.80, 0.90),
                       hr_grid: tuple = (1.3, 1.5, 1.75, 2.0)) -> dict:
    """Sensitivity analysis (Schoenfeld) — poprawna alternatywa dla post-hoc power.

    Dla posiadanej liczby ZDARZEŃ podaje minimalny wykrywalny HR przy mocy 80%/90%
    oraz moc dla siatki HR. Mówi, jak mały efekt jest wykrywalny — w przeciwieństwie
    do post-hoc power liczonej z OBSERWOWANEGO efektu, która jest nieważna
    (Hoenig & Heisey 2001) i tylko przelicza p-wartość. Dla przeżycia liczą się
    zdarzenia, nie liczba pacjentów (n jest tu dane — cała kohorta TCGA).
    """
    return {
        "n_events": int(n_events),
        "alpha": alpha,
        "allocation": allocation,
        "min_detectable_hr": {pw: schoenfeld_min_hr(n_events, alpha, pw, allocation)
                              for pw in powers},
        "power_at_hr": {hr: schoenfeld_power(n_events, hr, alpha, allocation)
                        for hr in hr_grid},
        "note": ("Median split (allocation 0.5). Post-hoc power z obserwowanego "
                 "efektu jest nieważne — to sensitivity analysis ex ante."),
    }


def statistical_rigor_report(ds: pl.DataFrame, alpha: float = ALPHA) -> dict:
    """Zbiorczy raport rygoru: α, FDR panelu (single-gene + Cox), sensitivity.

    Jedno źródło dla GUI i terminala. n_events z cohort_summary.
    """
    n_events = cohort_summary(ds).get("n_events", 0)
    return {
        "alpha": alpha,
        "fdr_single_gene": gene_panel_fdr_report(ds, source="single_gene", alpha=alpha),
        "fdr_cox": gene_panel_fdr_report(ds, source="cox", alpha=alpha),
        "sensitivity": sensitivity_report(n_events, alpha=alpha),
    }
