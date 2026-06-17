"""Headless podsumowanie macierzy ekspresji LUAD-HUBA — czyste dane.

Konsumowane przez terminal (panel EXPRESSION) tak jak survival_report przez
SURVIVAL. Czyta ``data/processed/expression_matrix.parquet`` (geny × próbki:
kolumna ``gene_id`` + po jednej kolumnie na ``sample_id``) i zwraca słownik liczb
do renderowania jako tabele.

Macierz może trzymać różne metryki (domyślnie ``unstranded`` = surowe zliczenia,
ale config może ustawić ``tpm_unstranded``), więc metryka jest **wykrywana z
danych** (suma per próbka ≈ 1e6 → TPM; wartości całkowite → zliczenia) i
raportowana jawnie — bez zaszytego założenia.

Najważniejsza sekcja dla obrony: rozkład próbek per ośrodek TSS (kod z barkodu
TCGA), bo to ujawnia strukturę batchu. PCA liczona na top-zmiennych genach
(subsampling) — zwracamy % wariancji PC1–PC5, nie współrzędne.
"""

__author__ = "Łukasz Połaski"

import numpy as np
import polars as pl

# Markery LUAD (a priori) — te same co w dashboardzie; tu jako warstwa analityczna.
LUAD_MARKERS = {
    "EGFR": "ENSG00000146648",
    "KRAS": "ENSG00000133703",
    "TP53": "ENSG00000141510",
    "ALK": "ENSG00000171094",
    "ROS1": "ENSG00000047936",
    "NKX2-1": "ENSG00000136352",
    "SFTPC": "ENSG00000168484",
}

# Pełny GENCODE v36 (TCGA STAR-Counts) — punkt odniesienia dla filtra biotypów.
GENCODE_TOTAL = 60660


def _detect_metric(col_sums: np.ndarray, sample_vals: np.ndarray) -> str:
    """Zgaduje metrykę z danych: TPM (sumy ≈ 1e6), zliczenia (całkowite), inne."""
    median_sum = float(np.median(col_sums))
    if 9.0e5 <= median_sum <= 1.1e6:
        return "TPM"
    # Próbka wartości całkowitych → surowe zliczenia.
    sample = sample_vals[np.isfinite(sample_vals)]
    if sample.size and np.allclose(sample, np.round(sample)):
        return "zliczenia (counts)"
    return "FPKM/inne (nieokreślona)"


def _parse_tss(sample_id: str) -> str:
    """Kod ośrodka (Tissue Source Site) z barkodu TCGA-XX-...; '?' gdy nie barkod."""
    parts = str(sample_id).split("-")
    if len(parts) >= 2 and parts[0].upper() == "TCGA":
        return parts[1]
    return "?"


def _marker_values(matrix: pl.DataFrame, gene_col: str, sample_cols: list, ensg: str):
    """Wektor ekspresji markera (po prefiksie ENSG, bo gene_id bywa wersjonowany)."""
    hit = matrix.filter(pl.col(gene_col).str.starts_with(ensg))
    if hit.height == 0:
        return None
    return hit.select(sample_cols).row(0)


def expression_summary(matrix_path, *, top_n_var: int = 2000) -> dict:
    """Pełne podsumowanie macierzy ekspresji (czyste dane) albo {'error': ...}."""
    from pathlib import Path

    path = Path(matrix_path)
    if not path.exists():
        return {"error": "BRAK MACIERZY EKSPRESJI\n\n"
                         f"Nie znaleziono:\n  {path}\n\n"
                         "Zbuduj etap 'Macierz ekspresji' (CLI lub dashboard)."}
    try:
        matrix = pl.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"BŁĄD ODCZYTU MACIERZY:\n  {exc}"}

    gene_col = "gene_id" if "gene_id" in matrix.columns else matrix.columns[0]
    sample_cols = [c for c in matrix.columns if c != gene_col]
    n_genes, n_samples = matrix.height, len(sample_cols)
    if n_samples == 0 or n_genes == 0:
        return {"error": "Macierz pusta lub bez kolumn próbek."}

    vals = matrix.select(sample_cols).to_numpy().astype(float)  # geny × próbki
    col_sums = np.nansum(vals, axis=0)

    # Metryka + rozkład wartości.
    metric = _detect_metric(col_sums, vals[: min(50, n_genes)].ravel())
    flat = vals[np.isfinite(vals)]
    distribution = {
        "metric": metric,
        "median": float(np.median(flat)),
        "p25": float(np.percentile(flat, 25)),
        "p75": float(np.percentile(flat, 75)),
        "max": float(np.max(flat)),
        "zero_pct": float((vals == 0).mean() * 100.0),
        "median_depth": float(np.median(col_sums)),
    }

    # Batch TSS.
    tss_counts: dict[str, int] = {}
    for s in sample_cols:
        code = _parse_tss(s)
        tss_counts[code] = tss_counts.get(code, 0) + 1
    tss_rows = [
        {"tss": code, "n": n, "pct": 100.0 * n / n_samples}
        for code, n in sorted(tss_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    parsed_tss = sum(1 for c in tss_counts if c != "?")

    # PCA na top-zmiennych genach (log2(x+1), z-score per gen) → % wariancji.
    log_vals = np.log2(vals + 1.0)
    gene_var = log_vals.var(axis=1)
    k = int(min(top_n_var, n_genes))
    top_idx = np.argsort(gene_var)[::-1][:k]
    sub = log_vals[top_idx, :]  # k × próbki
    mu = sub.mean(axis=1, keepdims=True)
    sd = sub.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    z = ((sub - mu) / sd).T  # próbki × geny (wystandaryzowane)
    pcs = []
    try:
        s_vals = np.linalg.svd(z - z.mean(axis=0), compute_uv=False)
        var_exp = (s_vals ** 2) / float((s_vals ** 2).sum())
        pcs = [{"pc": i + 1, "var_pct": float(var_exp[i] * 100.0)}
               for i in range(min(5, var_exp.size))]
    except Exception:  # noqa: BLE001 — gdyby SVD nie zbiegł
        pcs = []

    # Markery LUAD: mediana ekspresji (w metryce macierzy) + log2.
    markers = []
    for symbol, ensg in LUAD_MARKERS.items():
        vec = _marker_values(matrix, gene_col, sample_cols, ensg)
        if vec is None:
            markers.append({"symbol": symbol, "found": False})
            continue
        arr = np.asarray(vec, dtype=float)
        markers.append({
            "symbol": symbol,
            "found": True,
            "median": float(np.median(arr)),
            "median_log2": float(np.median(np.log2(arr + 1.0))),
        })

    return {
        "n_genes": n_genes,
        "n_samples": n_samples,
        "gencode_total": GENCODE_TOTAL,
        "biotype_filtered": n_genes < GENCODE_TOTAL,
        "distribution": distribution,
        "tss_rows": tss_rows,
        "n_tss": len(tss_counts),
        "parsed_tss": parsed_tss,
        "pca_top_n": k,
        "pcs": pcs,
        "markers": markers,
    }
