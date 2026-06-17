# -*- coding: utf-8 -*-
"""Ocena gotowości danych pod uczenie maszynowe (ML-readiness).

Liczy zestaw metryk diagnostycznych mówiących, czy macierz ekspresji + etykiety
nadają się jako materiał do modeli ML (docelowo multimodalnych): reżim p≫n,
uczenie z etykiety przeżycia, balans klas, jakość cech, efekt batch i integralność.
Każda metryka dostaje status green/yellow/red do kontrolek „rzut oka".

Jedno źródło — używane przez GUI (zakładka Gotowość ML). Bez zależności od Streamlita.
"""
from __future__ import annotations

import numpy as np

from .survival_report import MAIN_STAGES, _encode_clinical


def _tss_from_barcode(bc) -> str:
    """TSS (Tissue Source Site) = 2. pole barkodu TCGA, np. TCGA-44-2666-01B → '44'."""
    parts = str(bc).split("-")
    return parts[1] if len(parts) >= 2 else "?"


def _eta_squared(values, groups) -> float:
    """Correlation ratio η²: udział wariancji ``values`` tłumaczony przez ``groups``.

    Uwaga: rośnie z liczbą grup — traktować jako sygnał przesiewowy, nie test formalny.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    mask = np.isfinite(values)
    values, groups = values[mask], groups[mask]
    if values.size < 3:
        return float("nan")
    grand = values.mean()
    ss_total = float(((values - grand) ** 2).sum())
    if ss_total == 0:
        return float("nan")
    ss_between = 0.0
    for g in np.unique(groups):
        vg = values[groups == g]
        if vg.size:
            ss_between += vg.size * (vg.mean() - grand) ** 2
    return float(ss_between / ss_total)


def _pc_scores(gene_matrix, top_n: int = 2000):
    """PC1 scores z macierzy próbki×geny: log2(x+1), z-score, top-zmienne geny, SVD.

    Zwraca (pc1_scores, var_ratio) albo (None, None) gdy się nie uda.
    """
    vals = np.nan_to_num(np.asarray(gene_matrix, dtype=float), nan=0.0)  # próbki × geny
    if vals.ndim != 2 or vals.shape[1] == 0 or vals.shape[0] < 3:
        return None, None
    log_vals = np.log2(vals + 1.0)
    gene_var = log_vals.var(axis=0)
    k = int(min(top_n, log_vals.shape[1]))
    top_idx = np.argsort(gene_var)[::-1][:k]
    sub = log_vals[:, top_idx]
    mu = sub.mean(axis=0, keepdims=True)
    sd = sub.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    z = (sub - mu) / sd
    z = z - z.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    scores = u * s  # próbki × komponenty
    var_ratio = (s ** 2) / float((s ** 2).sum())
    return scores[:, 0], var_ratio


def _row(label, value, status, note, action=""):
    return {"label": label, "value": value, "status": status, "note": note, "action": action}


def _pl(n: int) -> str:
    """Liczba z odstępem tysięcy (styl PL)."""
    return f"{n:,}".replace(",", " ")


def ml_readiness_report(ds, esum=None) -> dict:
    """Zwraca metryki gotowości ML pogrupowane + zbiorczy werdykt.

    ``ds``   — zbiór przeżywalności (polars): sample_id, case_id, event, kowarianty
               kliniczne + kolumny genów (ENSG...).
    ``esum`` — opcjonalny wynik expression_summary (dla wykrytej metryki normalizacji).
    """
    gene_cols = [c for c in ds.columns if c.startswith("ENSG")]
    n_samples = int(ds.height)
    n_features = len(gene_cols)
    pdf = _encode_clinical(ds.to_pandas())
    dist = (esum or {}).get("distribution", {}) if isinstance(esum, dict) else {}

    groups_out = []

    # --- Wymiary i reżim uczenia ---
    dim = [
        _row("Liczba próbek (n)", _pl(n_samples),
             "green" if n_samples >= 100 else "yellow" if n_samples >= 30 else "red",
             "Ile obserwacji ma model do nauki.",
             "" if n_samples >= 100 else "Mało próbek — ostrożnie z pojemnością modelu i walidacją."),
        _row("Liczba cech (genów, p)", _pl(n_features), "info",
             "Wymiar przestrzeni cech przed selekcją.", ""),
    ]
    ratio = (n_features / n_samples) if n_samples else float("inf")
    dim.append(_row("Cechy na próbkę (p/n)", f"{ratio:.0f}",
                    "green" if ratio <= 2 else "yellow" if ratio <= 50 else "red",
                    "Reżim p≫n: cech wielokrotnie więcej niż próbek.",
                    "Konieczna redukcja wymiaru (PCA/selekcja) lub regularyzacja (ridge/lasso)."
                    if ratio > 2 else ""))
    n_events = int(pdf["event"].sum())
    epv = (n_events / n_features) if n_features else float("nan")
    dim.append(_row("Zdarzeń na cechę (EPV)", f"{epv:.3f}" if np.isfinite(epv) else "—",
                    "green" if epv >= 10 else "yellow" if epv >= 1 else "red",
                    "Reguła kciuka: ≥10 zdarzeń na zmienną dla stabilnego modelu.",
                    "EPV≪10 — nie fituj modelu na surowych genach; najpierw redukcja/penalizacja."
                    if np.isfinite(epv) and epv < 10 else ""))
    groups_out.append({"title": "Wymiary i reżim uczenia", "metrics": dim})

    # --- Etykieta (target) ---
    lab = []
    cens = float(pdf["event"].eq(0).mean() * 100.0)
    lab.append(_row("Zdarzenia (zgony)", _pl(n_events),
                    "green" if n_events >= 100 else "yellow" if n_events >= 50 else "red",
                    "To zdarzenia, nie próbki, niosą sygnał uczący w analizie przeżycia.",
                    "Mało zdarzeń — ogranicza złożoność modelu." if n_events < 100 else ""))
    lab.append(_row("Cenzurowanie", f"{cens:.1f}%",
                    "green" if cens < 70 else "yellow" if cens < 90 else "red",
                    "Odsetek obserwacji uciętych (bez zdarzenia).",
                    "Wysoka cenzura — rozważ modele z wagami (IPCW)." if cens >= 70 else ""))
    clin_fields = [c for c in ("age_at_index", "gender", "ajcc_pathologic_stage") if c in pdf.columns]
    if clin_fields:
        complete = float(pdf[clin_fields].notna().all(axis=1).mean() * 100.0)
        lab.append(_row("Kompletność etykiet klinicznych", f"{complete:.1f}%",
                        "green" if complete >= 95 else "yellow" if complete >= 80 else "red",
                        "Odsetek próbek z pełnym kompletem: wiek, płeć, stadium.",
                        "Braki w etykietach — imputacja albo wykluczenie niekompletnych."
                        if complete < 95 else ""))
    groups_out.append({"title": "Etykieta (target)", "metrics": lab})

    # --- Balans klas ---
    bal = []
    sc = pdf[pdf["stage_group"].isin(MAIN_STAGES)]["stage_group"].value_counts().to_dict()
    if sc:
        mx, mn = max(sc.values()), min(sc.values())
        imb = (mx / mn) if mn else float("inf")
        dist_str = " · ".join(f"{k}:{sc[k]}" for k in sorted(sc))
        bal.append(_row("Nierównowaga stadiów", f"{imb:.1f}:1  ({dist_str})",
                        "green" if imb <= 3 else "yellow" if imb <= 10 else "red",
                        "Stosunek najliczniejszej do najrzadszej klasy stadium.",
                        "Silna nierównowaga — ważenie klas / resampling / metryki F1, AUC-PR."
                        if imb > 3 else ""))
    groups_out.append({"title": "Balans klas", "metrics": bal})

    # --- Jakość cech ---
    gm = ds.select(gene_cols).to_numpy().astype(float) if n_features else np.empty((n_samples, 0))
    qual = []
    metric = dist.get("metric")
    if metric:
        qual.append(_row("Metryka normalizacji", str(metric).upper(),
                         "green" if metric == "tpm" else "yellow" if metric in ("fpkm", "fpkm_uq") else "red",
                         "TPM porównywalne między próbkami; FPKM zbliżone; counts surowe.",
                         "FPKM → rozważ przeliczenie na TPM dla porównań między próbkami."
                         if metric in ("fpkm", "fpkm_uq")
                         else "Surowe counts — znormalizuj (TMM/DESeq2) lub użyj narzędzi robiących to same."
                         if metric not in ("tpm", "fpkm", "fpkm_uq") else ""))
    zero_pct = dist.get("zero_pct")
    if zero_pct is None and n_features:
        zero_pct = float((gm == 0).mean() * 100.0)
    if zero_pct is not None:
        qual.append(_row("Rzadkość (wartości zerowe)", f"{zero_pct:.1f}%",
                         "green" if zero_pct < 20 else "yellow" if zero_pct < 50 else "red",
                         "Odsetek zer w macierzy — typowa cecha RNA-seq.",
                         "Wysoka rzadkość — modele odporne na zera / regularyzacja." if zero_pct >= 20 else ""))
    if n_features:
        gene_sd = gm.std(axis=0)
        n_zero_var = int((gene_sd == 0).sum())
        qual.append(_row("Geny o zerowej wariancji", _pl(n_zero_var),
                         "green" if n_zero_var == 0 else "yellow" if n_zero_var <= 0.1 * n_features else "red",
                         "Cechy stałe nic nie wnoszą i psują niektóre modele.",
                         "Odrzuć te geny przed treningiem (filtr wariancji)." if n_zero_var > 0 else ""))
    groups_out.append({"title": "Jakość cech", "metrics": qual})

    # --- Batch / wyciek techniczny ---
    batch = []
    tss = None
    if "sample_id" in pdf.columns:
        tss = np.array([_tss_from_barcode(b) for b in pdf["sample_id"]])
        known = tss[tss != "?"]
        if known.size:
            _, cnts = np.unique(known, return_counts=True)
            imb_b = int(cnts.max()) / int(cnts.min())
            batch.append(_row("Ośrodki TSS (batch)", f"{cnts.size} ośrodków · nierównowaga {imb_b:.0f}:1",
                              "green" if imb_b <= 5 else "yellow" if imb_b <= 20 else "red",
                              "Różne ośrodki = różne protokoły; źródło efektu batch.",
                              "Nierównowaga — sprawdź, czy ośrodek nie koreluje z targetem." if imb_b > 5 else ""))
    if n_features:
        pc1, var_ratio = _pc_scores(gm)
        if pc1 is not None and tss is not None:
            eta_stage = _eta_squared(pc1, pdf["stage_group"].to_numpy())
            eta_tss = _eta_squared(pc1, tss)
            pc1_var = float(var_ratio[0] * 100.0)
            if np.isfinite(eta_tss) and np.isfinite(eta_stage):
                if eta_tss >= 0.30 and eta_tss > eta_stage:
                    st_, act = "red", "PC1 idzie za ośrodkiem — skoryguj batch (ComBat/limma) przed ML."
                elif eta_stage >= eta_tss:
                    st_, act = "green", ""
                else:
                    st_, act = "yellow", "Batch współkształtuje PC1 — rozważ korektę i kontrolę TSS."
                batch.append(_row(
                    "PC1 — biologia czy batch?",
                    f"η²(stadium)={eta_stage:.2f} vs η²(ośrodek)={eta_tss:.2f} · PC1={pc1_var:.0f}% wariancji",
                    st_,
                    "Czy główna oś zmienności to biologia (stadium) czy artefakt (ośrodek). "
                    "Sygnał przesiewowy — η² rośnie z liczbą grup.",
                    act))
    groups_out.append({"title": "Batch / wyciek techniczny", "metrics": batch})

    # --- Integralność ---
    integ = []
    if "case_id" in pdf.columns:
        vc = pdf["case_id"].value_counts()
        dup = int((vc > 1).sum())
        integ.append(_row("Duplikaty próbek na pacjenta", _pl(dup),
                          "green" if dup == 0 else "yellow" if dup <= 5 else "red",
                          "Kilka próbek jednego pacjenta = wyciek między train/test.",
                          "Deduplikuj albo dziel zbiór po pacjencie (GroupKFold)." if dup > 0 else ""))
    groups_out.append({"title": "Integralność", "metrics": integ})

    # --- Podsumowanie / werdykt ---
    all_m = [m for g in groups_out for m in g["metrics"]]
    counts = {"green": 0, "yellow": 0, "red": 0, "info": 0}
    for m in all_m:
        counts[m["status"]] = counts.get(m["status"], 0) + 1
    actions = []
    for m in all_m:
        if m["status"] in ("yellow", "red") and m["action"] and m["action"] not in actions:
            actions.append(m["action"])
    dealbreaker = any(
        m["status"] == "red" and m["label"] in ("Liczba próbek (n)", "Zdarzenia (zgony)")
        for m in all_m
    )
    if dealbreaker:
        v_status = "red"
        verdict = "Dane w obecnej formie nie wystarczą — za mało próbek lub zdarzeń."
    elif counts["red"] or counts["yellow"]:
        v_status = "yellow"
        verdict = "Dane użyteczne pod ML, ale wymagają przygotowania (patrz zalecane kroki)."
    else:
        v_status = "green"
        verdict = "Dane gotowe pod ML bez większych zastrzeżeń."

    return {
        "groups": groups_out,
        "summary": {**counts, "verdict_status": v_status, "verdict": verdict, "actions": actions},
    }
