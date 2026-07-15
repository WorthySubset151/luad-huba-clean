# -*- coding: utf-8 -*-
"""Eksport zbioru gotowego pod uczenie maszynowe.

Zamienia audyt gotowości w audyt + remediację: bierze zbiór przeżywalności i
produkuje materiał treningowy — cechy (X) + etykiety (y) + podział train/test
grupowany po pacjencie + manifest dokumentujący każdy krok.

Dyscyplina braku wycieku (leakage): deduplikacja i podział po PACJENCIE, a
selekcja cech oraz standaryzacja są fitowane WYŁĄCZNIE na train i aplikowane na
test. Rzeczy, których nie wolno wpiekać w dane (wagi klas, regularyzacja, korekta
batchu, schemat walidacji) trafiają do manifestu jako rekomendacje dla modelu.

Jedno źródło — używane przez GUI (zakładka Gotowość ML). Bez zależności od Streamlita.
"""
from __future__ import annotations

import datetime as _dt
import io
import json
import zipfile

import numpy as np
import pandas as pd

_META_CANDIDATES = (
    "sample_id", "case_id", "time", "event",
    "age_at_index", "gender", "ajcc_pathologic_stage", "tissue_type",
)
_Y_CANDIDATES = (
    "sample_id", "case_id", "time", "event",
    "age_at_index", "gender", "ajcc_pathologic_stage",
)

EXPORT_FORMATS: tuple[str, ...] = ("parquet", "csv")
DEFAULT_EXPORT_FORMAT = "parquet"


def _readme_text(ext: str) -> str:
    """Treść README dołączanego do archiwum, dopasowana do formatu tabel."""
    note = (
        "Format tabel: Parquet — typy kolumn i wartości brakujące zachowane wiernie,\n"
        "odczyt kilkukrotnie szybszy niż z CSV, a pliki przy typowych rozmiarach mniejsze\n"
        "(pandas.read_parquet / polars.read_parquet).\n"
        if ext == "parquet"
        else "Format tabel: CSV — uniwersalny, ale typy kolumn odtwarzane przy wczytaniu.\n"
    )
    return (
        "Zbiór gotowy pod uczenie maszynowe (LUAD-HUBA)\n"
        "===============================================\n\n"
        "Pliki:\n"
        f"  X_train.{ext} / X_test.{ext}   – cechy (sample_id + wybrane geny) po transformacjach.\n"
        f"  y_train.{ext} / y_test.{ext}   – etykiety (sample_id, case_id, time, event, klinika).\n"
        "  selected_genes.txt         – lista wybranych genów (kolejność kolumn cech).\n"
        "  manifest.json              – pełna dokumentacja przygotowania + rekomendacje modelu.\n\n"
        + note +
        "\nKluczowe: podział train/test jest grupowany po pacjencie (case_id), a selekcja\n"
        "cech i standaryzacja fitowane WYŁĄCZNIE na train — test jest nietknięty do oceny.\n"
        "Do walidacji krzyżowej używaj GroupKFold po case_id. Szczegóły w manifest.json.\n"
    )


def _grouped_stratified_split(case_ids, events, test_frac: float, seed: int):
    """Maski train/test: całe PACJENTY do jednego zbioru, stratyfikacja po event."""
    rng = np.random.default_rng(int(seed))
    df = pd.DataFrame({"case": np.asarray(case_ids), "event": np.asarray(events).astype(int)})
    pat_event = df.groupby("case")["event"].max()
    test_patients: set = set()
    for ev in (0, 1):
        pts = pat_event.index[pat_event.values == ev].to_numpy()
        if pts.size == 0:
            continue
        rng.shuffle(pts)
        n_test = int(round(pts.size * float(test_frac)))
        n_test = min(max(n_test, 0), pts.size)
        test_patients.update(pts[:n_test].tolist())
    is_test = df["case"].isin(test_patients).to_numpy()
    return ~is_test, is_test


def prepare_ml_dataset(ds, top_k: int = 2000, test_frac: float = 0.2, seed: int = 42,
                       dedup: bool = True, standardize: bool = True) -> dict:
    """Buduje zbiór ML z ``ds`` (polars). Zwraca X_train/X_test, y_train/y_test,
    listę genów i manifest. Wszystkie fity są liczone tylko na train."""
    gene_cols = [c for c in ds.columns if c.startswith("ENSG")]
    if not gene_cols:
        raise ValueError("Zbiór nie zawiera kolumn genów (ENSG...).")

    pdf = ds.to_pandas()
    meta_cols = [c for c in _META_CANDIDATES if c in pdf.columns]
    n_before = len(pdf)

    # 1. deduplikacja: 1 próbka na pacjenta (deepest = maks. suma ekspresji)
    if dedup and "case_id" in pdf.columns:
        totals = np.nan_to_num(pdf[gene_cols].to_numpy().astype(float)).sum(axis=1)
        pdf = (
            pdf.assign(_tot=totals)
            .sort_values("_tot", ascending=False)
            .drop_duplicates("case_id", keep="first")
            .drop(columns="_tot")
            .reset_index(drop=True)
        )
    n_dedup = len(pdf)

    # 2. podział grupowany po pacjencie + stratyfikacja po event
    case = (pdf["case_id"].astype(str).to_numpy() if "case_id" in pdf.columns
            else np.array([f"s{i}" for i in range(len(pdf))]))
    event = (pdf["event"].fillna(0).astype(int).to_numpy() if "event" in pdf.columns
             else np.zeros(len(pdf), dtype=int))
    train_mask, test_mask = _grouped_stratified_split(case, event, test_frac, seed)
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Podział train/test dał pusty zbiór — zmień frakcję testu lub seed.")

    # 3. macierz cech + log2(x+1)
    X = np.log2(np.clip(np.nan_to_num(pdf[gene_cols].to_numpy().astype(float)), 0.0, None) + 1.0)
    Xtr, Xte = X[train_mask], X[test_mask]

    # 4. selekcja cech — FIT TYLKO NA TRAIN: odrzuć zerową wariancję + top-K wariancji
    var_tr = Xtr.var(axis=0)
    nonzero = np.where(var_tr > 0)[0]
    ranked = nonzero[np.argsort(var_tr[nonzero])[::-1]]
    k = int(min(max(top_k, 1), len(ranked)))
    sel_idx = np.sort(ranked[:k])
    sel_genes = [gene_cols[i] for i in sel_idx]
    Xtr_s, Xte_s = Xtr[:, sel_idx], Xte[:, sel_idx]

    # 5. standaryzacja — FIT TYLKO NA TRAIN
    if standardize:
        mu = Xtr_s.mean(axis=0)
        sd = Xtr_s.std(axis=0)
        sd[sd == 0] = 1.0
        Xtr_s = (Xtr_s - mu) / sd
        Xte_s = (Xte_s - mu) / sd

    # 6. złożenie X / y
    meta = pdf[meta_cols].reset_index(drop=True)
    y_cols = [c for c in _Y_CANDIDATES if c in meta.columns]

    def _mk_x(mask, xs):
        d = pd.DataFrame(xs, columns=sel_genes)
        if "sample_id" in meta.columns:
            d.insert(0, "sample_id", meta[mask]["sample_id"].to_numpy())
        return d

    X_train, X_test = _mk_x(train_mask, Xtr_s), _mk_x(test_mask, Xte_s)
    y_train = meta[train_mask][y_cols].reset_index(drop=True)
    y_test = meta[test_mask][y_cols].reset_index(drop=True)

    shared = set(case[train_mask]) & set(case[test_mask])

    def _erate(mask):
        return round(float(event[mask].mean()), 3) if mask.sum() else None

    manifest = {
        "utworzono": _dt.datetime.now().isoformat(timespec="seconds"),
        "opis": ("Zbiór gotowy pod ML: cechy (X) + etykiety (y), podział train/test "
                 "grupowany po pacjencie. Transformacje fitowane wyłącznie na train."),
        "zrodlo": {"n_probek": int(n_before), "n_genow": int(len(gene_cols))},
        "deduplikacja": {
            "wlaczona": bool(dedup),
            "strategia": "1 próbka na pacjenta — deepest (maks. suma ekspresji)",
            "n_przed": int(n_before), "n_po": int(n_dedup),
        },
        "selekcja_cech": {
            "metoda": "top-K wg wariancji, fit tylko na train (nienadzorowana, bez wycieku)",
            "geny_przed": int(len(gene_cols)),
            "zerowa_wariancja_odrzucone": int(len(gene_cols) - len(nonzero)),
            "wybrane_top_k": int(len(sel_genes)),
        },
        "transformacje": {
            "log2(x+1)": True,
            "standaryzacja": "z-score per gen, fit tylko na train" if standardize else "brak",
        },
        "podzial": {
            "metoda": "grupowany po pacjencie (case_id) + stratyfikacja po event",
            "test_frac": float(test_frac), "seed": int(seed),
            "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            "event_rate_train": _erate(train_mask), "event_rate_test": _erate(test_mask),
            "pacjenci_wspolni_train_test": int(len(shared)),
            "brak_wycieku_pacjenta": bool(len(shared) == 0),
        },
        "zalecane_ustawienia_modelu": [
            "p≫n / EPV≪10: użyj regularyzacji (ridge/lasso/elastic net) lub redukcji wymiaru; "
            "nie fituj nieregularyzowanego modelu na surowych genach.",
            "Nierównowaga stadiów: przy klasyfikacji stadium użyj wag klas / resamplingu; "
            "oceniaj F1 oraz AUC-PR, nie samą accuracy.",
            "Batch (ośrodki TSS): rozważ korektę ComBat/limma albo dodaj TSS jako kowariantę; "
            "zweryfikuj, czy PC1 nie tropi ośrodka.",
            "Walidacja krzyżowa: GroupKFold po case_id (pacjencie), nie zwykły KFold — inaczej wyciek.",
            "Cel przeżyciowy: modele świadome cenzury (Cox / Random Survival Forest / DeepSurv), "
            "nie zwykła regresja.",
        ],
    }

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "selected_genes": sel_genes, "manifest": manifest,
    }


def _table_bytes(df, fmt: str) -> bytes:
    """Serializuje ramkę do bajtów w wybranym formacie."""
    if fmt == "parquet":
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return buf.getvalue()
    return df.to_csv(index=False).encode("utf-8")


def build_ml_bundle(result: dict, fmt: str = DEFAULT_EXPORT_FORMAT) -> bytes:
    """Pakuje wynik prepare_ml_dataset w archiwum ZIP (X/y + manifest + README).

    ``fmt`` wybiera format tabel: ``parquet`` (domyślny — zachowuje typy kolumn,
    mniejsze pliki) albo ``csv`` (uniwersalny, do otwarcia w arkuszu).

    Uzupełnia ``result["manifest"]`` o sekcję ``eksport`` (format + spis plików),
    żeby manifest w archiwum opisywał to, co faktycznie w nim jest.
    """
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Nieznany format eksportu: {fmt!r} (dozwolone: {EXPORT_FORMATS}).")

    result["manifest"]["eksport"] = {
        "format_tabel": fmt,
        "pliki": {
            f"X_train.{fmt}": "cechy treningowe (sample_id + wybrane geny, po transformacjach)",
            f"X_test.{fmt}": "cechy testowe (te same geny, transformacje z train)",
            f"y_train.{fmt}": "etykiety treningowe (sample_id, case_id, time, event, klinika)",
            f"y_test.{fmt}": "etykiety testowe",
            "selected_genes.txt": "lista wybranych genów (kolejność cech)",
            "manifest.json": "pełna dokumentacja przygotowania",
        },
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("X_train", "X_test", "y_train", "y_test"):
            z.writestr(f"{name}.{fmt}", _table_bytes(result[name], fmt))
        z.writestr("selected_genes.txt", "\n".join(result["selected_genes"]))
        z.writestr("manifest.json", json.dumps(result["manifest"], ensure_ascii=False, indent=2))
        z.writestr("README.txt", _readme_text(fmt))
    return buf.getvalue()
