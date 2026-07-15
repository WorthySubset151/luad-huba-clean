# -*- coding: utf-8 -*-
"""Wspólne fixtures testowe.

Dane są syntetyczne i generowane w locie — testy nie wymagają pobrania kohorty
z GDC ani żadnych plików w ``data/``, więc działają na świeżym klonie repozytorium.
Format wejściowy odwzorowuje eksport „cart" z portalu GDC (kolumny z prefiksami
``cases.`` / ``demographic.`` / ``diagnoses.``, markery braków w stylu GDC).
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

N_PATIENTS = 60
N_GENES = 80
GDC_NULL = "'--"

CLINICAL_HEADER = [
    "cases.submitter_id",
    "cases.primary_site",
    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.age_at_index",
    "demographic.gender",
    "demographic.race",
    "diagnoses.diagnosis_is_primary_disease",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.ajcc_pathologic_stage",
]

STAGES = ["Stage IA", "Stage IB", "Stage IIA", "Stage IIB", "Stage IIIA", "Stage IV"]


def _case_id(i: int) -> str:
    """Barkod pacjenta zgodny ze wzorcem TCGA (TCGA-XX-YYYY)."""
    return f"TCGA-44-{i:04d}"


@pytest.fixture(scope="session")
def clinical_tsv_path(tmp_path_factory) -> str:
    """Ścieżka do syntetycznego clinical.tsv w formacie eksportu „cart" GDC.

    Odwzorowuje realną strukturę: plik jest zdenormalizowany (pacjent ma wiele
    wierszy — jedną diagnozę podstawową i dodatkowe niepodstawowe, które parser
    ma odfiltrować), a wiersze niepodstawowe mają puste stadium i czas obserwacji.
    """
    rng = np.random.default_rng(7)
    rows: list[list[str]] = []
    for i in range(N_PATIENTS):
        case = _case_id(i)
        dead = bool(rng.integers(0, 2))
        vital = "Dead" if dead else "Alive"
        days_death = str(int(rng.integers(30, 3000))) if dead else GDC_NULL
        age = str(int(rng.integers(40, 86)))
        gender = str(rng.choice(["male", "female"]))
        race = str(rng.choice(["white", "black or african american", "not reported"]))
        follow_up = GDC_NULL if dead else str(int(rng.integers(30, 3000)))
        stage = str(rng.choice(STAGES))
        # wiersz diagnozy podstawowej — komplet danych
        rows.append([case, "Bronchus and lung", vital, days_death, age, gender, race,
                     "true", follow_up, stage])
        # wiersze niepodstawowe — parser ma je odrzucić (stąd braki)
        for _ in range(int(rng.integers(0, 3))):
            rows.append([case, "Bronchus and lung", vital, days_death, age, gender, race,
                         "false", GDC_NULL, GDC_NULL])

    path = tmp_path_factory.mktemp("gdc") / "clinical.tsv"
    lines = ["\t".join(CLINICAL_HEADER)] + ["\t".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture(scope="session")
def clinical(clinical_tsv_path) -> pl.DataFrame:
    """Sparsowana klinika (jeden wiersz na pacjenta)."""
    from src.ingest.clinical_parser import parse_clinical

    return parse_clinical(clinical_tsv_path)


@pytest.fixture(scope="session")
def expression_inputs(clinical):
    """Syntetyczna macierz ekspresji + sample sheet dla pacjentów z kliniki.

    Część pacjentów ma dwie próbki — pozwala testować deduplikację i podział
    train/test grupowany po pacjencie.
    """
    rng = np.random.default_rng(11)
    cases = clinical["case_submitter_id"].to_list()
    n_dup = max(1, len(cases) // 10)
    sample_cases = cases + cases[:n_dup]
    samples = [f"{c}-01A" for c in cases] + [f"{c}-02A" for c in cases[:n_dup]]

    genes = [f"ENSG{i:011d}" for i in range(N_GENES)]
    matrix_data: dict[str, object] = {"gene_id": genes}
    for sample in samples:
        matrix_data[sample] = rng.uniform(0.0, 100.0, N_GENES)
    # dwa geny stałe — do testu filtra zerowej wariancji
    for sample in samples:
        matrix_data[sample][0] = 5.0
        matrix_data[sample][1] = 0.0

    sheet = pl.DataFrame({
        "sample_id": samples,
        "case_id": sample_cases,
        "tissue_type": ["Primary Tumor"] * len(samples),
        "is_tumor": [True] * len(samples),
    })
    return pl.DataFrame(matrix_data), sheet


@pytest.fixture(scope="session")
def survival_ds(clinical, expression_inputs) -> pl.DataFrame:
    """Zbudowany zbiór przeżywalności (metadane + kowarianty + geny)."""
    from src.transform.survival_dataset import build_survival_dataset

    matrix, sheet = expression_inputs
    return build_survival_dataset(matrix, sheet, clinical)


@pytest.fixture
def gene_columns(survival_ds) -> list[str]:
    return [c for c in survival_ds.columns if c.startswith("ENSG")]
