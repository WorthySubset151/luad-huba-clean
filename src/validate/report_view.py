"""Klasyfikacja i werdykt raportu QC — wspólne dla GUI i terminala.

Jedno źródło prawdy dla prezentacji walidacji kohorty: które rozjazdy pipeline
obsługuje sam (typowe dla TCGA), jak je nazywać, co z nimi robi pipeline oraz
jak brzmi werdykt zbiorczy. Dashboard (Streamlit) i terminal (Textual) konsumują
tę samą klasyfikację, więc nie mogą się rozejść w ocenie spójności kohorty.
"""

__author__ = "Łukasz Połaski"

# Kategorie rozjazdów, które pipeline obsługuje automatycznie (bez interwencji).
HANDLED_BY_PIPELINE = {
    "missing_clinical", "missing_survival", "orphan_star_file", "duplicate_sample",
}

# Czytelne etykiety kategorii.
CATEGORY_LABELS = {
    "missing_star_file": "Brakujący plik STAR",
    "orphan_star_file": "Osierocony plik STAR (bez próbki)",
    "missing_clinical": "Brak danych klinicznych",
    "duplicate_sample": "Duplikat próbki",
    "missing_survival": "Brak danych przeżycia",
}

# Co pipeline robi z każdą kategorią (wyjaśnienie dla użytkownika).
PIPELINE_ACTION = {
    "missing_clinical": "Pomijane przy budowie zbioru przeżywalności (brak danych klinicznych).",
    "missing_survival": "Pomijane przy budowie zbioru przeżywalności (brak czasu obserwacji lub statusu).",
    "orphan_star_file": "Ignorowane przy budowie macierzy (plik STAR bez próbki w sample sheet).",
    "duplicate_sample": "Obsługiwane przez strategię duplikatów (deepest/first) przy budowie macierzy.",
    "missing_star_file": "Próbka nie wejdzie do macierzy — jeśli oczekiwano pliku STAR, sprawdź pobieranie.",
}


def _group_by_category(items: list) -> dict:
    """Grupuje problemy po kategorii, zachowując kolejność wystąpienia."""
    cats: dict = {}
    for i in items:
        cats.setdefault(i["category"], []).append(i)
    return cats


def classify_qc(summary: dict, issues: list) -> dict:
    """Dzieli problemy QC na obsługiwane vs wymagające uwagi i ustala werdykt.

    Argumenty:
        summary: wynik ``QCReport.summary()`` ({'total', 'errors', ...}).
        issues: lista zserializowanych problemów (dict z kluczami
            ``severity``, ``category``, ``message``, ``context``).

    Zwraca strukturę gotową do renderu (Streamlit lub Rich) — czyste dane,
    bez zależności od warstwy prezentacji.
    """
    handled = [i for i in issues if i["category"] in HANDLED_BY_PIPELINE]
    action_needed = [i for i in issues if i["category"] not in HANDLED_BY_PIPELINE]

    if not issues:
        verdict_level = "ok"
        verdict = "Kohorta w pełni spójna — brak rozjazdów."
    elif not action_needed:
        verdict_level = "ok"
        verdict = (
            f"Kohorta gotowa do analizy. Wykryto {len(handled)} rozjazdów typowych "
            f"dla danych TCGA — wszystkie obsługiwane automatycznie przez pipeline "
            f"(pominięcie przy budowie zbioru przeżywalności lub deduplikacja). "
            f"Nie wymagają ręcznej interwencji."
        )
    else:
        verdict_level = "warn"
        verdict = (
            f"Wykryto {len(action_needed)} rozjazdów wartych sprawdzenia (poniżej) "
            f"oraz {len(handled)} typowych dla TCGA (obsługiwanych automatycznie)."
        )

    return {
        "total": summary.get("total", len(issues)),
        "n_handled": len(handled),
        "n_action": len(action_needed),
        "verdict": verdict,
        "verdict_level": verdict_level,
        "action_by_cat": _group_by_category(action_needed),
        "handled_by_cat": _group_by_category(handled),
        "labels": CATEGORY_LABELS,
        "actions": PIPELINE_ACTION,
    }
