# -*- coding: utf-8 -*-
"""Modalność danych — jedno miejsce definiujące, czym jest cecha w zbiorze.

Rdzeń pipeline'u zakładał wcześniej wprost, że cecha to gen z prefiksem ``ENSG``
(sprawdzenie ``startswith("ENSG")`` powtórzone w kilku modułach). Blokowało to
dodanie kolejnych modalności GDC — miRNA (``hsa-mir-…``), ekspresji białek (RPPA)
— bo każda wymagałaby dotknięcia wszystkich tych miejsc.

Modalność opisuje, jak rozpoznać kolumny cech i jak o nich mówić. Dodanie nowej
modalności to dopisanie instancji do :data:`REGISTRY`, a nie zmiana rdzenia.

Uwaga o zakresie: modalność parametryzuje analizy **generyczne** (podsumowanie
kohorty, gotowość ML, eksport). Analizy oparte na panelu genów LUAD (sygnatura
wielogenowa, model Coxa na panelu) pozostają swoiste dla RNA-seq — panel siedmiu
genów nie ma odpowiednika w miRNA, więc parametr modalności byłby tam pozorny.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Modality:
    """Opis modalności danych omicznych.

    Atrybuty:
        id: Identyfikator techniczny (klucz w rejestrze, wartość w manifeście).
        label: Nazwa czytelna dla człowieka (etykiety GUI).
        feature_prefix: Prefiks kolumn cech w zbiorze (np. ``ENSG``).
        feature_noun: Rzeczownik opisujący pojedynczą cechę (np. ``gen``).
        feature_noun_plural: Ten sam rzeczownik w liczbie mnogiej.
        default_metric: Typowa metryka ilościowa modalności (np. ``TPM``).
    """

    id: str
    label: str
    feature_prefix: str
    feature_noun: str
    feature_noun_plural: str
    default_metric: str

    def feature_columns(self, frame) -> list[str]:
        """Kolumny cech w ramce (polars lub pandas), w kolejności występowania."""
        return [c for c in frame.columns if c.startswith(self.feature_prefix)]

    def has_features(self, frame) -> bool:
        """Czy ramka zawiera choć jedną kolumnę cech tej modalności."""
        return any(c.startswith(self.feature_prefix) for c in frame.columns)


RNASEQ = Modality(
    id="rnaseq",
    label="Ekspresja genów (RNA-seq)",
    feature_prefix="ENSG",
    feature_noun="gen",
    feature_noun_plural="geny",
    default_metric="TPM",
)

REGISTRY: dict[str, Modality] = {RNASEQ.id: RNASEQ}

DEFAULT_MODALITY: Modality = RNASEQ


def get_modality(modality_id: str) -> Modality:
    """Zwraca modalność po identyfikatorze.

    Zgłasza:
        KeyError: Gdy identyfikator nie występuje w rejestrze.
    """
    try:
        return REGISTRY[modality_id]
    except KeyError as exc:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Nieznana modalność: {modality_id!r} (dostępne: {known}).") from exc


def detect_modality(frame) -> Modality | None:
    """Rozpoznaje modalność zbioru po prefiksach kolumn.

    Zwraca modalność z największą liczbą pasujących kolumn albo ``None``, gdy
    żadna nie pasuje. Przy zbiorach wielomodalnych wynik jest jedynie wskazówką —
    modalność należy wtedy podać wprost.
    """
    best: Modality | None = None
    best_count = 0
    for modality in REGISTRY.values():
        count = len(modality.feature_columns(frame))
        if count > best_count:
            best, best_count = modality, count
    return best
