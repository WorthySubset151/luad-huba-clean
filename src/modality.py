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
        feature_prefix: Prefiks (lub krotka prefiksów) kolumn cech w zbiorze.
            RNA-seq ma jeden prefiks (``ENSG``); miRNA ma ich kilka
            (``hsa-mir``, ``hsa-let``), bo nazwy miRBase nie mają wspólnego rdzenia
            poza ``hsa-`` — a ten byłby zbyt szeroki. Krotka jest tu jawnym sygnałem,
            że rozpoznanie po nazwie to uproszczenie; docelowo (multi-omics w jednym
            X) pochodzenie cechy powinna nieść osobna kolumna ``modality``.
        feature_noun: Rzeczownik opisujący pojedynczą cechę (np. ``gen``).
        feature_noun_plural: Ten sam rzeczownik w liczbie mnogiej.
        default_metric: Typowa metryka ilościowa modalności (np. ``TPM``).
        gdc_data_type: ``data_type`` w API GDC (do filtra pobierania).
        gdc_workflow_type: ``analysis.workflow_type`` w API GDC.
    """

    id: str
    label: str
    feature_prefix: str | tuple[str, ...]
    feature_noun: str
    feature_noun_plural: str
    default_metric: str
    gdc_data_type: str = ""
    gdc_workflow_type: str = ""

    @property
    def _prefixes(self) -> tuple[str, ...]:
        """Prefiksy jako krotka — niezależnie od tego, czy podano str czy tuple."""
        if isinstance(self.feature_prefix, str):
            return (self.feature_prefix,)
        return tuple(self.feature_prefix)

    def is_feature(self, name: str) -> bool:
        """Czy nazwa kolumny jest cechą tej modalności."""
        return name.startswith(self._prefixes)

    def feature_columns(self, frame) -> list[str]:
        """Kolumny cech w ramce (polars lub pandas), w kolejności występowania."""
        return [c for c in frame.columns if self.is_feature(c)]

    def has_features(self, frame) -> bool:
        """Czy ramka zawiera choć jedną kolumnę cech tej modalności."""
        return any(self.is_feature(c) for c in frame.columns)

    def gdc_filters(self) -> dict:
        """Filtry pobierania z GDC (data_type + workflow_type), jeśli zdefiniowane.

        Zwraca słownik gotowy do złożenia w zapytanie do API plików GDC. Pusty,
        gdy modalność nie deklaruje własnej ścieżki pobierania.
        """
        filters: dict = {}
        if self.gdc_data_type:
            filters["data_type"] = self.gdc_data_type
        if self.gdc_workflow_type:
            filters["workflow_type"] = self.gdc_workflow_type
        return filters


RNASEQ = Modality(
    id="rnaseq",
    label="Ekspresja genów (RNA-seq)",
    feature_prefix="ENSG",
    feature_noun="gen",
    feature_noun_plural="geny",
    default_metric="TPM",
    gdc_data_type="Gene Expression Quantification",
    gdc_workflow_type="STAR - Counts",
)

MIRNA = Modality(
    id="mirna",
    label="Ekspresja miRNA (miRNA-seq)",
    feature_prefix=("hsa-mir", "hsa-let", "hsa-miR"),
    feature_noun="miRNA",
    feature_noun_plural="miRNA",
    default_metric="RPM",
    gdc_data_type="miRNA Expression Quantification",
    gdc_workflow_type="BCGSC miRNA Profiling",
)

REGISTRY: dict[str, Modality] = {RNASEQ.id: RNASEQ, MIRNA.id: MIRNA}

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
