"""Wczytywanie i walidacja plików konfiguracyjnych YAML dla CLI pipeline'u LUAD-HUBA.

Pozwala definiować parametry pipeline'u (project_id, workflow_type, metryki
normalizacji, progi filtrów) w plikach YAML zamiast podawania ich za każdym
razem przez flagi CLI. Domyślne configi znajdują się w katalogu configs/.

Pierwszeństwo wartości:
    1. flaga CLI (jeśli użytkownik ją podał)
    2. wartość z pliku YAML (jeśli --config podany)
    3. wartość domyślna w kodzie

Bez flagi --config pipeline działa identycznie jak dotąd (backward compatible).
"""

__author__ = "Łukasz Połaski"

from pathlib import Path
from typing import Any

import yaml


METRIC_ALIASES: dict[str, str] = {
    "tpm": "tpm_unstranded",
    "counts": "unstranded",
    "raw_counts": "unstranded",
    "fpkm": "fpkm_unstranded",
    "fpkm_uq": "fpkm_uq_unstranded",
}

FULL_METRIC_NAMES: set[str] = {
    "unstranded",
    "stranded_first",
    "stranded_second",
    "tpm_unstranded",
    "fpkm_unstranded",
    "fpkm_uq_unstranded",
}


class ConfigError(Exception):
    """Błąd wczytywania lub walidacji pliku konfiguracyjnego YAML."""


def load_config(path: Path) -> dict[str, Any]:
    """Wczytuje plik YAML z parametrami pipeline'u.

    Argumenty:
        path: ścieżka do pliku YAML (np. configs/default.yaml).

    Zwraca:
        Słownik z parametrami konfiguracyjnymi. Dostęp przez zagnieżdżone klucze,
        np. config["normalization"]["method"]. Pusty dict jeśli plik YAML jest pusty.

    Rzuca:
        ConfigError: gdy plik nie istnieje, ma niepoprawny format YAML, lub
            top-level nie jest słownikiem.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Plik konfiguracyjny nie istnieje: {path}")

    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Niepoprawny YAML w {path.name}: {exc}") from exc

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ConfigError(
            f"Oczekiwano słownika na top-level w {path.name}, "
            f"otrzymano: {type(data).__name__}"
        )

    return data


def get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Bezpieczne pobranie zagnieżdżonej wartości z configu.

    Przykład:
        get_nested(config, "pipeline", "project_id", default="TCGA-LUAD")
        # zwraca config["pipeline"]["project_id"] albo "TCGA-LUAD"

    Argumenty:
        config: słownik z load_config.
        *keys: ścieżka kluczy (od najwyższego poziomu).
        default: wartość zwracana gdy ścieżka nie istnieje.

    Zwraca:
        Wartość z configu lub default.
    """
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def resolve_metric(config_metric: str) -> str:
    """Mapuje alias metryki z configu na pełną nazwę kolumny w STAR-Counts.

    YAML używa krótkich form (tpm, counts, fpkm), kod pracuje na pełnych
    nazwach kolumn z plików GDC (tpm_unstranded, unstranded itp.).

    Argumenty:
        config_metric: nazwa metryki z YAML (alias lub pełna nazwa).

    Zwraca:
        Pełna nazwa kolumny (jedna z FULL_METRIC_NAMES).

    Rzuca:
        ConfigError: gdy nazwa nie jest ani aliasem, ani pełną nazwą.
    """
    if config_metric in METRIC_ALIASES:
        return METRIC_ALIASES[config_metric]

    if config_metric in FULL_METRIC_NAMES:
        return config_metric

    raise ConfigError(
        f"Niepoprawna metryka w configu: '{config_metric}'. "
        f"Dostępne aliasy: {sorted(METRIC_ALIASES)}, "
        f"pełne nazwy: {sorted(FULL_METRIC_NAMES)}"
    )
