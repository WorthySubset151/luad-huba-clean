"""Generowanie i zapisywanie manifestów reproducibility dla artefaktów pipeline'u.

Manifest to JSON ze szczegółami runu - liczba próbek/genów, identyfikatory,
lista plików źródłowych, metryka, znacznik czasowy, hash treści. Pozwala
zweryfikować że dany plik wynikowy pochodzi z konkretnej kombinacji wejść
i parametrów, bez sięgania do plików źródłowych.

Funkcjonalność wcześniej żyła w src/transform/expression_matrix.py - tutaj
wydzielona jako osobna warstwa, żeby src/export/ przestało być pustym
szkieletem. Nowe artefakty pipeline'u (np. survival_dataset, embedding,
modele) będą mogły reużyć tej samej infrastruktury manifestu.
"""

__author__ = "Łukasz Połaski"

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


def build_manifest(
    matrix: pl.DataFrame,
    parquet_paths: list[Path],
    metric: str,
) -> dict:
    """Buduje słownik metadanych opisujący zbudowaną macierz ekspresji.

    Manifest dokumentuje: liczbę próbek i genów, identyfikatory próbek,
    listę plików źródłowych, metrykę, datę utworzenia i skrót zawartości.

    Argumenty:
        matrix: Wynik funkcji ``build_expression_matrix``.
        parquet_paths: Lista plików źródłowych w tej samej kolejności.
        metric: Nazwa zastosowanej metryki ekspresji.

    Zwraca:
        Słownik z polami: ``created_at``, ``metric``, ``n_samples``, ``n_genes``,
        ``sample_ids``, ``source_files``, ``content_hash``.
    """
    sample_ids = [c for c in matrix.columns if c != "gene_id"]
    content_hash = hashlib.sha256(
        matrix.write_csv().encode("utf-8")
    ).hexdigest()

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric": metric,
        "n_samples": len(sample_ids),
        "n_genes": matrix.height,
        "sample_ids": sample_ids,
        "source_files": [p.name for p in parquet_paths],
        "content_hash": content_hash,
    }


def save_manifest(manifest: dict, output_path: Path) -> None:
    """Zapisuje manifest jako sformatowany JSON.

    Argumenty:
        manifest: Słownik z ``build_manifest``.
        output_path: Ścieżka pliku JSON do utworzenia/nadpisania.
    """
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
