"""Warstwa eksportu artefaktów pipeline'u LUAD-HUBA.

Zawiera moduły do generowania metadanych reproducibility (manifest),
zapisu macierzy i datasetów w różnych formatach. Wzorzec analogiczny
do innych warstw (ingest, validate, transform).
"""

from src.export.manifest import build_manifest, save_manifest

__all__ = [
    "build_manifest",
    "save_manifest",
]
