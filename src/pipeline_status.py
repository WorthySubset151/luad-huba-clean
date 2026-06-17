"""Status pipeline'u ETL LUAD-HUBA — czysta inspekcja dysku (headless).

Sprawdza, które artefakty pipeline'u istnieją i w jakim są stanie (liczba
rekordów, rozmiar, data modyfikacji), oraz wylicza gotowość kolejnych etapów na
podstawie zależności — dokładnie jak ``detect_state`` w dashboardzie, tylko jako
czyste dane do dowolnego renderowania (terminal rysuje z tego listę zbiorów w
stylu z/OS).

Żadnych obliczeń statystycznych ani I/O poza odczytem metadanych parquet
(``read_parquet_schema`` + ``pl.len()`` na lazy frame — bez wczytywania danych).
"""

__author__ = "Łukasz Połaski"

from datetime import datetime
from pathlib import Path

import polars as pl

# Ścieżki zgodne z app/main.py (jedna konwencja katalogów dla całego projektu).
REL_RAW = ("data", "raw")
REL_INTERIM = ("data", "interim", "star_counts")
REL_PROCESSED = ("data", "processed")
REL_CONFIG = ("configs", "default.yaml")


def _human_size(n: int | None) -> str:
    if n is None:
        return "—"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _mtime(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return None


def _parquet_dims(path: Path):
    """(rows, cols) z metadanych parquet — bez wczytywania danych. None gdy błąd."""
    try:
        cols = len(pl.read_parquet_schema(path))
        rows = pl.scan_parquet(path).select(pl.len()).collect().item()
        return int(rows), int(cols)
    except Exception:  # noqa: BLE001 — uszkodzony/niepełny plik
        return None, None


def _file_artifact(name: str, path: Path, root: Path, *, parquet: bool = False) -> dict:
    exists = path.exists() and path.is_file()
    art = {
        "name": name,
        "rel_path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "exists": exists,
        "rows": None,
        "cols": None,
        "size": path.stat().st_size if exists else None,
        "mtime": _mtime(path) if exists else None,
        "note": "",
    }
    if exists and parquet:
        rows, cols = _parquet_dims(path)
        art["rows"], art["cols"] = rows, cols
        if rows is None:
            art["note"] = "nieczytelny parquet"
    return art


def _dir_artifact(name: str, directory: Path, pattern: str, root: Path) -> dict:
    files = sorted(directory.glob(pattern)) if directory.exists() else []
    total = sum(f.stat().st_size for f in files) if files else None
    latest = max((f.stat().st_mtime for f in files), default=None) if files else None
    mtime = datetime.fromtimestamp(latest).strftime("%Y-%m-%d %H:%M") if latest else None
    return {
        "name": name,
        "rel_path": str(directory.relative_to(root)) + f"/{pattern}"
        if directory.is_relative_to(root) else f"{directory}/{pattern}",
        "exists": len(files) > 0,
        "rows": None,
        "cols": None,
        "count": len(files),
        "size": total,
        "mtime": mtime,
        "note": f"{len(files)} plik(ów)",
    }


def pipeline_status(root: Path) -> dict:
    """Pełny status pipeline'u: artefakty + gotowość etapów (czyste dane)."""
    root = Path(root)
    data_raw = root.joinpath(*REL_RAW)
    data_interim = root.joinpath(*REL_INTERIM)
    data_processed = root.joinpath(*REL_PROCESSED)
    config_path = root.joinpath(*REL_CONFIG)

    clinical = data_raw / "clinical.tsv"
    sheets = sorted(data_raw.glob("gdc_sample_sheet*.tsv")) if data_raw.exists() else []
    matrix = data_processed / "expression_matrix.parquet"
    survival = data_processed / "survival_dataset.parquet"

    # Artefakty (lista zbiorów w stylu z/OS).
    artifacts = [
        _file_artifact("clinical.tsv", clinical, root),
        _dir_artifact("gdc_sample_sheet*.tsv", data_raw, "gdc_sample_sheet*.tsv", root),
        _dir_artifact("STAR counts (interim)", data_interim, "*.parquet", root),
        _file_artifact("expression_matrix.parquet", matrix, root, parquet=True),
        _file_artifact("survival_dataset.parquet", survival, root, parquet=True),
        _file_artifact("configs/default.yaml", config_path, root),
    ]

    # Flagi gotowości (jak detect_state w dashboardzie).
    has_clinical = clinical.exists()
    has_sheet = len(sheets) > 0
    parsed_count = len(list(data_interim.glob("*.parquet"))) if data_interim.exists() else 0
    matrix_built = matrix.exists()
    survival_built = survival.exists()

    raw_ready = has_clinical and has_sheet
    parsed = parsed_count > 0

    # Etapy z zależnościami: status = ok / brak / zablokowany (gdy poprzednik niegotowy).
    def stage(label, ok, dep_ready, dep_label, detail):
        if ok:
            status = "ok"
        elif dep_ready:
            status = "missing"
        else:
            status = "blocked"
        return {"label": label, "status": status, "blocked_by": dep_label,
                "detail": detail}

    stages = [
        stage("Wejście surowe (clinical + sample sheet)", raw_ready, True, None,
              ("clinical.tsv " + ("✓" if has_clinical else "✗")
               + "  sample sheet " + ("✓" if has_sheet else "✗"))),
        stage("Parsowanie STAR → interim", parsed, raw_ready, "wejście surowe",
              f"{parsed_count} parquet(ów) w data/interim/star_counts"),
        stage("Macierz ekspresji", matrix_built, parsed, "parsowanie",
              "data/processed/expression_matrix.parquet"),
        stage("Zbiór przeżywalności", survival_built, matrix_built, "macierz ekspresji",
              "data/processed/survival_dataset.parquet"),
        stage("Analiza (dashboard / terminal)", survival_built, survival_built,
              "zbiór przeżywalności",
              "panele SURVIVAL/STATUS gotowe do użycia" if survival_built
              else "wymaga zbioru przeżywalności"),
    ]

    n_done = sum(1 for s in stages if s["status"] == "ok")
    return {
        "root": str(root),
        "config_exists": config_path.exists(),
        "artifacts": artifacts,
        "stages": stages,
        "stages_done": n_done,
        "stages_total": len(stages),
    }
