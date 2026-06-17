"""Wspólna logika zarządzania danymi: archiwizacja (backup ZIP) i bezpieczne
kasowanie zakresów danych pipeline'u.

Jedno źródło prawdy dla GUI (Streamlit) i terminala (Textual) — oba interfejsy
importują stąd definicje zakresów i operacje, żeby zachowanie (gwarancja
bezpieczeństwa, tryby shallow/recursive, pakowanie) było identyczne.
"""

from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
DATA_RAW = DATA_ROOT / "raw"
DATA_UPLOADED_STAR = DATA_RAW / "uploaded_star"
DATA_INTERIM = DATA_ROOT / "interim" / "star_counts"
DATA_PROCESSED = DATA_ROOT / "processed"


def is_within_data(path: Path) -> bool:
    """Twarda gwarancja bezpieczeństwa: czy ścieżka leży wewnątrz katalogu data/.

    Chroni operacje kasowania - nigdy nie pozwala usunąć czegokolwiek poza
    PROJECT_ROOT/data, nawet przy błędzie w konfiguracji ścieżek (neutralizuje
    też próby ucieczki przez "..").
    """
    try:
        resolved = path.resolve()
        data_resolved = DATA_ROOT.resolve()
        return resolved == data_resolved or data_resolved in resolved.parents
    except Exception:
        return False


def iter_scope_files(path: Path, mode: str) -> list[Path]:
    """Listuje pliki w zakresie, z filtrem plików ukrytych (np. .gitkeep).

    mode='shallow' - tylko pliki bezpośrednio w katalogu (bez podkatalogów);
    używane dla 'Metadane kohorty' (data/raw bez uploaded_star).
    mode='recursive' - wszystkie pliki rekurencyjnie (cały katalog).
    """
    if not path.exists():
        return []
    if mode == "shallow":
        return sorted(f for f in path.iterdir()
                      if f.is_file() and not f.name.startswith("."))
    return sorted(f for f in path.rglob("*")
                  if f.is_file() and not f.name.startswith("."))


def scope_stats(path: Path, mode: str) -> tuple[int, int]:
    """Zwraca (liczba_plików, łączny_rozmiar_w_bajtach) dla zakresu."""
    files = iter_scope_files(path, mode)
    total = 0
    for f in files:
        try:
            total += f.stat().st_size
        except Exception:
            pass
    return len(files), total


def fmt_size(num_bytes: int) -> str:
    """Czytelny rozmiar (B/KB/MB/GB)."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def delete_scope(path: Path, mode: str,
                 progress_callback: "Callable[[int, int, str], None] | None" = None
                 ) -> tuple[int, list[str]]:
    """Usuwa pliki w zakresie (shallow/recursive), z gwarancją bezpieczeństwa.

    Odmawia działania, jeśli ścieżka nie jest wewnątrz data/. W trybie shallow
    usuwa tylko pliki bezpośrednio w katalogu (zachowuje podkatalogi, np.
    uploaded_star przy kasowaniu metadanych). progress_callback (idx, total,
    nazwa) wywoływany po każdym pliku. Zwraca (liczba_usuniętych, błędy).
    """
    if not is_within_data(path):
        return 0, [f"ODMOWA: ścieżka {path} poza katalogiem data/ - operacja zablokowana"]
    files = iter_scope_files(path, mode)
    total = len(files)
    deleted = 0
    errors = []
    for idx, f in enumerate(files, start=1):
        try:
            f.unlink()
            deleted += 1
        except Exception as exc:
            errors.append(f"{f.name}: {exc}")
        if progress_callback is not None:
            progress_callback(idx, total, f.name)
    return deleted, errors


def _collect_archive_files(targets: list[tuple[str, Path, str]]) -> list[tuple[Path, str]]:
    """Zbiera (plik, nazwa_w_archiwum) dla wszystkich zakresów (wspólne dla obu wariantów)."""
    all_files: list[tuple[Path, str]] = []
    for label, path, mode in targets:
        for f in iter_scope_files(path, mode):
            arcname = f"{label}/{f.name}" if mode == "shallow" else f"{label}/{f.relative_to(path)}"
            all_files.append((f, arcname))
    return all_files


def build_archive_zip(targets: list[tuple[str, Path, str]],
                      progress_callback: "Callable[[int, int, str], None] | None" = None
                      ) -> bytes:
    """Pakuje wybrane zakresy do archiwum ZIP w pamięci (zwraca bajty).

    Używane przez GUI do pobrania przez przeglądarkę. Dla dużych kohort (GB)
    trzyma całość w pamięci — w terminalu użyj build_archive_to_path.

    targets: lista (etykieta_w_archiwum, ścieżka, tryb). W trybie shallow pakuje
    tylko pliki bezpośrednio w katalogu, w recursive zachowuje strukturę względną.
    progress_callback (idx, total, nazwa) wywoływany po każdym spakowanym pliku.
    """
    import io
    import zipfile
    all_files = _collect_archive_files(targets)
    total = len(all_files)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, (f, arcname) in enumerate(all_files, start=1):
            zf.write(f, arcname)
            if progress_callback is not None:
                progress_callback(idx, total, f.name)
    return buf.getvalue()


def build_archive_to_path(targets: list[tuple[str, Path, str]],
                          out_path: Path,
                          progress_callback: "Callable[[int, int, str], None] | None" = None
                          ) -> tuple[int, int]:
    """Pakuje wybrane zakresy do pliku ZIP na dysku (strumieniowo, bez pamięci).

    Wariant dla terminala — pisze plik po pliku prosto do archiwum, więc bezpieczny
    dla gigabajtów surowych STAR. Zwraca (liczba_plików, rozmiar_archiwum_w_bajtach).
    """
    import zipfile
    all_files = _collect_archive_files(targets)
    total = len(all_files)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, (f, arcname) in enumerate(all_files, start=1):
            zf.write(f, arcname)
            if progress_callback is not None:
                progress_callback(idx, total, f.name)
    size = out_path.stat().st_size if out_path.exists() else 0
    return total, size


# Definicje zakresów zarządzania: klucz -> (etykieta, ścieżka, tryb, opis)
# 'metadata' używa trybu shallow (data/raw BEZ uploaded_star), reszta recursive.
MANAGE_SCOPES = {
    "star": ("Pliki STAR-Counts", DATA_UPLOADED_STAR, "recursive",
             "Surowe pliki ekspresji (data/raw/uploaded_star) — zwykle gigabajty"),
    "metadata": ("Metadane kohorty", DATA_RAW, "shallow",
                 "clinical.tsv, sample sheet, metadata.cart.json (bez plików STAR)"),
    "interim": ("Parquety pośrednie", DATA_INTERIM, "recursive",
                "Sparsowane pliki STAR (data/interim/star_counts)"),
    "processed": ("Wyniki finalne", DATA_PROCESSED, "recursive",
                  "Macierz ekspresji, zbiór przeżywalności, manifesty"),
}
