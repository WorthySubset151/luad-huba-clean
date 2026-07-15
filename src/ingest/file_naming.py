"""Stałe i funkcje pomocnicze do obsługi konwencji nazewniczych plików STAR-Counts."""

__author__ = "Łukasz Połaski"

from pathlib import Path

STAR_FILE_PATTERNS: list[str] = [
    "*.rna_seq.augmented_star_gene_counts.tsv",
    "*_rna_seq_augmented_star_gene_counts.tsv",
]

STAR_FILE_SUFFIXES: list[str] = [
    ".rna_seq.augmented_star_gene_counts.tsv",
    "_rna_seq_augmented_star_gene_counts.tsv",
]

MIRNA_FILE_SUFFIXES: list[str] = [
    ".mirbase21.mirnas.quantification.txt",
    "_mirbase21_mirnas_quantification.txt",
    ".mirnas.quantification.txt",
]


def extract_star_file_stem(file_name: str | Path) -> str:
    """Wyciąga identyfikator pliku STAR-Counts z jego nazwy.

    Obsługuje obie konwencje nazewnicze (z kropkami pochodzącymi z GDC oraz
    z podkreśleniami pojawiającymi się czasem po przetwarzaniu na macOS).
    Dla pliku ``11d52676-...rna_seq.augmented_star_gene_counts.tsv`` zwróci
    sam UUID ``11d52676-...``.

    Argumenty:
        file_name: Nazwa pliku lub pełna ścieżka do pliku STAR-Counts (TSV
            lub przetworzonego Parquet).

    Zwraca:
        Stem pliku - identyfikator UUID bez sufiksu STAR-Counts ani rozszerzenia.
    """
    name = Path(file_name).name
    for suffix in STAR_FILE_SUFFIXES:
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return Path(file_name).stem


def extract_mirna_file_stem(file_name: str | Path) -> str:
    """Wyciąga identyfikator pliku miRNA quantification z jego nazwy.

    Analogicznie do ``extract_star_file_stem``, ale dla sufiksów miRNA-Seq GDC
    (np. ``<uuid>.mirbase21.mirnas.quantification.txt``). Zwraca UUID pliku bez
    sufiksu miRNA ani rozszerzenia; gdy żaden sufiks nie pasuje, zwraca zwykły stem.
    """
    name = Path(file_name).name
    for suffix in MIRNA_FILE_SUFFIXES:
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return Path(file_name).stem
