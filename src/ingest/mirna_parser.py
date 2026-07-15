"""Parser plików miRNA Expression Quantification z przepływu pracy GDC miRNA-Seq."""

__author__ = "Łukasz Połaski"

from pathlib import Path

import polars as pl

REQUIRED_COLUMNS: list[str] = [
    "miRNA_ID",
    "read_count",
    "reads_per_million_miRNA_mapped",
    "cross-mapped",
]

COUNT_COLUMN = "read_count"
RPM_COLUMN = "reads_per_million_miRNA_mapped"

# Nazwy miRBase zaczynają się od hsa- (Homo sapiens); mir/let to dwie rodziny nazw.
_MIRNA_PREFIXES = ("hsa-mir", "hsa-let", "hsa-miR")


class MirnaParserError(Exception):
    """Zgłaszany, gdy plik miRNA ma nieprawidłowy format lub nie przejdzie walidacji."""


def parse_mirna_quantification(path: str | Path) -> pl.DataFrame:
    """Analizuje pojedynczy plik miRNA Expression Quantification z GDC.

    Format pliku (miRBase v21, GRCh38) to tabela z tabulatorami:
    - ``miRNA_ID`` — czytelny identyfikator miRBase (np. ``hsa-let-7a-1``,
      ``hsa-mir-21``); w przeciwieństwie do RNA-seq nie wymaga mapowania na symbol,
    - ``read_count`` — surowe zliczenia,
    - ``reads_per_million_miRNA_mapped`` — znormalizowane RPM,
    - ``cross-mapped`` — flaga ``Y``/``N`` (odczyt mapuje się na wiele miRNA).

    Plik ma jeden wiersz na miRNA prekursorowe (~1900 pozycji), bez wierszy meta —
    prostszy niż STAR-Counts. Poziom izoform (MIMAT) jest w osobnym pliku
    isoforms.quantification i nie jest tu obsługiwany.

    Argumenty:
        path: Ścieżka do pliku ``.txt``/``.tsv`` miRNA quantification.

    Zwraca:
        Obiekt polars DataFrame z REQUIRED_COLUMNS, gdzie ``read_count`` jest Int64,
        a RPM Float64.

    Zgłasza:
        FileNotFoundError: Jeśli plik nie istnieje.
        MirnaParserError: Jeśli brakuje wymaganych kolumn, identyfikatory nie są
            nazwami miRBase, lub występują duplikaty miRNA_ID.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku miRNA: {path}")

    df = pl.read_csv(
        path,
        separator="\t",
        comment_prefix="#",
        has_header=True,
        null_values=["", "NA", "-"],
        infer_schema_length=10000,
    )

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise MirnaParserError(f"Brak wymaganych kolumn w {path.name}: {sorted(missing)}")

    df = df.select(REQUIRED_COLUMNS)

    invalid = df.filter(~pl.col("miRNA_ID").str.starts_with("hsa-"))
    if invalid.height > 0:
        sample = invalid["miRNA_ID"].head(5).to_list()
        raise MirnaParserError(
            f"Identyfikatory inne niż miRBase (hsa-) w {path.name}: {sample}"
        )

    if df["miRNA_ID"].n_unique() != df.height:
        raise MirnaParserError(f"Znaleziono zduplikowane identyfikatory miRNA w {path.name}")

    df = df.with_columns([
        pl.col(COUNT_COLUMN).cast(pl.Int64),
        pl.col(RPM_COLUMN).cast(pl.Float64),
    ])

    return df
