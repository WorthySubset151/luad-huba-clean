"""Parser plików STAR-Counts wygenerowanych przez przepływ pracy GDC RNA-Seq."""

__author__ = "Łukasz Połaski"

from pathlib import Path

import polars as pl

META_ROW_IDS: set[str] = {
    "N_unmapped",
    "N_multimapping",
    "N_noFeature",
    "N_ambiguous",
}

REQUIRED_COLUMNS: list[str] = [
    "gene_id",
    "gene_name",
    "gene_type",
    "unstranded",
    "stranded_first",
    "stranded_second",
    "tpm_unstranded",
    "fpkm_unstranded",
    "fpkm_uq_unstranded",
]

COUNT_COLUMNS: list[str] = ["unstranded", "stranded_first", "stranded_second"]


class StarParserError(Exception):
    """Zgłaszany, gdy plik STAR-Counts ma nieprawidłowy format lub nie przejdzie walidacji."""


def parse_star_counts(path: str | Path) -> pl.DataFrame:
    """Analizuje pojedynczy plik STAR-Counts z przepływu pracy GDC RNA-Seq.

    Format pliku zawiera:
    - Jedną lub więcej linii komentarza poprzedzonych znakiem ``#`` (pomijane).
    - Wiersz nagłówka z 9 kolumnami oddzielonymi tabulatorami.
    - Cztery wiersze meta (``N_unmapped``, ``N_multimapping``, ``N_noFeature``,
      ``N_ambiguous``) raportujące statystyki dopasowania STAR. Są one usuwane.
    - Jeden wiersz na gen, gdzie gene_id jest wersjonowanym identyfikatorem Ensembl
      (np. ``ENSG00000000003.15``).

    Argumenty:
        path: Ścieżka do pliku ``.tsv`` STAR-Counts.

    Zwraca:
        Obiekt polars DataFrame zawierający REQUIRED_COLUMNS, z wykluczeniem wierszy meta,
        gdzie kolumny zliczeń są zrzutowane na typ Int64.

    Zgłasza:
        FileNotFoundError: Jeśli plik nie istnieje.
        StarParserError: Jeśli w pliku brakuje wymaganych kolumn, zawiera on identyfikatory
            genów inne niż Ensembl po filtrowaniu wierszy meta, lub zawiera zduplikowane
            identyfikatory genów.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku STAR-Counts: {path}")

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
        raise StarParserError(
            f"Brak wymaganych kolumn w {path.name}: {sorted(missing)}"
        )

    df = df.select(REQUIRED_COLUMNS)
    df = df.filter(~pl.col("gene_id").is_in(META_ROW_IDS))

    invalid = df.filter(~pl.col("gene_id").str.starts_with("ENSG"))
    if invalid.height > 0:
        sample = invalid["gene_id"].head(5).to_list()
        raise StarParserError(
            f"Identyfikatory genów inne niż Ensembl po filtrowaniu meta w {path.name}: {sample}"
        )

    if df["gene_id"].n_unique() != df.height:
        raise StarParserError(f"Znaleziono zduplikowane identyfikatory genów w {path.name}")

    df = df.with_columns([pl.col(c).cast(pl.Int64) for c in COUNT_COLUMNS])

    return df
