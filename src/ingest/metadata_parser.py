"""Parser pliku metadata.cart.json z portalu GDC dla projektu LUAD-HUBA."""

__author__ = "Łukasz Połaski"

import json
import re
from pathlib import Path

import polars as pl


REQUIRED_TOP_LEVEL_FIELDS: list[str] = [
    "file_id",
    "file_name",
    "md5sum",
    "file_size",
    "data_type",
    "experimental_strategy",
    "associated_entities",
    "analysis",
]

REQUIRED_ENTITY_FIELDS: list[str] = [
    "entity_submitter_id",
    "entity_type",
    "case_id",
    "entity_id",
]

REQUIRED_ANALYSIS_FIELDS: list[str] = [
    "workflow_type",
    "workflow_version",
    "input_files",
]

REQUIRED_INPUT_FILE_FIELDS: list[str] = [
    "total_reads",
    "proportion_reads_mapped",
    "average_base_quality",
    "average_read_length",
]

OUTPUT_COLUMNS: list[str] = [
    "file_id",
    "file_name",
    "md5sum",
    "file_size",
    "data_type",
    "experimental_strategy",
    "sample_id",
    "case_submitter_id",
    "aliquot_barcode",
    "case_uuid",
    "aliquot_uuid",
    "workflow_type",
    "workflow_version",
    "total_reads",
    "proportion_reads_mapped",
    "average_base_quality",
    "average_read_length",
]

ALIQUOT_BARCODE_PATTERN = re.compile(
    r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}[A-Z]-\d{2}[A-Z]-[A-Z0-9]{4}-\d{2}$"
)


class MetadataParserError(Exception):
    """Błąd parsowania pliku metadata.cart.json."""


def parse_metadata(path: Path) -> pl.DataFrame:
    """Parsuje plik metadata.cart.json z portalu GDC.

    Plik zawiera szczegółowe metadane każdego pliku w cart, w tym sumy
    kontrolne MD5 (do weryfikacji integralności po pobraniu), mapowanie
    na biologiczne identyfikatory (sample_id, case_submitter_id) oraz metryki
    QC z BAM-u upstream (proportion_reads_mapped, average_base_quality itd.).

    Argumenty:
        path: ścieżka do pliku metadata.cart.YYYY-MM-DD.json.

    Zwraca:
        DataFrame z jednym wierszem per plik, kolumny:
        - file_id, file_name, md5sum, file_size (integralność)
        - data_type, experimental_strategy (typ pliku)
        - sample_id, case_submitter_id, aliquot_barcode (mapowanie TCGA)
        - case_uuid, aliquot_uuid (mapowanie GDC)
        - workflow_type, workflow_version (audit trail)
        - total_reads, proportion_reads_mapped, average_base_quality,
          average_read_length (QC z BAM-u upstream)

    Rzuca:
        MetadataParserError: gdy struktura JSON jest niezgodna z oczekiwaną.
    """
    path = Path(path)
    if not path.exists():
        raise MetadataParserError(f"Plik nie istnieje: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MetadataParserError(f"Niepoprawny JSON w {path.name}: {exc}") from exc

    if not isinstance(data, list):
        raise MetadataParserError(
            f"Oczekiwano listy rekordów w {path.name}, otrzymano: {type(data).__name__}"
        )

    if len(data) == 0:
        raise MetadataParserError(f"Pusty plik metadata: {path.name}")

    rows = [_parse_record(record, path.name, idx) for idx, record in enumerate(data)]
    df = pl.DataFrame(rows)

    if df["file_id"].n_unique() != df.height:
        raise MetadataParserError(
            f"Duplikaty file_id w {path.name}: oczekiwano unikalnych UUID-ów"
        )

    return df.select(OUTPUT_COLUMNS).sort("file_id")


def _parse_record(record: dict, source_name: str, idx: int) -> dict:
    """Wyciąga płaski rekord z jednego wpisu metadata JSON."""
    missing_top = [f for f in REQUIRED_TOP_LEVEL_FIELDS if f not in record]
    if missing_top:
        raise MetadataParserError(
            f"Rekord #{idx} w {source_name}: brakuje pól top-level {missing_top}"
        )

    entities = record["associated_entities"]
    if not isinstance(entities, list) or len(entities) == 0:
        raise MetadataParserError(
            f"Rekord #{idx} w {source_name}: associated_entities jest puste lub nie jest listą"
        )

    entity = entities[0]
    missing_entity = [f for f in REQUIRED_ENTITY_FIELDS if f not in entity]
    if missing_entity:
        raise MetadataParserError(
            f"Rekord #{idx} w {source_name}: brakuje pól w associated_entities {missing_entity}"
        )

    aliquot_barcode = entity["entity_submitter_id"]
    if not ALIQUOT_BARCODE_PATTERN.match(aliquot_barcode):
        raise MetadataParserError(
            f"Rekord #{idx} w {source_name}: aliquot_barcode '{aliquot_barcode}' "
            f"nie pasuje do wzorca TCGA-XX-XXXX-NNA-NNA-XXXX-NN"
        )

    sample_id = _extract_sample_id(aliquot_barcode)
    case_submitter_id = _extract_case_submitter_id(aliquot_barcode)

    analysis = record["analysis"]
    missing_analysis = [f for f in REQUIRED_ANALYSIS_FIELDS if f not in analysis]
    if missing_analysis:
        raise MetadataParserError(
            f"Rekord #{idx} w {source_name}: brakuje pól w analysis {missing_analysis}"
        )

    input_files = analysis["input_files"]
    if not isinstance(input_files, list) or len(input_files) == 0:
        raise MetadataParserError(
            f"Rekord #{idx} w {source_name}: analysis.input_files jest puste lub nie jest listą"
        )

    input_file = input_files[0]
    missing_qc = [f for f in REQUIRED_INPUT_FILE_FIELDS if f not in input_file]
    if missing_qc:
        raise MetadataParserError(
            f"Rekord #{idx} w {source_name}: brakuje pól QC w analysis.input_files[0] {missing_qc}"
        )

    return {
        "file_id": record["file_id"],
        "file_name": record["file_name"],
        "md5sum": record["md5sum"],
        "file_size": int(record["file_size"]),
        "data_type": record["data_type"],
        "experimental_strategy": record["experimental_strategy"],
        "sample_id": sample_id,
        "case_submitter_id": case_submitter_id,
        "aliquot_barcode": aliquot_barcode,
        "case_uuid": entity["case_id"],
        "aliquot_uuid": entity["entity_id"],
        "workflow_type": analysis["workflow_type"],
        "workflow_version": analysis["workflow_version"],
        "total_reads": int(input_file["total_reads"]),
        "proportion_reads_mapped": float(input_file["proportion_reads_mapped"]),
        "average_base_quality": float(input_file["average_base_quality"]),
        "average_read_length": float(input_file["average_read_length"]),
    }


def _extract_sample_id(aliquot_barcode: str) -> str:
    """Wyciąga sample_id (TCGA-XX-XXXX-NNA) z aliquot barcode (7-segmentowego)."""
    return "-".join(aliquot_barcode.split("-")[:4])


def _extract_case_submitter_id(aliquot_barcode: str) -> str:
    """Wyciąga case_submitter_id (TCGA-XX-XXXX) z aliquot barcode."""
    return "-".join(aliquot_barcode.split("-")[:3])
