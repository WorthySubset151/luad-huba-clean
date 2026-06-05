"""Klient GDC API do programowego pobierania metadanych i danych z portalu GDC.

Ten moduł odpowiada za komunikację z REST API portalu Genomic Data Commons
(api.gdc.cancer.gov) - budowanie zapytań filtrujących, wywołanie endpointów
i parsowanie odpowiedzi do polars DataFrame zgodnego z resztą pipeline'u.

Część pierwsza (ten plik): zapytania i parsowanie odpowiedzi.
Część druga (osobny commit): pobieranie plików z weryfikacją MD5.
"""

__author__ = "Łukasz Połaski"

import json
from typing import Any

import polars as pl
import urllib.request
import urllib.error


BASE_URL = "https://api.gdc.cancer.gov"
FILES_ENDPOINT = f"{BASE_URL}/files"

DEFAULT_PROJECT_ID = "TCGA-LUAD"
DEFAULT_WORKFLOW_TYPE = "STAR - Counts"
DEFAULT_DATA_TYPE = "Gene Expression Quantification"

DEFAULT_FIELDS: list[str] = [
    "file_id",
    "file_name",
    "md5sum",
    "file_size",
    "data_type",
    "experimental_strategy",
    "cases.case_id",
    "cases.submitter_id",
    "cases.samples.sample_id",
    "cases.samples.submitter_id",
    "cases.samples.portions.analytes.aliquots.aliquot_id",
    "cases.samples.portions.analytes.aliquots.submitter_id",
    "analysis.workflow_type",
    "analysis.workflow_version",
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
]

DEFAULT_PAGE_SIZE = 1000
MAX_PAGE_SIZE = 10000
DEFAULT_TIMEOUT_SECONDS = 60


class GDCClientError(Exception):
    """Błąd komunikacji z GDC API lub parsowania odpowiedzi."""


def build_files_filter(
    project_id: str = DEFAULT_PROJECT_ID,
    workflow_type: str = DEFAULT_WORKFLOW_TYPE,
    data_type: str = DEFAULT_DATA_TYPE,
) -> dict[str, Any]:
    """Buduje filtr GDC dla endpointu /files w formacie JSON.

    Filtr GDC ma składnię {"op": "and", "content": [warunek1, warunek2, ...]}
    gdzie każdy warunek ma postać {"op": "in", "content": {"field": ..., "value": [...]}}.

    Argumenty:
        project_id: identyfikator projektu (np. "TCGA-LUAD").
        workflow_type: typ workflow GDC (np. "STAR - Counts").
        data_type: typ danych (np. "Gene Expression Quantification").

    Zwraca:
        Słownik z filtrem gotowym do serializacji JSON i przekazania do API.
    """
    return {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": [project_id],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "analysis.workflow_type",
                    "value": [workflow_type],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "data_type",
                    "value": [data_type],
                },
            },
        ],
    }


def query_files(
    filters: dict[str, Any] | None = None,
    fields: list[str] | None = None,
    size: int = DEFAULT_PAGE_SIZE,
    page_from: int = 0,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Wykonuje zapytanie do endpointu /files w GDC API.

    Argumenty:
        filters: filtr w formacie GDC (jeśli None - bierze build_files_filter()).
        fields: lista pól do zwrócenia (jeśli None - bierze DEFAULT_FIELDS).
        size: liczba wyników w odpowiedzi (max MAX_PAGE_SIZE).
        page_from: offset paginacji (0-based).
        timeout: timeout w sekundach.

    Zwraca:
        Surowy JSON z odpowiedzi GDC: {"data": {"hits": [...], "pagination": {...}}}.

    Rzuca:
        GDCClientError: timeout, błąd HTTP, niepoprawny JSON.
    """
    if filters is None:
        filters = build_files_filter()
    if fields is None:
        fields = DEFAULT_FIELDS

    if size < 1 or size > MAX_PAGE_SIZE:
        raise GDCClientError(
            f"Parametr size musi być w zakresie 1..{MAX_PAGE_SIZE}, otrzymano: {size}"
        )

    payload = {
        "filters": filters,
        "fields": ",".join(fields),
        "size": size,
        "from": page_from,
        "format": "JSON",
    }

    request_body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        FILES_ENDPOINT,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_bytes = response.read()
    except urllib.error.HTTPError as exc:
        raise GDCClientError(
            f"GDC API zwróciło HTTP {exc.code}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise GDCClientError(f"Nie można połączyć się z {FILES_ENDPOINT}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise GDCClientError(
            f"Timeout po {timeout}s przy zapytaniu do {FILES_ENDPOINT}"
        ) from exc

    try:
        return json.loads(response_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise GDCClientError(f"Niepoprawny JSON w odpowiedzi GDC API: {exc}") from exc


def parse_files_response(response: dict[str, Any]) -> pl.DataFrame:
    """Parsuje odpowiedź GDC API z endpointu /files do polars DataFrame.

    Odpowiedź GDC jest zagnieżdżona w strukturze:
        data.hits[].cases[].samples[].portions[].analytes[].aliquots[]
    Funkcja spłaszcza to do jednego wiersza per plik, zgodnie z OUTPUT_COLUMNS.

    Argumenty:
        response: surowy JSON z GDC API (output query_files).

    Zwraca:
        DataFrame z 13 kolumnami: file_id, file_name, md5sum, file_size,
        data_type, experimental_strategy, sample_id, case_submitter_id,
        aliquot_barcode, case_uuid, aliquot_uuid, workflow_type, workflow_version.

    Rzuca:
        GDCClientError: gdy struktura odpowiedzi jest niezgodna z oczekiwaną.
    """
    if not isinstance(response, dict) or "data" not in response:
        raise GDCClientError("Odpowiedź GDC nie zawiera klucza 'data'")

    data = response["data"]
    if "hits" not in data:
        raise GDCClientError("Odpowiedź GDC nie zawiera klucza 'data.hits'")

    hits = data["hits"]
    if not isinstance(hits, list):
        raise GDCClientError(
            f"data.hits powinno być listą, otrzymano: {type(hits).__name__}"
        )

    if len(hits) == 0:
        return pl.DataFrame(schema={col: pl.Utf8 for col in OUTPUT_COLUMNS})

    rows = [_parse_hit(hit, idx) for idx, hit in enumerate(hits)]
    df = pl.DataFrame(rows)

    if df["file_id"].n_unique() != df.height:
        raise GDCClientError("Duplikaty file_id w odpowiedzi GDC")

    return df.select(OUTPUT_COLUMNS).sort("file_id")


def _parse_hit(hit: dict[str, Any], idx: int) -> dict[str, Any]:
    """Wyciąga płaski rekord z jednego elementu data.hits."""
    required_top = ["file_id", "file_name", "md5sum", "file_size"]
    missing = [f for f in required_top if f not in hit]
    if missing:
        raise GDCClientError(
            f"Hit #{idx}: brakuje pól {missing}"
        )

    cases = hit.get("cases", [])
    if not isinstance(cases, list) or len(cases) == 0:
        raise GDCClientError(f"Hit #{idx} (file_id={hit['file_id']}): puste cases")

    case = cases[0]
    case_uuid = case.get("case_id", "")
    case_submitter_id = case.get("submitter_id", "")

    samples = case.get("samples", [])
    if not isinstance(samples, list) or len(samples) == 0:
        raise GDCClientError(
            f"Hit #{idx} (file_id={hit['file_id']}): puste cases[0].samples"
        )

    sample = samples[0]
    sample_id = sample.get("submitter_id", "")

    aliquot_uuid, aliquot_barcode = _extract_aliquot_info(sample, idx, hit["file_id"])

    analysis = hit.get("analysis", {})

    return {
        "file_id": hit["file_id"],
        "file_name": hit["file_name"],
        "md5sum": hit["md5sum"],
        "file_size": int(hit["file_size"]),
        "data_type": hit.get("data_type", ""),
        "experimental_strategy": hit.get("experimental_strategy", ""),
        "sample_id": sample_id,
        "case_submitter_id": case_submitter_id,
        "aliquot_barcode": aliquot_barcode,
        "case_uuid": case_uuid,
        "aliquot_uuid": aliquot_uuid,
        "workflow_type": analysis.get("workflow_type", ""),
        "workflow_version": analysis.get("workflow_version", ""),
    }


def _extract_aliquot_info(
    sample: dict[str, Any], idx: int, file_id: str
) -> tuple[str, str]:
    """Wyciąga aliquot UUID i barcode z głęboko zagnieżdżonej struktury sample."""
    portions = sample.get("portions", [])
    if not isinstance(portions, list) or len(portions) == 0:
        raise GDCClientError(
            f"Hit #{idx} (file_id={file_id}): puste portions"
        )

    analytes = portions[0].get("analytes", [])
    if not isinstance(analytes, list) or len(analytes) == 0:
        raise GDCClientError(
            f"Hit #{idx} (file_id={file_id}): puste analytes"
        )

    aliquots = analytes[0].get("aliquots", [])
    if not isinstance(aliquots, list) or len(aliquots) == 0:
        raise GDCClientError(
            f"Hit #{idx} (file_id={file_id}): puste aliquots"
        )

    aliquot = aliquots[0]
    return aliquot.get("aliquot_id", ""), aliquot.get("submitter_id", "")
