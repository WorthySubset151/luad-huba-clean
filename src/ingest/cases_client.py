"""Klient GDC API dla endpointu /cases - dane kliniczne pacjentów.

Ten moduł odpowiada za pobieranie danych klinicznych z portalu Genomic Data
Commons. W odróżnieniu od /files (gdc_client.py), endpoint /cases zwraca dane
zorientowane na pacjenta (vital_status, days_to_death, staging, demografia)
i jest separatnym modelem danych w GDC.

Output parse_cases_response jest formatem 1:1 zgodnym z clinical.tsv jak go
zwraca UI portalu - można go zapisać jako TSV i parse_clinical z modułu
ingest zadziała bez żadnej zmiany.
"""

__author__ = "Łukasz Połaski"

import json
from typing import Any

import polars as pl
import urllib.error
import urllib.request

from src.ingest.gdc_client import (
    BASE_URL,
    DEFAULT_PAGE_SIZE,
    DEFAULT_PROJECT_ID,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_PAGE_SIZE,
    GDCClientError,
)


CASES_ENDPOINT = f"{BASE_URL}/cases"

DEFAULT_FIELDS: list[str] = [
    "submitter_id",
    "primary_site",
    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.age_at_index",
    "demographic.gender",
    "demographic.race",
    "diagnoses.diagnosis_is_primary_disease",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.ajcc_pathologic_stage",
]

OUTPUT_COLUMNS: list[str] = [
    "cases.submitter_id",
    "cases.primary_site",
    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.age_at_index",
    "demographic.gender",
    "demographic.race",
    "diagnoses.diagnosis_is_primary_disease",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.ajcc_pathologic_stage",
]


class CasesClientError(GDCClientError):
    """Błąd komunikacji z endpointem /cases lub parsowania odpowiedzi."""


def build_cases_filter(project_id: str = DEFAULT_PROJECT_ID) -> dict[str, Any]:
    """Buduje filtr GDC dla endpointu /cases.

    Filtr /cases jest prostszy niż /files - pacjenci są encją podstawową,
    nie ma potrzeby filtrowania po workflow_type czy data_type (te dotyczą
    plików). Wystarczy filtrowanie po project_id.

    Argumenty:
        project_id: identyfikator projektu (np. "TCGA-LUAD").

    Zwraca:
        Słownik z filtrem gotowym do serializacji JSON.
    """
    return {
        "op": "in",
        "content": {
            "field": "project.project_id",
            "value": [project_id],
        },
    }


def query_cases(
    filters: dict[str, Any] | None = None,
    fields: list[str] | None = None,
    size: int = DEFAULT_PAGE_SIZE,
    page_from: int = 0,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Wykonuje zapytanie do endpointu /cases w GDC API.

    Argumenty:
        filters: filtr w formacie GDC (jeśli None - build_cases_filter()).
        fields: lista pól do zwrócenia (jeśli None - DEFAULT_FIELDS).
        size: liczba wyników (max MAX_PAGE_SIZE).
        page_from: offset paginacji.
        timeout: timeout w sekundach.

    Zwraca:
        Surowy JSON z odpowiedzi GDC: {"data": {"hits": [...], "pagination": {...}}}.

    Rzuca:
        CasesClientError: timeout, błąd HTTP, niepoprawny JSON.
    """
    if filters is None:
        filters = build_cases_filter()
    if fields is None:
        fields = DEFAULT_FIELDS

    if size < 1 or size > MAX_PAGE_SIZE:
        raise CasesClientError(
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
        CASES_ENDPOINT,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_bytes = response.read()
    except urllib.error.HTTPError as exc:
        raise CasesClientError(
            f"GDC API zwróciło HTTP {exc.code}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise CasesClientError(
            f"Nie można połączyć się z {CASES_ENDPOINT}: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise CasesClientError(
            f"Timeout po {timeout}s przy zapytaniu do {CASES_ENDPOINT}"
        ) from exc

    try:
        return json.loads(response_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise CasesClientError(f"Niepoprawny JSON w odpowiedzi GDC API: {exc}") from exc


def parse_cases_response(response: dict[str, Any]) -> pl.DataFrame:
    """Parsuje odpowiedź /cases do DataFrame w formacie clinical.tsv z portalu.

    GDC zwraca strukturę zagnieżdżoną:
        data.hits[].{submitter_id, demographic{}, diagnoses[]}

    Funkcja rozwija ją do formatu **flat** identycznego z eksportem TSV z UI
    portalu - kolumny mają nazwy "cases.X", "demographic.X", "diagnoses.X".
    Pacjent z N diagnozami daje N wierszy (cross-product diagnozy x pacjent),
    co jest standardowym zachowaniem GDC clinical.tsv. parse_clinical z modułu
    ingest deduplikuje to filtrem diagnoses.diagnosis_is_primary_disease == "true".

    Argumenty:
        response: surowy JSON z GDC API (output query_cases).

    Zwraca:
        DataFrame z 10 kolumnami (OUTPUT_COLUMNS). Liczba wierszy >= liczba
        pacjentów w odpowiedzi (zwykle 1:1, ale możliwe więcej dla pacjentów
        z wieloma diagnozami w bazie).

    Rzuca:
        CasesClientError: gdy struktura odpowiedzi jest niezgodna z oczekiwaną.
    """
    if not isinstance(response, dict) or "data" not in response:
        raise CasesClientError("Odpowiedź GDC nie zawiera klucza 'data'")

    data = response["data"]
    if "hits" not in data:
        raise CasesClientError("Odpowiedź GDC nie zawiera klucza 'data.hits'")

    hits = data["hits"]
    if not isinstance(hits, list):
        raise CasesClientError(
            f"data.hits powinno być listą, otrzymano: {type(hits).__name__}"
        )

    if len(hits) == 0:
        return pl.DataFrame(schema={col: pl.Utf8 for col in OUTPUT_COLUMNS})

    rows = []
    for idx, hit in enumerate(hits):
        rows.extend(_parse_case(hit, idx))

    df = pl.DataFrame(rows, infer_schema_length=len(rows))
    return df.select(OUTPUT_COLUMNS).sort(["cases.submitter_id"])


def _parse_case(hit: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    """Rozwija jednego pacjenta w listę wierszy (jeden per diagnoza).

    Jeśli pacjent nie ma diagnoz, zwraca jeden wiersz z pustymi polami
    diagnoses.X (parse_clinical odfiltruje go potem bo brak diagnozy
    primary_disease).
    """
    submitter_id = hit.get("submitter_id", "")
    if not submitter_id:
        raise CasesClientError(f"Hit #{idx}: brak submitter_id")

    primary_site = hit.get("primary_site", "")

    demographic = hit.get("demographic") or {}
    demographic_fields = {
        "demographic.vital_status": demographic.get("vital_status", ""),
        "demographic.days_to_death": _stringify(demographic.get("days_to_death")),
        "demographic.age_at_index": _stringify(demographic.get("age_at_index")),
        "demographic.gender": demographic.get("gender", ""),
        "demographic.race": demographic.get("race", ""),
    }

    diagnoses = hit.get("diagnoses") or []

    base_row = {
        "cases.submitter_id": submitter_id,
        "cases.primary_site": primary_site,
        **demographic_fields,
    }

    if not diagnoses:
        return [
            {
                **base_row,
                "diagnoses.diagnosis_is_primary_disease": "",
                "diagnoses.days_to_last_follow_up": "",
                "diagnoses.ajcc_pathologic_stage": "",
            }
        ]

    return [
        {
            **base_row,
            "diagnoses.diagnosis_is_primary_disease": _stringify_bool(
                diagnosis.get("diagnosis_is_primary_disease")
            ),
            "diagnoses.days_to_last_follow_up": _stringify(
                diagnosis.get("days_to_last_follow_up")
            ),
            "diagnoses.ajcc_pathologic_stage": diagnosis.get(
                "ajcc_pathologic_stage", ""
            ) or "",
        }
        for diagnosis in diagnoses
    ]


def _stringify(value: Any) -> str:
    """Konwersja wartości do stringa zgodnego z formatem TSV z portalu GDC.

    GDC w eksporcie TSV reprezentuje brak danych jako pusty string,
    nie jako "null" czy "None" - parser musi to odzwierciedlać.
    """
    if value is None:
        return ""
    return str(value)


def _stringify_bool(value: Any) -> str:
    """Konwertuje bool/None do formy oczekiwanej przez clinical_parser.

    GDC clinical.tsv używa stringów "true"/"false" (lowercase), nie Python-owych
    "True"/"False". Filtr w clinical_parser opiera się na "true".
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).lower()
