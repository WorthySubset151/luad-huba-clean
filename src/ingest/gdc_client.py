"""Klient GDC API do programowego pobierania metadanych i danych z portalu GDC.

Ten moduł odpowiada za komunikację z REST API portalu Genomic Data Commons
(api.gdc.cancer.gov) - budowanie zapytań filtrujących, wywołanie endpointów,
parsowanie odpowiedzi do polars DataFrame zgodnego z resztą pipeline'u oraz
pobieranie plików z weryfikacją sum kontrolnych MD5.
"""

__author__ = "Łukasz Połaski"

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import polars as pl
import requests
import urllib.error
import urllib.request
from tqdm import tqdm


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

DATA_ENDPOINT = f"{BASE_URL}/data"
DEFAULT_DOWNLOAD_TIMEOUT = 300
DEFAULT_MAX_RETRIES = 3
DEFAULT_CHUNK_SIZE = 1024 * 1024

DOWNLOAD_RESULT_COLUMNS: list[str] = [
    "file_id",
    "file_name",
    "local_path",
    "expected_md5",
    "actual_md5",
    "verified",
    "bytes_downloaded",
    "duration_s",
    "attempts",
    "error",
]


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


def download_files(
    metadata: pl.DataFrame,
    output_dir: Path,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_DOWNLOAD_TIMEOUT,
    show_progress: bool = True,
    skip_existing: bool = True,
) -> pl.DataFrame:
    """Pobiera pliki z GDC i weryfikuje sumy kontrolne MD5.

    Pobieranie odbywa się per-plik przez GET /data/<file_id>. Dla każdego pliku
    obliczana jest suma MD5 w trakcie pobierania (streaming, bez ładowania do
    pamięci) i porównywana z wartością z kolumny md5sum w DataFrame. Przy
    niezgodności lub błędzie sieci następuje retry z exponential backoff.

    Argumenty:
        metadata: DataFrame z kolumną file_id (wymagana), opcjonalnie md5sum
            do weryfikacji i file_name (jeśli nie ma, nazwa pochodzi z headera
            Content-Disposition GDC).
        output_dir: katalog docelowy. Utworzony jeśli nie istnieje.
        max_retries: maksymalna liczba prób per plik (domyślnie 3).
        timeout: timeout per zapytanie HTTP w sekundach (domyślnie 300).
        show_progress: czy pokazywać pasek postępu tqdm.
        skip_existing: jeśli True, pomija pliki które już istnieją lokalnie
            z poprawnym MD5 (idempotentność, można wznowić przerwane pobieranie).

    Zwraca:
        DataFrame z raportem pobierania, jeden wiersz per plik:
        file_id, file_name, local_path, expected_md5, actual_md5, verified
        (bool), bytes_downloaded, duration_s, attempts, error (pusty jeśli OK).

    Rzuca:
        GDCClientError: gdy DataFrame nie ma kolumny file_id.
    """
    if "file_id" not in metadata.columns:
        raise GDCClientError("DataFrame musi zawierać kolumnę file_id")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    has_md5 = "md5sum" in metadata.columns
    has_filename = "file_name" in metadata.columns

    results = []
    rows = metadata.iter_rows(named=True)
    if show_progress:
        rows = tqdm(rows, total=metadata.height, desc="Pobieranie z GDC", unit="plik")

    for row in rows:
        result = _download_one_file(
            file_id=row["file_id"],
            expected_md5=row.get("md5sum") if has_md5 else None,
            expected_name=row.get("file_name") if has_filename else None,
            output_dir=output_dir,
            max_retries=max_retries,
            timeout=timeout,
            skip_existing=skip_existing,
        )
        results.append(result)

    return pl.DataFrame(results).select(DOWNLOAD_RESULT_COLUMNS)


def _download_one_file(
    file_id: str,
    expected_md5: str | None,
    expected_name: str | None,
    output_dir: Path,
    max_retries: int,
    timeout: int,
    skip_existing: bool,
) -> dict[str, Any]:
    """Pobiera pojedynczy plik z retry i weryfikacją MD5."""
    if expected_name and skip_existing:
        candidate = output_dir / expected_name
        if candidate.exists() and expected_md5:
            existing_md5 = _compute_md5(candidate)
            if existing_md5 == expected_md5:
                return {
                    "file_id": file_id,
                    "file_name": expected_name,
                    "local_path": str(candidate),
                    "expected_md5": expected_md5,
                    "actual_md5": existing_md5,
                    "verified": True,
                    "bytes_downloaded": 0,
                    "duration_s": 0.0,
                    "attempts": 0,
                    "error": "",
                }

    url = f"{DATA_ENDPOINT}/{file_id}"
    last_error = ""

    for attempt in range(1, max_retries + 1):
        start = time.time()
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()

                filename = _resolve_filename(response, expected_name, file_id)
                local_path = output_dir / filename
                tmp_path = local_path.with_suffix(local_path.suffix + ".partial")

                hasher = hashlib.md5()
                bytes_downloaded = 0

                with tmp_path.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=DEFAULT_CHUNK_SIZE):
                        if chunk:
                            fh.write(chunk)
                            hasher.update(chunk)
                            bytes_downloaded += len(chunk)

                actual_md5 = hasher.hexdigest()
                duration = time.time() - start

                if expected_md5 and actual_md5 != expected_md5:
                    tmp_path.unlink()
                    last_error = f"MD5 mismatch: expected {expected_md5}, got {actual_md5}"
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
                        continue
                    return {
                        "file_id": file_id,
                        "file_name": filename,
                        "local_path": "",
                        "expected_md5": expected_md5,
                        "actual_md5": actual_md5,
                        "verified": False,
                        "bytes_downloaded": bytes_downloaded,
                        "duration_s": duration,
                        "attempts": attempt,
                        "error": last_error,
                    }

                tmp_path.rename(local_path)
                return {
                    "file_id": file_id,
                    "file_name": filename,
                    "local_path": str(local_path),
                    "expected_md5": expected_md5 or "",
                    "actual_md5": actual_md5,
                    "verified": (actual_md5 == expected_md5) if expected_md5 else True,
                    "bytes_downloaded": bytes_downloaded,
                    "duration_s": duration,
                    "attempts": attempt,
                    "error": "",
                }

        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue

        except requests.HTTPError as exc:
            last_error = f"HTTP {exc.response.status_code}: {exc.response.reason}"
            if exc.response.status_code >= 500 and attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            break

    return {
        "file_id": file_id,
        "file_name": expected_name or "",
        "local_path": "",
        "expected_md5": expected_md5 or "",
        "actual_md5": "",
        "verified": False,
        "bytes_downloaded": 0,
        "duration_s": 0.0,
        "attempts": max_retries,
        "error": last_error or "Nieznany błąd po max retries",
    }


def _resolve_filename(response: requests.Response, expected: str | None, file_id: str) -> str:
    """Wyciąga nazwę pliku z Content-Disposition lub używa fallbacku."""
    if expected:
        return expected

    cd = response.headers.get("Content-Disposition", "")
    match = re.search(r'filename="?([^";]+)"?', cd)
    if match:
        return match.group(1).strip()

    return f"{file_id}.bin"


def _compute_md5(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Oblicza MD5 pliku w trybie streaming (nie ładuje całości do pamięci)."""
    hasher = hashlib.md5()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()
