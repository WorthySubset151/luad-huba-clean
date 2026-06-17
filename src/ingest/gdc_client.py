"""Klient GDC API do programowego pobierania metadanych i danych z portalu GDC.

Ten moduł odpowiada za komunikację z REST API portalu Genomic Data Commons
(api.gdc.cancer.gov) - budowanie zapytań filtrujących, wywołanie endpointów,
parsowanie odpowiedzi do polars DataFrame zgodnego z resztą pipeline'u oraz
pobieranie plików z weryfikacją sum kontrolnych MD5.
"""

__author__ = "Łukasz Połaski"

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import polars as pl
from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
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


DEFAULT_MAX_CONCURRENCY = 15  # ile plików pobierać równolegle (semafor)

# Kody HTTP i wyjątki traktowane jako przejściowe (warte ponowienia).
_TRANSIENT_STATUS: set[int] = {429, 500, 502, 503, 504}
_TRANSIENT_HTTPX_EXC = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
    httpx.NetworkError,
)


class _RetryableMD5Error(Exception):
    """Niezgodność MD5 — pobranie uszkodzone, warto ponowić."""


def _is_transient(exc: BaseException) -> bool:
    """Czy błąd jest przejściowy (timeout, zerwane połączenie, 5xx/429, zła suma MD5).

    Błędy 4xx poza 429 NIE są przejściowe — ponawianie 404/400 nie ma sensu,
    tylko maskuje realny problem i marnuje czas.
    """
    if isinstance(exc, _RetryableMD5Error):
        return True
    if isinstance(exc, _TRANSIENT_HTTPX_EXC):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _TRANSIENT_STATUS
    return False


def _retry_wait():
    """Exponential backoff: ~2s, 4s, 8s… z górnym limitem 30s."""
    return wait_exponential(multiplier=1, min=2, max=30)


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
    project_id: str = DEFAULT_PROJECT_ID,
    max_retries: int = DEFAULT_MAX_RETRIES,
    _transport: "httpx.BaseTransport | None" = None,
) -> dict[str, Any]:
    """Wykonuje zapytanie do endpointu /files w GDC API (httpx + retry).

    Argumenty:
        filters: filtr w formacie GDC. Jeśli None — budowany z project_id.
        fields: lista pól do zwrócenia (jeśli None - bierze DEFAULT_FIELDS).
        size: liczba wyników w odpowiedzi (max MAX_PAGE_SIZE).
        page_from: offset paginacji (0-based).
        timeout: timeout w sekundach.
        project_id: projekt do filtra, gdy filters=None (domyślnie TCGA-LUAD).
            Pozwala odpytać dowolny projekt (BRCA, GBM, …) bez zmiany kodu.
        max_retries: liczba prób przy błędach przejściowych (5xx/timeout).

    Zwraca:
        Surowy JSON z odpowiedzi GDC: {"data": {"hits": [...], "pagination": {...}}}.

    Rzuca:
        GDCClientError: błąd HTTP (po wyczerpaniu prób), połączenia lub JSON.
    """
    if filters is None:
        filters = build_files_filter(project_id=project_id)
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

    def _post() -> httpx.Response:
        with httpx.Client(transport=_transport, timeout=timeout) as client:
            response = client.post(FILES_ENDPOINT, json=payload)
        response.raise_for_status()
        return response

    retryer = Retrying(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(max_retries),
        wait=_retry_wait(),
        reraise=True,
    )

    try:
        response = retryer(_post)
    except httpx.HTTPStatusError as exc:
        raise GDCClientError(
            f"GDC API zwróciło HTTP {exc.response.status_code}: {exc.response.reason_phrase}"
        ) from exc
    except httpx.HTTPError as exc:
        raise GDCClientError(f"Nie można połączyć się z {FILES_ENDPOINT}: {exc}") from exc

    try:
        return response.json()
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
    progress_callback: "Callable[[int, int, str], None] | None" = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    _transport: "httpx.AsyncBaseTransport | None" = None,
) -> pl.DataFrame:
    """Pobiera pliki z GDC współbieżnie (httpx.AsyncClient) i weryfikuje MD5.

    Pliki pobierane są równolegle — do max_concurrency naraz (semafor), co
    skraca czas z godzin do minut przy setkach plików. Każde pobranie jest
    streamowane (MD5 liczone w locie, bez ładowania całości do pamięci), a przy
    błędach przejściowych (timeout, zerwane połączenie, 5xx/429, zła suma MD5)
    ponawiane z exponential backoff (tenacity). Błędy 4xx nie są ponawiane.

    Argumenty:
        metadata: DataFrame z kolumną file_id (wymagana), opcjonalnie md5sum
            (weryfikacja) i file_name (inaczej z nagłówka Content-Disposition).
        output_dir: katalog docelowy. Utworzony jeśli nie istnieje.
        max_retries: maksymalna liczba prób per plik (domyślnie 3).
        timeout: timeout per zapytanie HTTP w sekundach (domyślnie 300).
        show_progress: czy pokazywać pasek postępu tqdm.
        skip_existing: jeśli True, pomija pliki które już istnieją lokalnie
            z poprawnym MD5 (idempotentność, można wznowić przerwane pobieranie).
        progress_callback: opcjonalna funkcja wywoływana po każdym ukończonym
            pliku z argumentami (ukończone, wszystkie, nazwa). Kolejność jest
            niedeterministyczna przy współbieżności. Używana przez GUI/terminal.
        max_concurrency: ile plików pobierać równolegle (domyślnie 15).
        _transport: wewnętrzne — transport httpx do wstrzyknięcia w testach.

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

    bar = (
        tqdm(total=metadata.height, desc="Pobieranie z GDC", unit="plik")
        if show_progress else None
    )

    def _on_done(done: int, total: int, name: str) -> None:
        if bar is not None:
            bar.update(1)
        if progress_callback is not None:
            progress_callback(done, total, name)

    try:
        results = asyncio.run(
            _download_all_async(
                metadata, output_dir, max_retries, timeout,
                max_concurrency, skip_existing, _on_done, _transport,
            )
        )
    finally:
        if bar is not None:
            bar.close()

    return pl.DataFrame(results).select(DOWNLOAD_RESULT_COLUMNS)


async def _download_all_async(
    metadata, output_dir, max_retries, timeout,
    max_concurrency, skip_existing, on_done, transport,
):
    """Orkiestracja współbieżna: semafor + asyncio.gather, kolejność wyników zachowana."""
    rows = list(metadata.iter_rows(named=True))
    has_md5 = "md5sum" in metadata.columns
    has_name = "file_name" in metadata.columns
    total = len(rows)
    results: list = [None] * total
    done = {"n": 0}
    sem = asyncio.Semaphore(max_concurrency)
    limits = httpx.Limits(
        max_connections=max_concurrency,
        max_keepalive_connections=max_concurrency,
    )

    async with httpx.AsyncClient(
        follow_redirects=True, limits=limits, transport=transport,
    ) as client:

        async def worker(idx: int, row: dict) -> None:
            async with sem:
                res = await _download_one_async(
                    client=client,
                    file_id=row["file_id"],
                    expected_md5=row.get("md5sum") if has_md5 else None,
                    expected_name=row.get("file_name") if has_name else None,
                    output_dir=output_dir,
                    max_retries=max_retries,
                    timeout=timeout,
                    skip_existing=skip_existing,
                )
            results[idx] = res
            done["n"] += 1
            on_done(done["n"], total, res.get("file_name", ""))

        await asyncio.gather(*(worker(i, r) for i, r in enumerate(rows)))

    return results


async def _download_one_async(
    client: httpx.AsyncClient,
    file_id: str,
    expected_md5: str | None,
    expected_name: str | None,
    output_dir: Path,
    max_retries: int,
    timeout: int,
    skip_existing: bool,
) -> dict[str, Any]:
    """Pobiera pojedynczy plik (stream + MD5) z retry na błędach przejściowych."""
    existing = _check_existing(file_id, expected_md5, expected_name, output_dir, skip_existing)
    if existing is not None:
        return existing

    url = f"{DATA_ENDPOINT}/{file_id}"
    start = time.time()
    attempts = {"n": 0}

    async def _attempt() -> dict[str, Any]:
        attempts["n"] += 1
        async with client.stream("GET", url, timeout=timeout) as response:
            response.raise_for_status()
            filename = _resolve_filename(response, expected_name, file_id)
            local_path = output_dir / filename
            tmp_path = local_path.with_suffix(local_path.suffix + ".partial")
            hasher = hashlib.md5()
            bytes_downloaded = 0
            with tmp_path.open("wb") as fh:
                async for chunk in response.aiter_bytes(DEFAULT_CHUNK_SIZE):
                    fh.write(chunk)
                    hasher.update(chunk)
                    bytes_downloaded += len(chunk)
            actual_md5 = hasher.hexdigest()
            if expected_md5 and actual_md5 != expected_md5:
                tmp_path.unlink(missing_ok=True)
                raise _RetryableMD5Error(
                    f"MD5 mismatch: expected {expected_md5}, got {actual_md5}"
                )
            tmp_path.rename(local_path)
            return {
                "file_name": filename,
                "local_path": str(local_path),
                "actual_md5": actual_md5,
                "bytes_downloaded": bytes_downloaded,
            }

    retryer = AsyncRetrying(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(max_retries),
        wait=_retry_wait(),
        reraise=True,
    )

    try:
        ok = await retryer(_attempt)
    except Exception as exc:  # noqa: BLE001
        return {
            "file_id": file_id,
            "file_name": expected_name or "",
            "local_path": "",
            "expected_md5": expected_md5 or "",
            "actual_md5": "",
            "verified": False,
            "bytes_downloaded": 0,
            "duration_s": round(time.time() - start, 3),
            "attempts": attempts["n"],
            "error": _format_error(exc),
        }

    return {
        "file_id": file_id,
        "file_name": ok["file_name"],
        "local_path": ok["local_path"],
        "expected_md5": expected_md5 or "",
        "actual_md5": ok["actual_md5"],
        "verified": (ok["actual_md5"] == expected_md5) if expected_md5 else True,
        "bytes_downloaded": ok["bytes_downloaded"],
        "duration_s": round(time.time() - start, 3),
        "attempts": attempts["n"],
        "error": "",
    }


def _check_existing(file_id, expected_md5, expected_name, output_dir, skip_existing):
    """Gotowy wynik, jeśli plik jest już lokalnie z poprawnym MD5 (skip_existing)."""
    if not (expected_name and skip_existing and expected_md5):
        return None
    candidate = output_dir / expected_name
    if candidate.exists() and _compute_md5(candidate) == expected_md5:
        return {
            "file_id": file_id,
            "file_name": expected_name,
            "local_path": str(candidate),
            "expected_md5": expected_md5,
            "actual_md5": expected_md5,
            "verified": True,
            "bytes_downloaded": 0,
            "duration_s": 0.0,
            "attempts": 0,
            "error": "",
        }
    return None


def _format_error(exc: BaseException) -> str:
    """Czytelny opis błędu do kolumny error w raporcie."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}: {exc.response.reason_phrase}"
    if isinstance(exc, _RetryableMD5Error):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def _resolve_filename(response: httpx.Response, expected: str | None, file_id: str) -> str:
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
