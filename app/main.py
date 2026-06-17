"""LUAD-HUBA — interfejs graficzny pipeline'u (Streamlit).

Jeden plik, ręczny sidebar, wymuszanie kolejności etapów pipeline'u.
Uruchomienie: uv run streamlit run app/main.py
"""

__author__ = "Łukasz Połaski"

import sys
from pathlib import Path
from typing import Callable

import polars as pl
import streamlit as st
import yaml

# --- Ścieżki projektu (app/main.py -> root to parent.parent) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Funkcje pipeline'u z src/ (wołane bezpośrednio) ---
from src.ingest.file_naming import STAR_FILE_PATTERNS, STAR_FILE_SUFFIXES
from src.ingest.star_parser import StarParserError, parse_star_counts
from src.ingest.sample_sheet_parser import parse_sample_sheet
from src.ingest.clinical_parser import parse_clinical
from src.transform.expression_matrix import build_expression_matrix
from src.transform.survival_dataset import build_survival_dataset
from src.cli_config import load_config, get_nested, resolve_metric, ConfigError
from src.validate.runner import run_cohort_qc, discover_stems, save_qc_report
from src.validate.qc_result import Severity, QCCategory
from src.validate.report_view import classify_qc
from src.ingest.gdc_client import (
    build_files_filter, query_files, parse_files_response, download_files, GDCClientError,
)
from src.ingest.cases_client import query_cases, parse_cases_response, CasesClientError

# Moduł wizualizacji dashboardu (Plotly)
import app.dashboard_viz as viz
# Warstwa analityczna (jedno źródło liczb — spójność GUI z terminalem)
from src.analysis import survival_report as sr
from src.analysis.expression_report import expression_summary
from src.analysis import readiness_report as readiness
from src.analysis import ml_export

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim" / "star_counts"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_UPLOADED_STAR = DATA_RAW / "uploaded_star"
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


@st.cache_data(show_spinner=False)
def _expression_summary_cached(path_str: str, mtime: float) -> dict:
    """Cache podsumowania macierzy (PCA jest kosztowna) — klucz: ścieżka + mtime."""
    return expression_summary(path_str)

# --- Definicja etapów pipeline'u (kolejność + zależności) ---
# klucz: id etapu, label: nazwa w sidebarze, requires: id etapu wymaganego wcześniej
STAGES = [
    {"id": "download", "label": "Pobieranie", "requires": None, "always": True},
    {"id": "upload", "label": "Wgrywanie", "requires": None, "always": True},
    {"id": "browse", "label": "Przeglądanie", "requires": None, "always": True},
    {"id": "parse", "label": "Parsowanie", "requires": "raw_ready", "always": False},
    {"id": "validate", "label": "Walidacja", "requires": "parsed", "always": False},
    {"id": "build_matrix", "label": "Macierz ekspresji", "requires": "parsed", "always": False},
    {"id": "build_survival", "label": "Zbiór przeżywalności", "requires": "matrix_built", "always": False},
    {"id": "dashboard", "label": "Dashboard analityczny", "requires": "survival_built", "always": False},
    {"id": "manage", "label": "Zarządzanie danymi", "requires": None, "always": True},
    {"id": "config", "label": "Konfiguracja", "requires": None, "always": True},
]


# =====================================================================
#  WYKRYWANIE STANU Z DYSKU
# =====================================================================
def detect_state() -> dict:
    """Sprawdza co już jest na dysku i zwraca flagi gotowości etapów.

    Stan jest wykrywany z plików (nie tylko z pamięci sesji), więc po
    restarcie GUI poprawnie rozpoznaje co zostało już zrobione.
    """
    # Surowe wejście: clinical + sample sheet w data/raw
    has_clinical = (DATA_RAW / "clinical.tsv").exists()
    sample_sheets = list(DATA_RAW.glob("gdc_sample_sheet*.tsv")) if DATA_RAW.exists() else []
    has_sample_sheet = len(sample_sheets) > 0

    # Sparsowane pliki STAR (parquety w interim)
    parquets = list(DATA_INTERIM.glob("*.parquet")) if DATA_INTERIM.exists() else []

    matrix_exists = (DATA_PROCESSED / "expression_matrix.parquet").exists()
    survival_exists = (DATA_PROCESSED / "survival_dataset.parquet").exists()

    return {
        "raw_ready": has_clinical and has_sample_sheet,
        "has_clinical": has_clinical,
        "has_sample_sheet": has_sample_sheet,
        "parsed": len(parquets) > 0,
        "parsed_count": len(parquets),
        "matrix_built": matrix_exists,
        "survival_built": survival_exists,
    }


def stage_unlocked(stage: dict, state: dict) -> bool:
    """Czy etap jest odblokowany (always-on lub spełniona zależność)."""
    if stage["always"]:
        return True
    if stage["requires"] is None:
        return True
    return state.get(stage["requires"], False)


# =====================================================================
#  SEKCJE — RENDER
# =====================================================================
def _preview_full(df, csv_filename: str) -> None:
    """Przewijalny podgląd CAŁEJ ramki (nie tylko kilku wierszy) + eksport CSV."""
    st.write(f"Wymiary: **{df.height:,} wierszy × {df.width:,} kolumn**".replace(",", " "))
    max_show = 20000
    if df.height > max_show:
        st.caption(
            f"Tabela pokazuje pierwsze {max_show:,} z {df.height:,} wierszy "
            "(cały plik pobierz poniżej).".replace(",", " ")
        )
        shown = df.head(max_show)
    else:
        shown = df
    st.dataframe(shown.to_pandas(), use_container_width=True, height=560)
    if df.height * df.width <= 5_000_000:
        st.download_button(
            "⬇ Pobierz cały plik jako CSV",
            df.write_csv().encode("utf-8"),
            file_name=csv_filename,
            mime="text/csv",
        )
    else:
        st.caption("Plik za duży na eksport CSV w przeglądarce — pobierz plik źródłowy z dysku.")


def render_browse(state: dict) -> None:
    """Podgląd plików w data/ (raw, interim, processed)."""
    st.header("Przeglądanie danych")
    st.caption("Przeglądaj pliki pipeline'u: surowe dane, parquety, wyniki.")

    dir_choice = st.selectbox(
        "Katalog",
        options=[
            ("data/raw", DATA_RAW),
            ("data/raw/uploaded_star (pliki STAR)", DATA_UPLOADED_STAR),
            ("data/interim/star_counts", DATA_INTERIM),
            ("data/processed", DATA_PROCESSED),
        ],
        format_func=lambda x: x[0],
    )
    _, chosen_dir = dir_choice

    if not chosen_dir.exists():
        st.warning(f"Katalog `{chosen_dir.relative_to(PROJECT_ROOT)}` nie istnieje jeszcze.")
        return

    files = sorted(
        [f for f in chosen_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
    )
    if not files:
        st.info("Katalog jest pusty (brak plików do podglądu).")
        return

    # Licznik na górze - od razu widać ile plików (istotne dla 601 STAR)
    st.metric("Plików w katalogu", len(files))

    file_choice = st.selectbox(
        "Plik do podglądu",
        options=files,
        format_func=lambda f: f"{f.name}  ({f.stat().st_size / 1024:.1f} KB)",
    )

    suffix = file_choice.suffix.lower()
    if suffix == ".parquet":
        try:
            df = pl.read_parquet(file_choice)
            _preview_full(df, file_choice.stem + ".csv")
            with st.expander("Nazwy kolumn"):
                st.write(list(df.columns))
        except Exception as exc:
            st.error(f"Nie udało się wczytać parquet: {exc}")
    elif suffix in (".tsv", ".csv", ".txt"):
        sep = "\t" if suffix == ".tsv" else ","
        try:
            # comment_prefix="#" pomija linie komentarza (np. nagłówek GENCODE
            # w surowych plikach STAR-Counts: "# gene-model: GENCODE v36"),
            # które mają inną liczbę pól niż dane. Nieszkodliwe dla plików bez
            # komentarzy (zwykły sample sheet, clinical).
            df = pl.read_csv(file_choice, separator=sep, infer_schema_length=1000,
                             comment_prefix="#")
            _preview_full(df, file_choice.stem + ".csv")
        except Exception as exc:
            st.error(f"Nie udało się wczytać pliku tabelarycznego: {exc}")
    elif suffix in (".json", ".yaml", ".yml"):
        text = file_choice.read_text(encoding="utf-8")
        st.code(text, language="yaml" if suffix in (".yaml", ".yml") else "json")
    else:
        st.info(f"Podgląd niedostępny dla rozszerzenia `{suffix}`.")


def render_config(state: dict) -> None:
    """Edycja configs/default.yaml przez formularz."""
    st.header("Konfiguracja pipeline'u")
    st.caption(f"Edycja `{CONFIG_PATH.relative_to(PROJECT_ROOT)}`")

    if not CONFIG_PATH.exists():
        st.error(f"Brak pliku konfiguracyjnego: {CONFIG_PATH}")
        return

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    st.subheader("Normalizacja")
    norm = cfg.get("normalization", {})
    metric_options = ["tpm", "unstranded", "fpkm", "fpkm_uq", "counts"]
    current_method = norm.get("method", "tpm")
    method = st.selectbox(
        "Metryka (normalization.method)",
        options=metric_options,
        index=metric_options.index(current_method) if current_method in metric_options else 0,
        help="Aliasy akceptowane przez pipeline. tpm/fpkm znormalizowane, counts/unstranded surowe.",
    )
    biotype = st.text_input(
        "Filtr biotype (normalization.biotype_filter)",
        value=norm.get("biotype_filter", "protein_coding") or "",
        help="np. protein_coding. Puste = bez filtra (wszystkie 60660 genów).",
    )
    min_samples = st.number_input(
        "Min próbek z ekspresją (normalization.min_samples_expressed)",
        min_value=0, value=int(norm.get("min_samples_expressed", 10)),
    )

    st.subheader("Przeżywalność")
    surv = cfg.get("survival", {})
    min_follow = st.number_input(
        "Min follow-up w dniach (survival.min_follow_up_days)",
        min_value=0, value=int(surv.get("min_follow_up_days", 30)),
        help="Usuwa krótkie CENZURY. Wczesne zgony są zachowane.",
    )
    drop_zero = st.checkbox(
        "Usuń artefakty time<=0 (survival.drop_zero_time)",
        value=bool(surv.get("drop_zero_time", True)),
        help="Usuwa próbki z time<=0 (artefakt PHI - zaokrąglenie daty diagnozy).",
    )

    st.divider()
    if st.button("Zapisz konfigurację", type="primary"):
        cfg.setdefault("normalization", {})
        cfg["normalization"]["method"] = method
        cfg["normalization"]["biotype_filter"] = biotype if biotype.strip() else None
        cfg["normalization"]["min_samples_expressed"] = int(min_samples)
        cfg.setdefault("survival", {})
        cfg["survival"]["min_follow_up_days"] = int(min_follow)
        cfg["survival"]["drop_zero_time"] = bool(drop_zero)

        CONFIG_PATH.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        st.success("Zapisano configs/default.yaml")

    with st.expander("Podgląd pełnego YAML"):
        st.code(CONFIG_PATH.read_text(encoding="utf-8"), language="yaml")


def _discover_star_files(input_dir: Path) -> list[Path]:
    """Wyszukuje rekurencyjnie surowe pliki STAR-Counts (jak CLI)."""
    found: set[Path] = set()
    if input_dir.exists():
        for pattern in STAR_FILE_PATTERNS:
            found.update(input_dir.rglob(pattern))
    return sorted(found)


def _output_stem(path: Path) -> str:
    """Buduje nazwę pliku wyjściowego niezależnie od konwencji (jak CLI)."""
    name = path.name
    for suffix in STAR_FILE_SUFFIXES:
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return path.stem


def _find_sample_sheet() -> Path | None:
    sheets = sorted(DATA_RAW.glob("gdc_sample_sheet*.tsv"))
    return sheets[0] if sheets else None


def render_parse(state: dict) -> None:
    """Parsowanie surowych plików STAR-Counts (TSV) do parquet."""
    st.header("Parsowanie plików STAR-Counts")
    st.caption("Konwersja surowych plików STAR-Counts (TSV) do formatu parquet.")

    star_files = _discover_star_files(DATA_RAW)

    if not star_files:
        st.warning(
            "Nie znaleziono surowych plików STAR-Counts w `data/raw/`.\n\n"
            "Pliki STAR (`*.rna_seq.augmented_star_gene_counts.tsv`) są pobierane "
            "przez etap **Pobieranie** do podkatalogów `data/raw/`. Jeśli zostały "
            "już sparsowane i usunięte, użyj gotowych parquetów w `data/interim/star_counts/`."
        )
        if state["parsed"]:
            st.info(f"W `data/interim/star_counts/` jest już **{state['parsed_count']}** "
                    "sparsowanych parquetów. Możesz przejść do etapu Macierz ekspresji.")
        return

    st.write(f"Znaleziono **{len(star_files)}** surowych plików STAR-Counts.")
    if state["parsed"]:
        st.info(f"Uwaga: w `data/interim/` jest już {state['parsed_count']} parquetów. "
                "Ponowne parsowanie nadpisze istniejące pliki.")

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)

    # Pokaż wynik z poprzedniego uruchomienia (przetrwał rerun)
    if st.session_state.get("parse_result"):
        st.success(st.session_state.parse_result)
        st.session_state.parse_result = None

    if st.button("Rozpocznij parsowanie", type="primary"):
        progress = st.progress(0.0, text="Parsowanie...")
        status = st.empty()
        errors = []
        total = len(star_files)

        for i, path in enumerate(star_files, start=1):
            try:
                df = parse_star_counts(path)
                out_path = DATA_INTERIM / f"{_output_stem(path)}.parquet"
                df.write_parquet(out_path)
            except (StarParserError, FileNotFoundError) as exc:
                errors.append(f"{path.name}: {exc}")
            progress.progress(i / total, text=f"Przetworzono {i}/{total} plików")
            if i % 25 == 0 or i == total:
                status.caption(f"Ostatni: {path.name}")

        if errors:
            st.error(f"Zakończono z {len(errors)} błędami:")
            with st.expander("Szczegóły błędów"):
                for e in errors:
                    st.text(e)
        else:
            st.session_state.parse_result = (
                f"Sparsowano {total} plików do data/interim/star_counts/."
            )
            st.rerun()


def render_build_matrix(state: dict) -> None:
    """Budowa macierzy ekspresji z parquetów (z progress barem)."""
    st.header("Budowa macierzy ekspresji")
    st.caption("Łączenie sparsowanych parquetów w macierz geny × próbki, z filtrem biotype.")

    st.markdown(
        "Ten etap scala pojedyncze pliki STAR-Counts (surowe zliczenia odczytów na gen, "
        "jeden plik na próbkę) w **jedną macierz geny × próbki** — wiersze to geny, kolumny "
        "to próbki, a każda komórka to poziom ekspresji genu w danej próbce. **To źródło cech "
        "(features) do analiz i modeli ML** — trzy wybory poniżej decydują, jak te cechy będą "
        "wyglądać. Rozwijane „📖” pod formularzem tłumaczą skróty i pomagają wybrać świadomie."
    )

    if not state["parsed"]:
        st.warning("Brak sparsowanych parquetów. Najpierw uruchom etap Parsowanie.")
        return

    sheet_path = _find_sample_sheet()
    if sheet_path is None:
        st.error("Brak sample sheet (`gdc_sample_sheet*.tsv`) w `data/raw/`.")
        return

    # Wczytaj config dla domyślnych wartości
    cfg = {}
    if CONFIG_PATH.exists():
        try:
            cfg = load_config(CONFIG_PATH)
        except ConfigError:
            cfg = {}

    norm = cfg.get("normalization", {}) if isinstance(cfg, dict) else {}
    default_metric = norm.get("method", "tpm")
    default_biotype = norm.get("biotype_filter", "protein_coding")

    st.write(f"Parquetów do połączenia: **{state['parsed_count']}**")

    col1, col2 = st.columns(2)
    metric_choice = col1.selectbox(
        "Metryka ekspresji",
        options=["tpm", "unstranded", "fpkm", "fpkm_uq", "counts"],
        index=0,
        help="Jednostka ekspresji w macierzy. Domyślna z konfiguracji: " + str(default_metric)
        + ". TPM/FPKM znormalizowane (porównywalne między próbkami), counts/unstranded surowe. "
        "Szczegóły i rekomendacja w bloku „📖 Metryki ekspresji” poniżej.",
    )
    dup_strategy = col2.selectbox(
        "Strategia duplikatów",
        options=["deepest", "first", "fail"],
        index=0,
        help="Gdy jeden przypadek ma kilka próbek (aliquotów), która trafi do macierzy. "
        "deepest = ta z największym sygnałem (zalecane dla TCGA). Szczegóły w bloku "
        "„📖 Strategia duplikatów” poniżej.",
    )
    biotype = st.text_input(
        "Filtr biotype",
        value=default_biotype or "",
        help="Które klasy genów zostawić. protein_coding = tylko geny kodujące białko "
        "(~20 tys., mniej szumu); puste = wszystkie ~60 tys. Szczegóły w bloku "
        "„📖 Filtr biotype” poniżej.",
    )

    with st.expander("📖 Metryki ekspresji — co znaczą skróty i którą wybrać"):
        st.markdown("**Metryka to jednostka, w jakiej zapisana jest ekspresja każdego genu.** "
                    "Wybór wprost decyduje, czy wartości są porównywalne między próbkami — "
                    "co jest kluczowe dla analiz i modeli ML.")
        st.markdown("**counts / unstranded** — surowa liczba odczytów (reads) zmapowanych do "
                    "genu; „unstranded” = zliczanie bez rozróżniania nici DNA (standard STAR w "
                    "TCGA). Nieznormalizowane: dłuższe geny i głębiej zsekwencjonowane próbki "
                    "mają z definicji wyższe wartości, więc bez dalszej normalizacji nie są "
                    "porównywalne ani między genami, ani między próbkami.")
        st.markdown("**TPM — Transcripts Per Million** — normalizuje na długość genu i głębokość "
                    "tak, że wartości w każdej próbce sumują się do miliona. Dzięki temu są "
                    "porównywalne MIĘDZY próbkami. Najczęstszy wybór do eksploracji i jako "
                    "wejście do ML.")
        st.markdown("**FPKM — Fragments Per Kilobase per Million** — też normalizuje na długość "
                    "i głębokość, ale ich sumy nie są stałe między próbkami, co lekko obciąża "
                    "porównania między próbkami. Zbliżony do TPM, lecz TPM jest zwykle "
                    "preferowany.")
        st.markdown("**FPKM-UQ (upper-quartile)** — wariant FPKM normalizowany górnym kwartylem "
                    "zamiast całkowitą głębokością; odporniejszy na kilka skrajnie wysokich "
                    "genów, używany w części potoków TCGA.")
        st.markdown("**Rekomendacja:** do eksploracji i modeli ML → **TPM** (najlepiej "
                    "log2(TPM+1)). Jeśli planujesz analizę różnicowej ekspresji narzędziami "
                    "DESeq2/edgeR → wybierz **counts**, bo one oczekują surowych zliczeń i "
                    "normalizują je same.")
        st.markdown("**Pod ML:** metryka to jednostka Twoich cech. Do modeli porównujących "
                    "próbki (klasyfikacja, przeżycie) używaj TPM/log-TPM. Surowe counts podawaj "
                    "tylko narzędziom normalizującym wewnętrznie. Niespójna metryka między "
                    "próbkami = obciążenie i ryzyko wycieku.")

    with st.expander("📖 Strategia duplikatów — skąd się biorą i którą wybrać"):
        st.markdown("**Jeden przypadek (pacjent) bywa zsekwencjonowany w kilku próbkach — "
                    "aliquotach:** np. dwa wycinki tego samego guza albo powtórki techniczne. "
                    "Macierz potrzebuje jednej kolumny na próbkę, więc trzeba zdecydować, którą "
                    "wziąć.")
        st.markdown("**deepest** — bierze aliquot z największą sumą ekspresji (proxy głębokości "
                    "sekwencjonowania: najwięcej sygnału, najmniej szumu). **Zalecane dla TCGA** "
                    "i deterministyczne.")
        st.markdown("**first** — bierze pierwszy napotkany, arbitralnie. Szybkie, ale wybór nie "
                    "jest biologicznie uzasadniony ani stabilny między buildami.")
        st.markdown("**fail** — przerywa budowę z błędem, gdy wykryje duplikat. Wybierz, gdy "
                    "chcesz ręcznie rozstrzygnąć każdy przypadek.")
        st.markdown("**Pod ML:** spójny, deterministyczny wybór (deepest) daje powtarzalną "
                    "macierz — ta sama próbka trafia do cech przy każdym buildzie, bez "
                    "przypadkowej zmienności psującej porównania modeli.")

    with st.expander("📖 Filtr biotype — co to jest i po co filtrować"):
        st.markdown("**Biotyp to klasa funkcjonalna genu** z anotacji: protein_coding (koduje "
                    "białko), lncRNA, miRNA, pseudogene, rRNA i inne. STAR zlicza WSZYSTKIE "
                    "(~60 tys.), ale duża część to geny niekodujące i pseudogeny o niskiej, "
                    "szumnej ekspresji.")
        st.markdown("**protein_coding** — zostawia tylko geny kodujące białko (~19–20 tys., "
                    "czyli ~33% z ~60 tys.). Mniej szumu; standard do większości analiz.")
        st.markdown("**puste pole** — zatrzymuje wszystkie geny. Wybierz świadomie, gdy "
                    "interesują Cię klasy niekodujące (lncRNA, miRNA itp.).")
        st.markdown("**Pod ML:** filtr to redukcja wymiaru i szumu już na wejściu — mniej cech, "
                    "za to każda niesie więcej sygnału, co zmniejsza ryzyko przeuczenia. Puste "
                    "zostaw tylko przy świadomym badaniu genów niekodujących.")

    if state["matrix_built"]:
        st.info("Macierz już istnieje — ponowne budowanie ją nadpisze.")

    # Pokaż wynik z poprzedniego uruchomienia (przetrwał rerun)
    if st.session_state.get("matrix_result"):
        st.success(st.session_state.matrix_result)
        st.session_state.matrix_result = None

    if st.button("Zbuduj macierz", type="primary"):
        try:
            sheet_df = parse_sample_sheet(sheet_path)
        except Exception as exc:
            st.error(f"Błąd wczytania sample sheet: {exc}")
            return

        parquet_paths = sorted(DATA_INTERIM.glob("*.parquet"))
        try:
            metric_resolved = resolve_metric(metric_choice) if metric_choice else "tpm_unstranded"
        except ConfigError as exc:
            st.error(f"Niepoprawna metryka: {exc}")
            return

        progress = st.progress(0.0, text="Budowanie macierzy...")

        def cb(done: int, total: int) -> None:
            progress.progress(done / total, text=f"Połączono {done}/{total} próbek")

        try:
            matrix = build_expression_matrix(
                parquet_paths,
                sheet_df,
                metric=metric_resolved,
                duplicate_strategy=dup_strategy,
                biotype_filter=biotype if biotype.strip() else None,
                progress_callback=cb,
            )
            DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
            out_path = DATA_PROCESSED / "expression_matrix.parquet"
            matrix.write_parquet(out_path)
            n_genes = matrix.height
            n_samples = matrix.width - 1
            st.session_state.matrix_result = (
                f"Macierz zbudowana: {n_genes} genów × {n_samples} próbek. "
                f"Zapisano do data/processed/expression_matrix.parquet."
            )
        except Exception as exc:
            st.error(f"Błąd budowy macierzy: {type(exc).__name__}: {exc}")
            import traceback
            with st.expander("Szczegóły błędu"):
                st.code(traceback.format_exc())
            return
        st.rerun()


def render_build_survival(state: dict) -> None:
    """Budowa zbioru przeżywalności (integracja macierz + clinical)."""
    st.header("Budowa zbioru przeżywalności")
    st.caption("Integracja macierzy ekspresji z danymi klinicznymi i filtrami warunkowymi.")

    if not state["matrix_built"]:
        st.warning("Brak macierzy ekspresji. Najpierw uruchom etap Macierz ekspresji.")
        return

    sheet_path = _find_sample_sheet()
    clinical_path = DATA_RAW / "clinical.tsv"

    with st.expander("📄 Wgraj / podmień plik kliniczny clinical.tsv",
                     expanded=not clinical_path.exists()):
        st.caption(
            "Build używa `data/raw/clinical.tsv`. Jeśli obecny pochodzi z API i ma braki "
            "(np. pusta płeć), wgraj pełny `clinical.tsv` z eksportu **„cart”** portalu GDC "
            "(portal.gdc.cancer.gov → koszyk → Download → Clinical → TSV) — zastąpi obecny."
        )
        _up = st.file_uploader("clinical.tsv", type=["tsv", "txt"], key="clinical_override")
        if _up is not None:
            DATA_RAW.mkdir(parents=True, exist_ok=True)
            clinical_path.write_bytes(_up.getvalue())
            try:
                _pc = parse_clinical(clinical_path)
                _fill = " · ".join(
                    f"{c} {100 * int(_pc[c].is_not_null().sum()) / _pc.height:.0f}%"
                    for c in ("age_at_index", "gender", "ajcc_pathologic_stage")
                    if c in _pc.columns
                )
                st.success(f"Zapisano clinical.tsv. Kompletność po parsowaniu: {_fill}.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Wgrany plik nie parsuje się jako clinical.tsv: {exc}")

    if sheet_path is None or not clinical_path.exists():
        st.error(
            "Brak sample sheet (`gdc_sample_sheet*.tsv`) lub `clinical.tsv` w `data/raw/`. "
            "Wgraj clinical.tsv powyżej albo dodaj pliki przez etap Pobieranie/Wgrywanie."
        )
        return

    cfg = {}
    if CONFIG_PATH.exists():
        try:
            cfg = load_config(CONFIG_PATH)
        except ConfigError:
            cfg = {}
    surv = cfg.get("survival", {}) if isinstance(cfg, dict) else {}
    default_min_follow = int(surv.get("min_follow_up_days", 30))
    default_drop_zero = bool(surv.get("drop_zero_time", True))

    col1, col2 = st.columns(2)
    tumor_only = col1.checkbox("Tylko próbki nowotworowe", value=True,
                               help="Wyklucza próbki normalne (tissue normal).")
    drop_zero = col2.checkbox("Usuń artefakty time<=0", value=default_drop_zero,
                              help="Usuwa próbki z czasem <=0 (artefakt PHI).")
    min_follow = st.number_input(
        "Min follow-up (dni)", min_value=0, value=default_min_follow,
        help="Usuwa krótkie cenzury. Wczesne zgony są zachowane.",
    )

    if state["survival_built"]:
        st.info("Zbiór przeżywalności już istnieje — ponowne budowanie go nadpisze.")
        with st.expander("🔧 Napraw dane kliniczne w istniejącym zbiorze (bez przebudowy)"):
            st.caption(
                "Dograja wiek/płeć/stadium z `data/raw/clinical.tsv` do istniejącego zbioru "
                "(złączenie po pacjencie). Etykiety przeżycia i geny pozostają nietknięte — "
                "użyj, gdy zbiór ma pustą/niepełną kolumnę kliniczną, a nie chcesz przebudowywać."
            )
            if st.button("Napraw dane kliniczne"):
                try:
                    _dsp = DATA_PROCESSED / "survival_dataset.parquet"
                    _ds = pl.read_parquet(_dsp)
                    _clin = parse_clinical(clinical_path)
                    _cov = ["age_at_index", "gender", "ajcc_pathologic_stage"]
                    _before = " · ".join(
                        f"{c} {100 * int(_ds[c].is_not_null().sum()) / _ds.height:.0f}%"
                        for c in _cov if c in _ds.columns
                    )
                    _order = _ds.columns
                    _ds = _ds.drop([c for c in _cov if c in _ds.columns]).join(
                        _clin.select(
                            ["case_submitter_id"] + [c for c in _cov if c in _clin.columns]
                        ),
                        left_on="case_id", right_on="case_submitter_id", how="left",
                    ).select([c for c in _order if c in _ds.columns])
                    _after = " · ".join(
                        f"{c} {100 * int(_ds[c].is_not_null().sum()) / _ds.height:.0f}%"
                        for c in _cov if c in _ds.columns
                    )
                    _ds.write_parquet(_dsp)
                    st.success(f"Naprawiono. PRZED: {_before}  →  PO: {_after}")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Naprawa nie powiodła się: {exc}")

    # Pokaż wynik z poprzedniego uruchomienia (przetrwał rerun)
    if st.session_state.get("survival_result"):
        st.success(st.session_state.survival_result)
        st.session_state.survival_result = None

    if st.button("Zbuduj zbiór przeżywalności", type="primary"):
        try:
            with st.spinner("Wczytywanie danych (macierz, sample sheet, clinical)..."):
                matrix = pl.read_parquet(DATA_PROCESSED / "expression_matrix.parquet")
                sheet_df = parse_sample_sheet(sheet_path)
                clinical_df = parse_clinical(clinical_path)

            with st.spinner("Integracja i filtrowanie warunkowe..."):
                dataset = build_survival_dataset(
                    matrix,
                    sheet_df,
                    clinical_df,
                    tumor_only=tumor_only,
                    min_follow_up_days=int(min_follow),
                    drop_zero_time=drop_zero,
                )
            DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
            out_path = DATA_PROCESSED / "survival_dataset.parquet"
            dataset.write_parquet(out_path)

            n_samples = dataset.height
            n_events = int(dataset["event"].sum()) if "event" in dataset.columns else 0
            censoring = (1 - n_events / n_samples) * 100 if n_samples else 0
            st.session_state.survival_result = (
                f"Zbiór zbudowany: {n_samples} próbek, zdarzenia: {n_events}, "
                f"cenzurowanie: {censoring:.1f}%. "
                f"Zapisano do data/processed/survival_dataset.parquet."
            )
        except Exception as exc:
            st.error(f"Błąd budowy zbioru: {type(exc).__name__}: {exc}")
            import traceback
            with st.expander("Szczegóły błędu"):
                st.code(traceback.format_exc())
            return
        st.rerun()


def render_validate(state: dict) -> None:
    """Walidacja spójności kohorty (QC) z raportem."""
    st.header("Walidacja kohorty")
    st.caption("Kontrola spójności: dopasowanie próbek do plików STAR, danych klinicznych, duplikaty.")

    if not state["parsed"]:
        st.warning("Brak sparsowanych parquetów. Najpierw uruchom etap Parsowanie.")
        return

    sheet_path = _find_sample_sheet()
    clinical_path = DATA_RAW / "clinical.tsv"
    if sheet_path is None:
        st.error("Brak sample sheet (`gdc_sample_sheet*.tsv`) w `data/raw/`.")
        return
    if not clinical_path.exists():
        st.error("Brak pliku `clinical.tsv` w `data/raw/`.")
        return

    st.write(f"Parquetów do sprawdzenia: **{state['parsed_count']}**")

    # Pokaż raport z poprzedniego uruchomienia (przetrwał rerun)
    if st.session_state.get("qc_summary"):
        _render_qc_report(st.session_state.qc_summary, st.session_state.get("qc_issues", []))

    if st.button("Uruchom walidację", type="primary"):
        try:
            with st.spinner("Wczytywanie metadanych i sprawdzanie spójności..."):
                sheet_df = parse_sample_sheet(sheet_path)
                clinical_df = parse_clinical(clinical_path)
                available_stems = discover_stems(DATA_INTERIM)
                report = run_cohort_qc(sheet_df, clinical_df, available_stems)
                log_path = save_qc_report(report, PROJECT_ROOT / "logs" / "qc")

            # Zapis do session_state (struktura serializowalna)
            st.session_state.qc_summary = report.summary()
            st.session_state.qc_issues = [
                {
                    "severity": i.severity.value,
                    "category": i.category.value,
                    "message": i.message,
                    "context": i.context,
                }
                for i in report.issues
            ]
            st.session_state.qc_log_path = str(log_path)
        except Exception as exc:
            st.error(f"Błąd walidacji: {type(exc).__name__}: {exc}")
            import traceback
            with st.expander("Szczegóły błędu"):
                st.code(traceback.format_exc())
            return
        st.rerun()


def _render_qc_report(summary: dict, issues: list) -> None:
    """Renderuje raport QC: werdykt kontekstowy + problemy z wyjaśnieniem."""
    c = classify_qc(summary, issues)

    # Podsumowanie liczbowe
    col1, col2, col3 = st.columns(3)
    col1.metric("Wszystkie rozjazdy", c["total"])
    col2.metric("Obsługiwane automatycznie", c["n_handled"])
    col3.metric("Wymagają uwagi", c["n_action"])

    # Werdykt kontekstowy (jedno źródło: report_view.classify_qc)
    if c["verdict_level"] == "ok":
        st.success(c["verdict"])
    else:
        st.warning(c["verdict"])

    log_path = st.session_state.get("qc_log_path")
    if log_path:
        st.caption(f"Raport JSON: `{log_path}`")

    # Sekcja: wymagające uwagi (pokazane jako pierwsze, rozwinięte)
    if c["action_by_cat"]:
        st.subheader("Warto sprawdzić")
        for cat, items in c["action_by_cat"].items():
            with st.expander(f"⚠️ {c['labels'].get(cat, cat)} ({len(items)})", expanded=True):
                st.caption(c["actions"].get(cat, ""))
                for issue in items:
                    st.markdown(f"- {issue['message']}")

    # Sekcja: obsługiwane automatycznie (zwinięte, informacyjne)
    if c["handled_by_cat"]:
        st.subheader("Obsługiwane automatycznie przez pipeline")
        st.caption(
            "Poniższe rozjazdy są typowe dla danych TCGA i nie wymagają działania — "
            "pipeline radzi sobie z nimi sam. Pokazane dla pełnej transparentności."
        )
        for cat, items in c["handled_by_cat"].items():
            with st.expander(f"{c['labels'].get(cat, cat)} ({len(items)})", expanded=False):
                st.caption(c["actions"].get(cat, ""))
                for issue in items[:50]:
                    ctx = issue.get("context", {})
                    ctx_str = ""
                    if ctx:
                        sample = ctx.get("sample_id") or ctx.get("case_id") or ctx.get("stem", "")
                        ctx_str = f" ({sample})" if sample else ""
                    st.markdown(f"- {issue['message']}{ctx_str}")
                if len(items) > 50:
                    st.caption(f"... oraz {len(items) - 50} więcej (pełna lista w raporcie JSON)")


def _render_sample_clinical(ds, sample_col: str) -> None:
    """Pokazuje dane kliniczne wybranej próbki (jeśli jest w survival_dataset)."""
    # sample_id w macierzy to pełny aliquot; w survival to sample_id (krótszy)
    # Dopasowanie po prefiksie (case_id: pierwsze 3 segmenty TCGA-XX-XXXX)
    if "sample_id" not in ds.columns:
        return

    # Próba dopasowania: dokładne, potem po prefiksie próbki, potem po case
    match = ds.filter(pl.col("sample_id") == sample_col)
    if match.height == 0:
        # spróbuj po prefiksie (case_id z sample_col, np. TCGA-XX-XXXX)
        parts = sample_col.split("-")
        if len(parts) >= 3:
            case_prefix = "-".join(parts[:3])
            match = ds.filter(pl.col("case_id").str.starts_with(case_prefix))

    if match.height == 0:
        st.caption("ℹ️ Ta próbka nie występuje w zbiorze przeżywalności "
                   "(np. próbka normalna lub odfiltrowana) — brak danych klinicznych.")
        return

    row = match.row(0, named=True)
    st.markdown("**Dane kliniczne próbki:**")
    cols = st.columns(4)
    fields = [
        ("case_id", "Pacjent"),
        ("ajcc_pathologic_stage", "Stadium"),
        ("age_at_index", "Wiek"),
        ("gender", "Płeć"),
    ]
    for i, (key, label) in enumerate(fields):
        val = row.get(key)
        if val is not None:
            cols[i].metric(label, str(val))

    # Druga linia: czas obserwacji + status
    time_days = row.get("time")
    event = row.get("event")
    tissue = row.get("tissue_type")
    info_parts = []
    if time_days is not None:
        info_parts.append(f"Czas obserwacji: **{time_days:.0f} dni** ({time_days/365.25:.1f} lat)")
    if event is not None:
        status = "zgon" if event else "żyje/cenzura"
        info_parts.append(f"Status: **{status}**")
    if tissue is not None:
        info_parts.append(f"Tkanka: **{tissue}**")
    if info_parts:
        st.caption(" | ".join(info_parts))


def _dash_explain(co):
    """Jednozdaniowy podpis pod wykresem (co pokazuje). Pełny przewodnik: README."""
    st.caption("ℹ️ " + co)


# Pastelowa paleta statusów dla zakładki Gotowość ML — stonowana, pod ciepłe GUI.
_ML_PASTEL = {
    "green":  {"bg": "rgba(122,155,94,0.13)",  "bar": "#7a9b5e", "dot": "#8caf6e"},
    "yellow": {"bg": "rgba(201,166,74,0.15)",  "bar": "#c2a24a", "dot": "#d4b45e"},
    "red":    {"bg": "rgba(191,122,106,0.15)", "bar": "#bd7a6a", "dot": "#cf8a78"},
    "info":   {"bg": "rgba(150,150,140,0.11)", "bar": "#9a9a90", "dot": "#adad9f"},
}


def _html_esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _ml_card_html(m: dict) -> str:
    """Pastelowa karta metryki: miękkie tło + boczny akcent + kropka statusu."""
    c = _ML_PASTEL.get(m["status"], _ML_PASTEL["info"])
    action = (
        f'<div style="font-size:0.82em;margin-top:4px;opacity:0.85;">\u279C '
        f'{_html_esc(m["action"])}</div>'
        if m.get("action") else ""
    )
    return (
        f'<div style="border-left:4px solid {c["bar"]};background:{c["bg"]};'
        f'padding:9px 13px;margin:6px 0;border-radius:6px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px;">'
        f'<span style="font-weight:600;">'
        f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
        f'background:{c["dot"]};margin-right:8px;vertical-align:middle;"></span>'
        f'{_html_esc(m["label"])}</span>'
        f'<span style="font-weight:700;white-space:nowrap;">{_html_esc(m["value"])}</span>'
        f'</div>'
        f'<div style="font-size:0.82em;opacity:0.72;margin-top:4px;">{_html_esc(m["note"])}</div>'
        f'{action}</div>'
    )


def render_dashboard(state: dict) -> None:
    """Dashboard analityczny - wizualizacje ekspresji i przeżywalności."""
    st.header("Dashboard analityczny")
    st.caption("Kluczowe wizualizacje biologiczne kohorty TCGA-LUAD.")

    if not state["survival_built"]:
        st.warning("Brak zbioru przeżywalności. Najpierw uruchom etap Zbiór przeżywalności.")
        return

    # Wczytanie danych (cache w session_state by nie czytać przy każdym przełączeniu zakładki)
    try:
        survival_path = DATA_PROCESSED / "survival_dataset.parquet"
        ds = pl.read_parquet(survival_path)
        pdf = ds.to_pandas()
    except Exception as exc:
        st.error(f"Błąd wczytania zbioru przeżywalności: {exc}")
        return

    # Podstawowe statystyki kohorty (jedno źródło: sr.cohort_summary — 1:1 z terminalem)
    summary = sr.cohort_summary(ds)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Próbki (tumor)", summary["n_samples"])
    c2.metric("Zdarzenia (zgony)", summary["n_events"])
    c3.metric("Cenzurowane", f'{summary["n_censored"]} ({summary["censoring_pct"]:.1f}%)')
    c4.metric("Geny w macierzy", summary["n_genes"])
    c5.metric("Mediana obserwacji", f'{summary["median_followup_years"]:.2f} lat')
    stage_str = "  ·  ".join(
        f'{k}: {summary["stage_counts"][k]}'
        for k in ("I", "II", "III", "IV", "Unknown") if k in summary["stage_counts"]
    )
    st.caption(f"Rozkład stadiów (AJCC): {stage_str}")

    st.caption(
        "Trzy zakładki: **Przeżywalność** (jakość etykiety), **Ekspresja** (jakość "
        "cech), **Gotowość ML** (ocena przydatności danych pod modele). Pełny "
        "przewodnik po wykresach — w README, sekcja „Przewodnik po dashboardzie "
        "analitycznym”."
    )

    tab_surv, tab_expr, tab_ml = st.tabs(["Przeżywalność", "Ekspresja", "Gotowość ML"])

    # ===== ZAKŁADKA PRZEŻYWALNOŚĆ =====
    with tab_surv:
        st.markdown(
            "**Target — czy etykieta przeżycia jest dobrze zdefiniowana i niesie sygnał?** "
            "Te wykresy patrzą na zmienną celu: od surowego rozkładu, przez kliniczny baseline, "
            "po sprawdzenie, czy ekspresja genów dokłada wartość prognostyczną ponad klinikę."
        )
        st.subheader("Kaplan-Meier — cała kohorta")
        try:
            fig, stats = viz.km_overall(ds)
            st.plotly_chart(fig, use_container_width=True)
            if stats["median_os"]:
                cols = st.columns(4)
                cols[0].metric("Mediana OS", f"{stats['median_os']:.2f} lat")
                for i, y in enumerate([1, 3, 5]):
                    s = stats.get(f"surv_{y}y")
                    if s is not None:
                        cols[i+1].metric(f"Przeżycie {y}-letnie", f"{s*100:.0f}%")
        except Exception as exc:
            st.error(f"Błąd wykresu KM overall: {exc}")

        _dash_explain(co="Krzywą przeżycia całej kohorty: jaki odsetek pacjentów żyje po danym czasie (spada z liczbą zgonów). Zacieniony pas to 95% przedział ufności — im węższy, tym pewniejsze oszacowanie.")
        st.divider()
        st.subheader("Kaplan-Meier — per stadium")
        st.caption("Stadium zaawansowania jako najsilniejszy predyktor kliniczny.")
        try:
            fig, info = viz.km_per_stage(ds)
            st.plotly_chart(fig, use_container_width=True)
            if info["p_value"] is not None:
                sig = "istotne" if info["p_value"] < 0.05 else "nieistotne"
                st.caption(f"Log-rank test (różnice między stadiami): p = {info['p_value']:.2e} ({sig})")
        except Exception as exc:
            st.error(f"Błąd wykresu KM per stage: {exc}")

        _dash_explain(co="Te same krzywe przeżycia, ale rozbite na stadia AJCC I–IV — każda grupa osobno — plus wynik testu log-rank na różnice między nimi.")
        st.divider()
        st.subheader("Model Coxa — kowarianty kliniczne")
        st.caption("Proporcjonalne hazardy (wiek + płeć + stadium). Forest plot "
                   "kwantyfikuje wpływ każdego czynnika na ryzyko zgonu (HR).")
        try:
            fig, info = viz.cox_clinical(ds)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                cc = st.columns(3)
                cc[0].metric("C-index", f"{info['c_index']:.3f}")
                cc[1].metric("Pacjenci (I–IV)", info["n"])
                cc[2].metric("Zdarzenia", info["n_events"])
                st.dataframe(info["table"], use_container_width=True, hide_index=True)
                st.caption("HR > 1 = wyższe ryzyko zgonu, HR < 1 = ochronne. "
                           "C-index: 0.5 losowy, 1.0 perfekcyjny (baseline kliniczny zwykle 0.6–0.7).")
            else:
                st.info(info.get("error", "Brak danych do modelu Coxa."))
        except Exception as exc:
            st.error(f"Błąd modelu Coxa: {exc}")

        _dash_explain(co="Model proporcjonalnych hazardów Coxa: ile każdy czynnik (wiek, płeć, stadium) dokłada do ryzyka zgonu niezależnie od pozostałych. Forest plot pokazuje HR z 95% przedziałem ufności, tabela podaje liczby.")
        st.divider()
        st.subheader("Kaplan-Meier — sygnatura wielogenowa")
        st.caption("Panel ekspresyjny a priori (różnicowanie + proliferacja + inwazja). "
                   "Kombinacja genów jako sygnał prognostyczny.")
        try:
            fig, info = viz.km_signature(ds)
            st.plotly_chart(fig, use_container_width=True)
            if info["p_value"] is not None:
                sig = "istotne" if info["p_value"] < 0.05 else "nieistotne"
                st.caption(f"Log-rank test (high vs low): p = {info['p_value']:.4f} ({sig}), "
                           f"panel {info['n_genes']} genów")
        except Exception as exc:
            st.error(f"Błąd wykresu sygnatury: {exc}")

        _dash_explain(co="Pacjentów podzielonych na grupy high/low według łącznego score 7-genowego panelu (różnicowanie + proliferacja + inwazja), z testem log-rank między grupami.")
        st.divider()
        st.subheader("Kaplan-Meier — pojedynczy gen")
        st.caption("Wybierz gen, by zobaczyć stratyfikację przeżycia high/low względem mediany ekspresji.")
        gene_options = list(viz.LUAD_MARKERS.keys()) + [
            g for g in viz.SIGNATURE_PANEL.keys() if g not in viz.LUAD_MARKERS
        ]
        selected_gene = st.selectbox(
            "Gen", options=sorted(set(gene_options)), index=0,
            help="Wybierz gen z listy markerów LUAD i panelu sygnatury. Wykres porówna "
                 "przeżycie próbek z ekspresją powyżej vs poniżej mediany tego genu "
                 "(high vs low).",
        )
        if selected_gene:
            # Charakterystyka genu (precyzyjna, spójna)
            gene_desc = viz.GENE_INFO.get(selected_gene)
            if gene_desc:
                st.info(f"**{selected_gene}** — {gene_desc}")
            ensg = viz.LUAD_MARKERS.get(selected_gene) or viz.SIGNATURE_PANEL.get(selected_gene, (None,))[0]
            if ensg:
                try:
                    fig, info = viz.km_single_gene(ds, selected_gene, ensg)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                        if info.get("p_value") is not None:
                            sig = "istotne" if info["p_value"] < 0.05 else "nieistotne (trend)"
                            st.caption(f"Log-rank test: p = {info['p_value']:.4f} ({sig})")
                    else:
                        st.info(info.get("error", "Brak danych dla genu."))
                except Exception as exc:
                    st.error(f"Błąd wykresu genu: {exc}")

        _dash_explain(co="Dla wybranego z listy genu — przeżycie grupy o ekspresji powyżej vs poniżej mediany — z krótką charakterystyką biologiczną genu i testem log-rank.")
        st.divider()
        st.subheader("Kaplan-Meier — porównanie wielu genów")
        st.caption("Wybierz kilka genów, by porównać prognostyczny efekt ich "
                   "wysokiej ekspresji (krzywe grup „high”) na jednym wykresie.")
        multi_options = list(viz.LUAD_MARKERS.keys()) + [
            g for g in viz.SIGNATURE_PANEL.keys() if g not in viz.LUAD_MARKERS
        ]
        default_multi = [g for g in ["NKX2-1", "MKI67", "BIRC5"] if g in multi_options]
        selected_multi = st.multiselect(
            "Geny do porównania (2-7)", options=sorted(set(multi_options)),
            default=default_multi, max_selections=7,
            help="Wybierz 2–7 genów. Wykres nałoży krzywe grup high (ekspresja powyżej "
                 "mediany) każdego genu, a tabela poda log-rank p osobno dla każdego — "
                 "szybkie porównanie siły prognostycznej.",
        )
        if len(selected_multi) >= 2:
            genes_pairs = []
            for g in selected_multi:
                ensg = viz.LUAD_MARKERS.get(g) or viz.SIGNATURE_PANEL.get(g, (None,))[0]
                if ensg:
                    genes_pairs.append((g, ensg))
            try:
                fig, results = viz.km_multi_gene(ds, genes_pairs)
                st.plotly_chart(fig, use_container_width=True)
                # Tabela log-rank per gen
                rows = []
                for r in results:
                    if r["p_value"] is not None:
                        sig = "istotne" if r["p_value"] < 0.05 else "nieistotne"
                        rows.append({"Gen": r["gene"], "Log-rank p (high vs low)": f"{r['p_value']:.4f}", "Ocena": sig})
                    else:
                        rows.append({"Gen": r["gene"], "Log-rank p (high vs low)": "—", "Ocena": r.get("note", "")})
                if rows:
                    st.dataframe(rows, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"Błąd wykresu porównania: {exc}")
        elif len(selected_multi) == 1:
            st.info("Wybierz co najmniej 2 geny do porównania.")

        _dash_explain(co="Krzywe grup high dla kilku wybranych genów naraz na jednym wykresie, plus tabelę log-rank p osobno dla każdego genu.")
        st.divider()
        st.subheader("Model Coxa — klinika + panel genów")
        st.caption("Czy panel ekspresyjny dodaje wartość prognostyczną PONAD model kliniczny? "
                   "Każdy gen wchodzi osobno (z-score log2 TPM); porównujemy C-index obu modeli "
                   "na tej samej kohorcie.")
        try:
            fig, info = viz.cox_clinical_genes(ds)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                cc = st.columns(3)
                cc[0].metric("C-index (klinika)", f"{info['c_index_clinical']:.3f}")
                cc[1].metric("C-index (klinika + geny)", f"{info['c_index_genes']:.3f}",
                             delta=f"{info['delta']:+.3f}")
                cc[2].metric("Pacjenci", info["n"])
                st.dataframe(info["gene_table"], use_container_width=True, hide_index=True)
                if info["delta"] > 0.01:
                    st.success(f"Panel genów poprawia predykcję o {info['delta']:+.3f} C-index — "
                               "ekspresja dodaje sygnał prognostyczny ponad samą klinikę.")
                elif info["delta"] > 0:
                    st.info(f"Panel genów nieznacznie poprawia predykcję ({info['delta']:+.3f}).")
                else:
                    st.warning(f"Panel genów nie poprawia C-index w tej kohorcie ({info['delta']:+.3f}).")
                if info.get("missing"):
                    st.caption(f"Geny pominięte (brak w macierzy): {', '.join(info['missing'])}.")
                st.caption("HR per 1 SD ekspresji. Pojedyncze geny bywają nieistotne — "
                           "siła sygnału tkwi w kombinacji (możliwe artefakty kolinearności).")
            else:
                st.info(info.get("error", "Brak danych do modelu Coxa."))
        except Exception as exc:
            st.error(f"Błąd modelu Coxa (geny): {exc}")

        _dash_explain(co="Model łączący klinikę I panel genów, z bezpośrednim porównaniem C-index: sama klinika vs klinika + geny. Forest plot pokazuje HR genów już po korekcie o klinikę.")
        # ===== Rygor statystyczny: α, FDR panelu, sensitivity (jedno źródło: sr) =====
        st.divider()
        st.subheader("Rygor statystyczny — α, korekcja FDR, sensitivity analysis")
        st.caption("Poprawki metodologiczne: jawny próg istotności, korekcja wielokrotnego "
                   "testowania (Benjamini-Hochberg FDR) na panelu 7 genów oraz sensitivity "
                   "analysis (Schoenfeld) zamiast nieważnego post-hoc power.")
        try:
            stat = sr.statistical_rigor_report(ds)
            st.markdown(f"**Poziom istotności α = {stat['alpha']}** (dwustronnie; "
                        "korekcja wielokrotności: Benjamini-Hochberg FDR)")

            cfdr1, cfdr2 = st.columns(2)
            for col, key, title in ((cfdr1, "fdr_single_gene", "FDR — single-gene (log-rank KM)"),
                                    (cfdr2, "fdr_cox", "FDR — Cox (multivariate)")):
                with col:
                    st.markdown(f"**{title}**")
                    fr = stat[key]
                    if "error" in fr:
                        st.info(fr["error"])
                    else:
                        tbl = pl.DataFrame({
                            "Gen": [r["name"] for r in fr["rows"]],
                            "p (surowe)": [r["p"] for r in fr["rows"]],
                            "q (BH-FDR)": [r["q"] for r in fr["rows"]],
                            "FDR<α": ["TAK" if r["reject_fdr"] else "—" for r in fr["rows"]],
                        })
                        st.dataframe(tbl, use_container_width=True, hide_index=True)
                        st.caption(f"Istotnych: surowo {fr['n_sig_raw']} → po FDR "
                                   f"{fr['n_sig_fdr']} z {fr['n_tested']}.")

            sens = stat["sensitivity"]
            st.markdown(f"**Sensitivity analysis (Schoenfeld)** — przy {sens['n_events']} "
                        "zdarzeniach (median split):")
            mc = st.columns(len(sens["min_detectable_hr"]))
            for j, (pw, hr) in enumerate(sens["min_detectable_hr"].items()):
                mc[j].metric(f"Min. wykrywalny HR @ {pw:.0%} mocy", f"{hr:.2f}")
            pc = st.columns(len(sens["power_at_hr"]))
            for j, (hr, pw) in enumerate(sens["power_at_hr"].items()):
                pc[j].metric(f"Moc @ HR={hr}", f"{pw:.0%}")
            st.caption("Post-hoc power z obserwowanego efektu jest nieważne (Hoenig & Heisey 2001); "
                       "dla przeżycia liczą się zdarzenia, nie liczba pacjentów (n = cała kohorta TCGA).")
        except Exception as exc:
            st.error(f"Błąd raportu statystycznego: {exc}")

        _dash_explain(co="Dwie poprawki uczciwości statystycznej: (1) korekcję wielokrotnego testowania Benjamini-Hochberg (FDR) na panelu 7 genów — surowe p obok q po korekcie; (2) sensitivity analysis Schoenfelda: jak silny efekt (HR) da się w ogóle wykryć przy danej liczbie zdarzeń.")
    # ===== ZAKŁADKA EKSPRESJA =====
    with tab_expr:
        st.markdown(
            "**Features — czy cechy są czyste, porównywalne i wolne od artefaktów?** "
            "Te wykresy patrzą na ekspresję genów jako wejście modelu: format i normalizację, "
            "jakość pojedynczych próbek, zgodność z biologią oraz ryzyko efektu batch."
        )
        # Macierz ekspresji (osobny plik - większy)
        matrix_path = DATA_PROCESSED / "expression_matrix.parquet"
        if not matrix_path.exists():
            st.warning("Brak macierzy ekspresji do wizualizacji ekspresji.")
        else:
            try:
                matrix = pl.read_parquet(matrix_path)
                sample_cols = [c for c in matrix.columns if c != "gene_id"]
            except Exception as exc:
                st.error(f"Błąd wczytania macierzy: {exc}")
                sample_cols = []

            if sample_cols:
                # Podsumowanie macierzy (jedno źródło: expression_summary — 1:1 z terminalem)
                esum = _expression_summary_cached(str(matrix_path), matrix_path.stat().st_mtime)
                if "error" not in esum:
                    d = esum["distribution"]
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Geny × próbki", f'{esum["n_genes"]:,} × {esum["n_samples"]}')
                    m2.metric("Metryka (wykryta)", d["metric"])
                    m3.metric("Mediana", f'{d["median"]:.2f}')
                    m4.metric("Wartości zerowe", f'{d["zero_pct"]:.1f}%')
                    st.caption(
                        f"IQR: {d['p25']:.2f}–{d['p75']:.2f}  ·  max: {d['max']:,.0f}  ·  "
                        f"mediana głębokości: {d['median_depth']:,.0f}  ·  "
                        + ("filtr biotypów zastosowany" if esum["biotype_filtered"]
                           else "bez filtra biotypów")
                    )
                    _dash_explain(co="Podstawowe parametry macierzy cech: wymiar (geny × próbki), wykrytą metrykę normalizacji, medianę ekspresji, odsetek wartości zerowych, rozstęp i głębokość sekwencjonowania.")
                    st.divider()

                st.subheader("Rozkład ekspresji (log2 TPM)")
                st.caption("Histogram ekspresji jednej próbki — bimodalność (geny off/on) "
                           "to cecha zdrowego RNA-seq.")
                sample_choice = st.selectbox(
                    "Próbka", options=sample_cols, index=0,
                    help="Wybierz próbkę (wpisz fragment nazwy, by szybko odfiltrować listę) "
                         "— zobaczysz rozkład ekspresji jej genów i jej dane kliniczne, "
                         "kontrola jakości pojedynczej próbki.",
                )

                # Panel informacji klinicznych o wybranej próbce (jeśli jest w survival_dataset)
                _render_sample_clinical(ds, sample_choice)

                try:
                    fig = viz.histogram_tpm(matrix, sample_choice)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as exc:
                    st.error(f"Błąd histogramu: {exc}")

                _dash_explain(co="Histogram ekspresji log2(TPM+1) dla jednej wybranej próbki — ile genów ma jaki poziom ekspresji — wraz z danymi klinicznymi tej próbki.")
                st.divider()
                st.subheader("Ekspresja markerów LUAD")
                st.caption("Rozkład ekspresji klasycznych markerów raka płuca po wszystkich próbkach. "
                           "NKX2-1/SFTPC (różnicowanie) zwykle wysokie; ALK/ROS1 (działają przez "
                           "fuzje) zwykle niskie w ekspresji.")
                try:
                    # Ograniczamy liczbę próbek dla wydajności boxplotu
                    sample_subset = sample_cols[:200]
                    fig = viz.markers_expression(matrix, sample_subset)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as exc:
                    st.error(f"Błąd wykresu markerów: {exc}")

                _dash_explain(co="Rozkład ekspresji znanych markerów raka płuca (EGFR, KRAS, TP53, ALK, NKX2-1, SFTPC i in.) po wszystkich próbkach — wykres pudełkowy na gen.")
                if "error" not in esum:
                    st.divider()
                    st.subheader("Batch — ośrodki TSS")
                    st.caption(
                        f"Rozkład próbek per ośrodek (TSS z barkodu TCGA): "
                        f"{esum['parsed_tss']}/{esum['n_tss']} rozpoznanych. "
                        "Nierównowaga = potencjalny efekt batch (istotne dla obrony)."
                    )
                    try:
                        st.plotly_chart(viz.tss_batch_figure(esum["tss_rows"]),
                                        use_container_width=True)
                    except Exception as exc:
                        st.error(f"Błąd wykresu TSS: {exc}")

                    _dash_explain(co="Ile próbek pochodzi z którego ośrodka (TSS — Tissue Source Site, zakodowany w barkodzie TCGA). Słupki posortowane od najliczniejszego ośrodka.")
                    st.divider()
                    st.subheader("PCA — wariancja składowych")
                    st.caption(
                        f"Wariancja wyjaśniona przez PC1–5 (PCA na top {esum['pca_top_n']} "
                        "zmiennych genach, log2 + z-score)."
                    )
                    if esum["pcs"]:
                        try:
                            st.plotly_chart(viz.pca_variance_figure(esum["pcs"]),
                                            use_container_width=True)
                        except Exception as exc:
                            st.error(f"Błąd wykresu PCA: {exc}")
                    else:
                        st.info("PCA nie policzona dla tej macierzy.")

                    _dash_explain(co="Ile zmienności w danych tłumaczy każda główna składowa (PC1–5). PCA liczona na najbardziej zmiennych genach, po log2 i standaryzacji (z-score).")

    # ===== ZAKŁADKA GOTOWOŚĆ ML =====
    with tab_ml:
        st.markdown(
            "**Gotowość danych pod uczenie maszynowe.** Metryki oceniające, czy ta macierz "
            "ekspresji i etykiety nadają się jako materiał do modeli (docelowo multimodalnych). "
            "Kontrolki na skróty: zielona = OK, bursztynowa = wymaga uwagi, "
            "czerwona = wymaga działania."
        )
        with st.expander("🎯 Co znaczy „dane gotowe pod ML”"):
            st.markdown("Dane nadają się do uczenia maszynowego, gdy: **1)** target jest dobrze "
                        "zdefiniowany i niesie sygnał, **2)** cechy korelują z targetem ponad "
                        "przypadek, **3)** cechy wnoszą wartość ponad kliniczny baseline, **4)** "
                        "znany jest realistyczny sufit czułości, **5)** cechy są czyste i "
                        "porównywalne, **6)** nie ma wycieku technicznego (batch). Metryki poniżej "
                        "sprawdzają te warunki liczbowo.")
        _ml_matrix_path = DATA_PROCESSED / "expression_matrix.parquet"
        _esum_ml = (
            _expression_summary_cached(str(_ml_matrix_path), _ml_matrix_path.stat().st_mtime)
            if _ml_matrix_path.exists() else None
        )
        try:
            _report = readiness.ml_readiness_report(ds, _esum_ml)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Błąd oceny gotowości ML: {exc}")
            _report = None
        if _report:
            _summ = _report["summary"]
            _vc = _ML_PASTEL[_summ["verdict_status"]]
            _bg = _ML_PASTEL["green"]["bar"]
            _by = _ML_PASTEL["yellow"]["bar"]
            _br = _ML_PASTEL["red"]["bar"]
            st.markdown(
                f'<div style="border-left:6px solid {_vc["bar"]};background:{_vc["bg"]};'
                f'padding:14px 18px;border-radius:8px;margin:2px 0 12px 0;">'
                f'<div style="font-size:1.05em;font-weight:650;">'
                f'{_html_esc(_summ["verdict"])}</div>'
                f'<div style="font-size:0.85em;opacity:0.78;margin-top:6px;">Kontrolki: '
                f'<b style="color:{_bg};">\u25CF {_summ["green"]}</b> OK &nbsp;\u00B7&nbsp; '
                f'<b style="color:{_by};">\u25CF {_summ["yellow"]}</b> uwaga &nbsp;\u00B7&nbsp; '
                f'<b style="color:{_br};">\u25CF {_summ["red"]}</b> działanie</div></div>',
                unsafe_allow_html=True,
            )
            if _summ["actions"]:
                st.markdown("**Zalecane kroki przygotowania danych pod ML**")
                for _a in _summ["actions"]:
                    st.markdown("- " + _a)
            for _grp in _report["groups"]:
                if not _grp["metrics"]:
                    continue
                st.markdown(f"##### {_grp['title']}")
                st.markdown(
                    "".join(_ml_card_html(_m) for _m in _grp["metrics"]),
                    unsafe_allow_html=True,
                )

        st.divider()
        st.markdown("#### Eksport zbioru gotowego pod ML")
        st.markdown(
            "Z audytu w gotowy materiał: deduplikacja do 1 próbki/pacjenta, podział "
            "train/test **grupowany po pacjencie**, selekcja cech i standaryzacja "
            "**fitowane tylko na train** (bez wycieku do testu). Wynik: X/y + manifest w ZIP-ie."
        )
        _n_genes = len([c for c in ds.columns if c.startswith("ENSG")])
        _c1, _c2, _c3 = st.columns(3)
        _topk = _c1.number_input(
            "Geny (top-K wg wariancji)", min_value=10, max_value=max(_n_genes, 10),
            value=min(2000, _n_genes), step=100,
            help="Ile najbardziej zmiennych genów zostawić jako cechy (selekcja fit na train).",
        )
        _testf = _c2.slider("Frakcja testu", 0.10, 0.40, 0.20, 0.05)
        _seed = _c3.number_input("Seed", min_value=0, max_value=99999, value=42, step=1)
        _c4, _c5 = st.columns(2)
        _dedup = _c4.checkbox("Deduplikuj do 1 próbki/pacjenta", value=True)
        _stdz = _c5.checkbox("Standaryzuj (z-score, fit na train)", value=True)
        if st.button("Przygotuj zbiór ML", type="primary"):
            with st.spinner("Buduję zbiór gotowy pod ML…"):
                try:
                    _res = ml_export.prepare_ml_dataset(
                        ds, top_k=int(_topk), test_frac=float(_testf), seed=int(_seed),
                        dedup=bool(_dedup), standardize=bool(_stdz),
                    )
                    st.session_state["_ml_export"] = (
                        ml_export.build_ml_bundle(_res), _res["manifest"],
                    )
                except Exception as exc:  # noqa: BLE001
                    st.session_state.pop("_ml_export", None)
                    st.error(f"Nie udało się przygotować zbioru: {exc}")
        if "_ml_export" in st.session_state:
            _zip_bytes, _mf = st.session_state["_ml_export"]
            _pz, _sc = _mf["podzial"], _mf["selekcja_cech"]
            st.success(
                f"Gotowe: train {_pz['n_train']} · test {_pz['n_test']} próbek · "
                f"{_sc['wybrane_top_k']} genów · wspólni pacjenci train/test: "
                f"{_pz['pacjenci_wspolni_train_test']}."
            )
            st.download_button(
                "⬇ Pobierz zbiór ML (ZIP)", _zip_bytes,
                file_name="luad_ml_dataset.zip", mime="application/zip",
            )
            with st.expander("Podgląd manifestu"):
                st.json(_mf)


def _detect_file_type(content: bytes) -> str:
    """Rozpoznaje typ wgranego pliku po sygnaturze nagłówka.

    Zwraca: 'star', 'sample_sheet', 'clinical' lub 'unknown'.
    Sprawdza charakterystyczne kolumny/strukturę każdego formatu.
    """
    try:
        # Pierwsze ~4KB wystarczy na nagłówek + kilka wierszy
        head = content[:4096].decode("utf-8", errors="ignore")
    except Exception:
        return "unknown"

    lines = head.splitlines()
    if not lines:
        return "unknown"

    # STAR-Counts: komentarz GENCODE LUB gene_id+tpm_unstranded LUB wiersz meta
    if head.startswith("#") and "gene-model" in head.lower():
        return "star"
    first_lines = "\n".join(lines[:8])
    if "N_unmapped" in first_lines or "N_multimapping" in first_lines:
        return "star"
    # nagłówek STAR (po ewentualnym komentarzu)
    header = lines[1] if lines[0].startswith("#") and len(lines) > 1 else lines[0]
    header_cols = header.split("\t")
    if "gene_id" in header_cols and "tpm_unstranded" in header_cols:
        return "star"

    # Clinical: kolumny z kropkami (cases.submitter_id, demographic.vital_status)
    if "cases.submitter_id" in header_cols or "demographic.vital_status" in header_cols:
        return "clinical"
    if "diagnoses.ajcc_pathologic_stage" in header_cols:
        return "clinical"

    # Sample sheet: kolumny GDC ze spacjami (File Name, Sample ID, Case ID)
    sheet_markers = {"File ID", "File Name", "Case ID", "Sample ID"}
    if len(sheet_markers & set(header_cols)) >= 3:
        return "sample_sheet"

    return "unknown"


# Czytelne nazwy typów do komunikatów
_TYPE_LABELS = {
    "star": "plik STAR-Counts",
    "sample_sheet": "arkusz próbek (sample sheet)",
    "clinical": "dane kliniczne (clinical)",
    "unknown": "nierozpoznany format",
}


def _validate_upload(content: bytes, expected: str, slot_label: str) -> tuple[bool, str]:
    """Sprawdza czy wgrany plik pasuje do oczekiwanego slotu.

    Zwraca (czy_ok, komunikat). Przy niezgodności podpowiada właściwy slot.
    """
    detected = _detect_file_type(content)
    if detected == expected:
        return True, ""

    detected_label = _TYPE_LABELS.get(detected, "nierozpoznany format")
    if detected == "unknown":
        return False, (
            f"Ten plik ma nierozpoznaną strukturę — nie wygląda na {slot_label}. "
            f"Sprawdź, czy wgrywasz właściwy plik (oczekiwany nagłówek pliku {slot_label})."
        )
    # Plik pasuje do innego znanego typu - podpowiedz gdzie go wgrać
    slot_hints = {
        "star": "uploadera „Pliki STAR-Counts” poniżej",
        "sample_sheet": "uploadera „Arkusz próbek”",
        "clinical": "uploadera „Dane kliniczne”",
    }
    hint = slot_hints.get(detected, "")
    return False, (
        f"Ten plik wygląda na {detected_label}, a nie na {slot_label}. "
        f"Użyj {hint}." if hint else
        f"Ten plik wygląda na {detected_label}, a nie na {slot_label}."
    )


def _extract_star_from_zip(zip_bytes: bytes, target_dir: Path) -> dict:
    """Rozpakowuje ZIP, rekurencyjnie znajduje pliki STAR i kopiuje do target_dir.

    Obsługuje dowolną strukturę archiwum (pliki luzem, w podfolderach,
    zagnieżdżone głęboko) - szuka wzorca na każdej głębokości. Każdy
    znaleziony plik waliduje (czy faktycznie STAR-Counts). Duplikaty nazw
    pomija z ostrzeżeniem. Chroni przed path traversal.

    Zwraca słownik: saved, rejected, duplicates, errors (listy/liczby).
    """
    import io
    import zipfile

    result = {"saved": 0, "rejected": [], "duplicates": [], "errors": [], "total_matched": 0}
    target_dir.mkdir(parents=True, exist_ok=True)
    seen_names: set[str] = set()
    # Nazwy plików już obecnych na dysku (kolizje z poprzednim importem)
    existing = {p.name for p in target_dir.glob("*.tsv")}

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        result["errors"].append("Plik nie jest poprawnym archiwum ZIP.")
        return result

    with zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            # Nazwa pliku bez ścieżki (rekurencja niezależna od struktury)
            base = Path(info.filename).name
            # Wzorzec STAR-Counts na dowolnej głębokości
            if not base.endswith(".rna_seq.augmented_star_gene_counts.tsv"):
                continue
            result["total_matched"] += 1

            # Duplikat w obrębie archiwum lub względem dysku
            if base in seen_names or base in existing:
                result["duplicates"].append(base)
                continue

            # Strumieniowe rozpakowanie na dysk (bez ładowania całego pliku do RAM).
            # Walidacja zawartości na podstawie nagłówka (pierwszy blok), potem
            # kopiowanie reszty blokami - bezpieczne dla dużych archiwów.
            out_path = target_dir / base
            CHUNK = 1 << 20  # 1 MB
            try:
                with zf.open(info) as src:
                    header = src.read(4096)
                    # Walidacja: czy faktycznie STAR (nie tylko nazwa)
                    if _detect_file_type(header) != "star":
                        result["rejected"].append(base)
                        continue
                    # Zapis: najpierw nagłówek, potem reszta strumieniowo
                    with open(out_path, "wb") as dst:
                        dst.write(header)
                        while True:
                            chunk = src.read(CHUNK)
                            if not chunk:
                                break
                            dst.write(chunk)
                seen_names.add(base)
                result["saved"] += 1
            except Exception as exc:
                result["errors"].append(f"{base}: błąd ({exc})")
                # sprzątanie częściowo zapisanego pliku
                if out_path.exists():
                    try:
                        out_path.unlink()
                    except Exception:
                        pass

    return result


def render_download(state: dict) -> None:
    """Pobieranie danych z GDC API (rozsądne podzbiory; pełna kohorta przez CLI)."""
    st.header("Pobieranie z GDC")
    st.caption("Pobierz dane TCGA bezpośrednio z Genomic Data Commons przez API. "
               "Sekcja jest przeznaczona do pobierania rozsądnych podzbiorów "
               "(np. do testów). Dla pełnej kohorty zobacz uwagę na dole.")

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # Feedback z poprzedniego pobierania (przetrwa rerun)
    if st.session_state.get("download_feedback"):
        for msg in st.session_state.download_feedback:
            st.success(msg)
        st.session_state.download_feedback = []

    col1, col2 = st.columns(2)
    with col1:
        project = st.text_input("Projekt GDC", value="TCGA-LUAD",
                                help="Identyfikator projektu, np. TCGA-LUAD, TCGA-BRCA.")
    with col2:
        workflow = st.text_input("Workflow", value="STAR - Counts",
                                 help="Typ workflow GDC. Domyślnie STAR - Counts.")

    st.divider()

    # --- Krok 1: sprawdzenie dostępności (samo metadane, szybkie) ---
    st.subheader("1. Sprawdź dostępność")
    st.caption("Zapytanie o metadane (bez pobierania plików) — ile plików i jaki rozmiar.")
    if st.button("Sprawdź dostępność w GDC", key="btn_check_gdc"):
        with st.spinner("Zapytanie do GDC API..."):
            try:
                filt = build_files_filter(project_id=project, workflow_type=workflow)
                response = query_files(filters=filt, size=10000)
                meta = parse_files_response(response)
                total = response.get("data", {}).get("pagination", {}).get("total", 0)
                total_mb = meta["file_size"].sum() / 1024**2 if "file_size" in meta.columns else 0
                st.session_state.gdc_check = {
                    "n_files": meta.height, "total": total, "total_mb": total_mb,
                    "project": project, "workflow": workflow,
                }
            except GDCClientError as exc:
                st.session_state.gdc_check = {"error": str(exc)}
        st.rerun()

    check = st.session_state.get("gdc_check")
    if check:
        if check.get("error"):
            st.error(f"Błąd zapytania do GDC: {check['error']}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Plików dostępnych", check["total"])
            c2.metric("W odpowiedzi", check["n_files"])
            c3.metric("Rozmiar (MB)", f"{check['total_mb']:.0f}")

    st.divider()

    # --- Krok 2: pobieranie ---
    st.subheader("2. Pobierz dane")

    n_default = 10
    max_files = check["n_files"] if check and not check.get("error") else 601
    n_files = st.number_input(
        "Liczba plików STAR do pobrania", min_value=1, max_value=max(max_files, 1),
        value=min(n_default, max(max_files, 1)), step=1,
        help="Domyślnie mała liczba (do szybkiego testu). Możesz pobrać całą dostępną kohortę.",
    )
    colA, colB = st.columns(2)
    with colA:
        get_clinical = st.checkbox("Pobierz dane kliniczne (clinical.tsv)", value=True)
    with colB:
        get_sheet = st.checkbox("Zapisz arkusz próbek (sample sheet)", value=True)

    if st.button("Pobierz z GDC", key="btn_download_gdc", type="primary"):
        progress = st.progress(0.0, text="Przygotowanie...")
        status = st.empty()
        msg = []
        try:
            # Metadane
            status.write("Zapytanie o metadane plików...")
            filt = build_files_filter(project_id=project, workflow_type=workflow)
            response = query_files(filters=filt, size=int(n_files))
            files_metadata = parse_files_response(response).head(int(n_files))
            msg.append(f"Metadane: {files_metadata.height} plików")

            # Sample sheet
            if get_sheet:
                from src.cli import _write_sample_sheet, _write_metadata_cart
                sheet_path = DATA_RAW / "gdc_sample_sheet.tsv"
                _write_sample_sheet(files_metadata, sheet_path)
                _write_metadata_cart(response, DATA_RAW / "metadata.cart.json")
                msg.append(f"Zapisano sample sheet: {sheet_path.name}")

            # Clinical
            if get_clinical:
                status.write("Pobieranie danych klinicznych (/cases)...")
                try:
                    resp_cases = query_cases(size=10000)
                    cases_df = parse_cases_response(resp_cases)
                    clinical_path = DATA_RAW / "clinical.tsv"
                    cases_df.write_csv(clinical_path, separator="\t", quote_style="never")
                    msg.append(f"Zapisano clinical.tsv ({cases_df.height} wierszy)")
                except CasesClientError as exc:
                    msg.append(f"Błąd danych klinicznych: {exc}")

            # Pliki STAR z paskiem postępu
            star_dir = DATA_RAW / "uploaded_star"

            def on_progress(idx, total, name):
                progress.progress(idx / total, text=f"Pobieranie {idx}/{total}: {name}")

            status.write(f"Pobieranie {files_metadata.height} plików STAR...")
            report = download_files(
                metadata=files_metadata, output_dir=star_dir,
                show_progress=False, progress_callback=on_progress,
            )
            n_ok = report.filter(pl.col("verified")).height
            n_fail = report.filter(~pl.col("verified")).height
            msg.append(f"Pobrano {n_ok}/{report.height} plików STAR (zweryfikowanych MD5)")
            if n_fail > 0:
                msg.append(f"Błędy: {n_fail} plików nie zweryfikowanych")

            progress.progress(1.0, text="Gotowe")
            status.empty()
        except GDCClientError as exc:
            status.empty()
            st.error(f"Błąd pobierania z GDC: {exc}")
            return
        except Exception as exc:
            status.empty()
            st.error(f"Nieoczekiwany błąd: {exc}")
            return

        st.session_state.download_feedback = msg
        st.rerun()


def render_upload(state: dict) -> None:
    """Ręczne wgrywanie plików do data/raw/ (alternatywa dla Pobierania)."""
    st.header("Wgrywanie plików")
    st.caption("Wgraj ręcznie pliki kohorty zamiast pobierać je z GDC. "
               "Pliki trafiają do `data/raw/`.")

    st.markdown(
        "Pipeline potrzebuje trzech rodzajów plików:\n"
        "- **clinical.tsv** — dane kliniczne (czas obserwacji, status, kowarianty)\n"
        "- **gdc_sample_sheet** — arkusz próbek (mapowanie plik → próbka → pacjent)\n"
        "- **pliki STAR-Counts** — surowe zliczenia ekspresji (jeden plik per próbka)"
    )

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # Pokaż feedback z poprzedniego wgrania (przetrwa rerun)
    if st.session_state.get("upload_feedback"):
        for msg in st.session_state.upload_feedback:
            st.success(msg)
        st.session_state.upload_feedback = []

    st.divider()

    # --- Dane kliniczne ---
    st.subheader("Dane kliniczne")
    clinical_file = st.file_uploader("clinical.tsv", type=["tsv", "txt"], key="up_clinical")
    if clinical_file is not None:
        if st.button("Zapisz clinical.tsv", key="btn_clinical"):
            content = clinical_file.getvalue()
            ok, msg = _validate_upload(content, "clinical", "dane kliniczne (clinical.tsv)")
            if not ok:
                st.error(msg)
                return
            target = DATA_RAW / "clinical.tsv"
            try:
                target.write_bytes(content)
                st.session_state.upload_feedback = [f"Zapisano: {target.name} ({clinical_file.size} B)"]
            except Exception as exc:
                st.error(f"Błąd zapisu: {exc}")
                return
            st.rerun()

    # --- Sample sheet ---
    st.subheader("Arkusz próbek (sample sheet)")
    sheet_file = st.file_uploader("gdc_sample_sheet*.tsv", type=["tsv", "txt"], key="up_sheet")
    if sheet_file is not None:
        if st.button("Zapisz sample sheet", key="btn_sheet"):
            content = sheet_file.getvalue()
            ok, msg = _validate_upload(content, "sample_sheet", "arkusz próbek (sample sheet)")
            if not ok:
                st.error(msg)
                return
            # zachowujemy oryginalną nazwę jeśli pasuje do wzorca, inaczej standaryzujemy
            fname = sheet_file.name
            if not fname.startswith("gdc_sample_sheet"):
                fname = "gdc_sample_sheet.tsv"
            target = DATA_RAW / fname
            try:
                target.write_bytes(content)
                st.session_state.upload_feedback = [f"Zapisano: {target.name} ({sheet_file.size} B)"]
            except Exception as exc:
                st.error(f"Błąd zapisu: {exc}")
                return
            st.rerun()

    # --- Pliki STAR (wiele naraz) ---
    st.subheader("Pliki STAR-Counts (archiwum ZIP)")
    st.caption("Wgraj archiwum ZIP zawierające pliki STAR-Counts. Pliki mogą być "
               "luzem lub w dowolnej strukturze podfolderów (np. tak jak pobiera je "
               "GDC — każdy plik w osobnym katalogu). Przeszukiwanie jest rekurencyjne.")
    st.caption("Rozpoznawane są pliki pasujące do wzorca "
               "`<UUID>.rna_seq.augmented_star_gene_counts.tsv` na dowolnej głębokości.")
    st.caption("Limit rozmiaru archiwum: **1 GB**. Dla bardzo dużych kohort — "
               "zwłaszcza w instalacji hostowanej (Streamlit Cloud), gdzie obowiązują "
               "dodatkowe limity zasobów serwera — rozważ pobranie danych bezpośrednio "
               "narzędziem [GDC Data Transfer Tool]"
               "(https://gdc.cancer.gov/access-data/gdc-data-transfer-tool) "
               "lub komendą CLI `download`.")
    zip_file = st.file_uploader("Archiwum .zip z plikami STAR", type=["zip"], key="up_star_zip")
    if zip_file is not None:
        st.write(f"Archiwum: **{zip_file.name}** ({zip_file.size / 1024 / 1024:.1f} MB)")
        if st.button("Rozpakuj i importuj pliki STAR", key="btn_star_zip"):
            star_dir = DATA_RAW / "uploaded_star"
            with st.spinner("Rozpakowywanie archiwum i wyszukiwanie plików STAR..."):
                res = _extract_star_from_zip(zip_file.getvalue(), star_dir)

            msg = []
            if res["total_matched"] == 0 and not res["errors"]:
                msg.append("W archiwum nie znaleziono żadnych plików pasujących do wzorca "
                           "STAR-Counts. Sprawdź, czy ZIP zawiera pliki "
                           "`*.rna_seq.augmented_star_gene_counts.tsv`.")
            if res["saved"]:
                msg.append(f"Zaimportowano {res['saved']} plików STAR do `{star_dir}`")
            if res["duplicates"]:
                preview = ", ".join(res["duplicates"][:3]) + ("..." if len(res["duplicates"]) > 3 else "")
                msg.append(f"Pominięto {len(res['duplicates'])} duplikatów "
                           f"(plik o tej nazwie już istnieje): {preview}")
            if res["rejected"]:
                preview = ", ".join(res["rejected"][:3]) + ("..." if len(res["rejected"]) > 3 else "")
                msg.append(f"Odrzucono {len(res['rejected'])} plików — pasują nazwą, ale "
                           f"ich zawartość nie jest formatem STAR-Counts: {preview}")
            if res["errors"]:
                preview = "; ".join(res["errors"][:2]) + ("..." if len(res["errors"]) > 2 else "")
                msg.append(f"Błędy: {len(res['errors'])} ({preview})")
            if not msg:
                msg = ["Nie zaimportowano żadnych plików."]
            st.session_state.upload_feedback = msg
            st.rerun()

    st.divider()

    # --- Status gotowości ---
    st.subheader("Status kohorty")
    c1, c2, c3 = st.columns(3)
    c1.metric("clinical.tsv", "✓" if state["has_clinical"] else "—")
    c2.metric("sample sheet", "✓" if state["has_sample_sheet"] else "—")
    star_count = len(_discover_star_files(DATA_RAW)) if DATA_RAW.exists() else 0
    c3.metric("Pliki STAR", star_count)

    if state["raw_ready"] and star_count > 0:
        st.success("Kohorta gotowa — możesz przejść do etapu Parsowanie.")
    else:
        missing = []
        if not state["has_clinical"]:
            missing.append("clinical.tsv")
        if not state["has_sample_sheet"]:
            missing.append("sample sheet")
        if star_count == 0:
            missing.append("pliki STAR")
        st.info(f"Brakuje: {', '.join(missing)}. Wgraj pozostałe pliki, by odblokować Parsowanie.")


from src.manage.data_ops import (  # noqa: E402
    MANAGE_SCOPES,
    is_within_data as _is_within_data,
    iter_scope_files as _iter_scope_files,
    scope_stats as _scope_stats,
    fmt_size as _fmt_size,
    delete_scope as _delete_scope,
    build_archive_zip as _build_archive_zip,
)


def render_manage(state: dict) -> None:
    """Zarządzanie danymi: archiwizacja (backup) i bezpieczne kasowanie.

    Cztery niezależne kategorie - pliki STAR (ciężkie) oddzielone od lekkich
    metadanych, by można je archiwizować/kasować osobno.
    """
    st.header("Zarządzanie danymi")
    st.caption("Archiwizuj dane przed skasowaniem (backup) lub wyczyść pliki "
               "pipeline'u, by zacząć od nowa. Pliki STAR (gigabajty) są osobną "
               "kategorią — można je zachować/usunąć niezależnie od reszty.")

    # Statystyki per kategoria
    stats = {k: _scope_stats(v[1], v[2]) for k, v in MANAGE_SCOPES.items()}

    st.subheader("Stan danych")
    cols = st.columns(4)
    for i, (key, (label, path, mode, _desc)) in enumerate(MANAGE_SCOPES.items()):
        n, b = stats[key]
        cols[i].metric(label, f"{n} plików", _fmt_size(b))

    def _fmt_option(k):
        label, _p, _m, _d = MANAGE_SCOPES[k]
        n, b = stats[k]
        return f"{label} ({n} plików, {_fmt_size(b)})"

    st.divider()

    # =================================================================
    #  ARCHIWIZACJA
    # =================================================================
    st.subheader("Archiwizacja (backup)")
    st.caption("Wybierz, co spakować do archiwum ZIP. Przydatne, by nie pobierać "
               "ponownie gigabajtów surowych danych — można zarchiwizować same "
               "pliki STAR, same wyniki, albo dowolną kombinację.")

    arch_choices = st.multiselect(
        "Co zarchiwizować", options=list(MANAGE_SCOPES.keys()),
        default=["processed"], format_func=_fmt_option, key="arch_scope",
    )

    sel_bytes = sum(stats[k][1] for k in arch_choices)
    if sel_bytes > 300 * 1024**2:
        st.warning(
            f"Wybrane dane mają {_fmt_size(sel_bytes)}. Pakowanie odbywa się "
            f"w pamięci — w instalacji hostowanej (Streamlit Cloud) z ograniczonym "
            f"RAM może się nie powieść. Lokalnie powinno zadziałać. Dla bardzo dużych "
            f"zbiorów rozważ kopię katalogu `data/` ręcznie."
        )

    if arch_choices:
        if st.button("Przygotuj archiwum ZIP", key="btn_archive"):
            total_files = sum(stats[k][0] for k in arch_choices)
            if total_files == 0:
                st.info("Wybrane kategorie są puste — nie ma czego archiwizować.")
            else:
                targets = [(MANAGE_SCOPES[k][0].replace(" ", "_"),
                            MANAGE_SCOPES[k][1], MANAGE_SCOPES[k][2]) for k in arch_choices]
                progress = st.progress(0.0, text="Przygotowanie archiwum...")

                def _on_archive(idx, total, name):
                    progress.progress(idx / total if total else 1.0,
                                      text=f"Pakowanie {idx}/{total}: {name}")
                try:
                    data = _build_archive_zip(targets, progress_callback=_on_archive)
                    st.session_state.archive_data = data
                    st.session_state.archive_size = len(data)
                    progress.progress(1.0, text="Gotowe")
                except MemoryError:
                    st.error("Zabrakło pamięci przy pakowaniu. Wybierz mniej danych "
                             "lub zarchiwizuj lokalnie.")
                except Exception as exc:
                    st.error(f"Błąd pakowania: {exc}")

    if st.session_state.get("archive_data"):
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.success(f"Archiwum gotowe ({_fmt_size(st.session_state.archive_size)}). "
                   f"Kliknij, by pobrać.")
        st.download_button(
            "Pobierz archiwum ZIP", data=st.session_state.archive_data,
            file_name=f"luad-huba-backup_{ts}.zip", mime="application/zip", key="dl_archive",
        )

    st.divider()

    # =================================================================
    #  KASOWANIE
    # =================================================================
    st.subheader("Kasowanie danych")
    st.error("**Uwaga: kasowanie jest nieodwracalne.** Zarchiwizuj dane powyżej, "
             "jeśli chcesz zachować kopię. Operacja usuwa pliki trwale z dysku.")

    del_choices = st.multiselect(
        "Co skasować", options=list(MANAGE_SCOPES.keys()),
        default=[], format_func=_fmt_option, key="del_scope",
    )

    if del_choices:
        total_files = sum(stats[k][0] for k in del_choices)
        total_bytes = sum(stats[k][1] for k in del_choices)

        st.markdown("**Zostanie trwale usunięte:**")
        for k in del_choices:
            label, path, mode, desc = MANAGE_SCOPES[k]
            n, b = stats[k]
            st.markdown(f"- **{label}** — {n} plików, {_fmt_size(b)}  \n  "
                        f"<span style='color:gray'>{desc}</span>", unsafe_allow_html=True)
        st.markdown(f"**Razem: {total_files} plików, {_fmt_size(total_bytes)}**")

        if total_files == 0:
            st.info("Wybrane kategorie są już puste — nie ma czego kasować.")
        else:
            st.caption("Aby potwierdzić, wpisz **USUŃ** w polu poniżej i kliknij przycisk.")
            confirm_text = st.text_input("Potwierdzenie", key="del_confirm",
                                         placeholder="wpisz: USUŃ")
            confirm_ok = confirm_text.strip().upper() == "USUŃ"

            if st.button("Skasuj wybrane dane", key="btn_delete",
                         type="primary", disabled=not confirm_ok):
                # Pasek postępu - przy tysiącach plików/GB operacja trwa
                progress = st.progress(0.0, text="Przygotowanie...")
                results = []
                # Łączna liczba plików do usunięcia (dla paska)
                grand_total = sum(stats[k][0] for k in del_choices)
                done = 0
                for k in del_choices:
                    label, path, mode, _ = MANAGE_SCOPES[k]

                    def _on_delete(idx, total, name, _label=label):
                        nonlocal done
                        done += 1
                        frac = done / grand_total if grand_total else 1.0
                        progress.progress(min(frac, 1.0),
                                          text=f"Usuwanie {_label}: {idx}/{total}")

                    deleted, errors = _delete_scope(path, mode, progress_callback=_on_delete)
                    results.append(f"{label}: usunięto {deleted} plików")
                    for e in errors:
                        results.append(f"  ⚠️ {e}")
                progress.progress(1.0, text="Gotowe")
                st.session_state.delete_feedback = results
                st.rerun()

            if not confirm_ok and confirm_text:
                st.caption("Wpisane słowo nie pasuje — przycisk pozostaje zablokowany.")

    if st.session_state.get("delete_feedback"):
        st.success("Kasowanie zakończone:")
        for msg in st.session_state.delete_feedback:
            st.write(msg)
        st.session_state.delete_feedback = []


def render_placeholder(stage: dict, state: dict) -> None:
    """Fallback dla ewentualnych sekcji bez dedykowanego widoku (obecnie nieużywany —
    wszystkie etapy mają własne funkcje render)."""
    st.header(f"{stage['label']}")
    st.info(f"Sekcja **{stage['label']}** nie ma jeszcze dedykowanego widoku.")


# =====================================================================
#  GŁÓWNA APLIKACJA
# =====================================================================
def main() -> None:
    st.set_page_config(
        page_title="LUAD-HUBA",
        page_icon="🧬",
        layout="wide",
    )

    state = detect_state()

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("🧬 LUAD-HUBA")
        st.caption("Pipeline ETL — TCGA-LUAD")
        st.divider()

        # Status kohorty (skrót)
        st.markdown("**Status danych**")
        c1, c2 = st.columns(2)
        c1.metric("Pliki STAR", state["parsed_count"], help="Sparsowane parquety w data/interim/star_counts")
        c2.metric("Macierz", "✓" if state["matrix_built"] else "—")
        raw_ok = "✓" if state["raw_ready"] else "○"
        surv_ok = "✓" if state["survival_built"] else "○"
        st.caption(f"{raw_ok} Dane surowe  |  {surv_ok} Przeżywalność")
        st.divider()

        # Nawigacja — etapy z kłódkami
        st.markdown("**Etapy pipeline'u**")
        # Inicjalizacja wybranej sekcji
        if "active_stage" not in st.session_state:
            st.session_state.active_stage = "browse"

        for stage in STAGES:
            unlocked = stage_unlocked(stage, state)
            label = stage["label"]
            if unlocked:
                if st.button(label, key=f"nav_{stage['id']}", use_container_width=True):
                    st.session_state.active_stage = stage["id"]
            else:
                st.button(
                    f"🔒 {label}",
                    key=f"nav_{stage['id']}",
                    use_container_width=True,
                    disabled=True,
                    help=f"Wymaga ukończenia wcześniejszego etapu.",
                )

    # --- GŁÓWNY PANEL ---
    active_id = st.session_state.active_stage
    active_stage = next(s for s in STAGES if s["id"] == active_id)

    # Bezpieczeństwo: jeśli aktywna sekcja została zablokowana, wróć do Browse
    if not stage_unlocked(active_stage, state):
        st.session_state.active_stage = "browse"
        active_stage = next(s for s in STAGES if s["id"] == "browse")

    # Routing do sekcji
    if active_stage["id"] == "browse":
        render_browse(state)
    elif active_stage["id"] == "config":
        render_config(state)
    elif active_stage["id"] == "parse":
        render_parse(state)
    elif active_stage["id"] == "build_matrix":
        render_build_matrix(state)
    elif active_stage["id"] == "build_survival":
        render_build_survival(state)
    elif active_stage["id"] == "validate":
        render_validate(state)
    elif active_stage["id"] == "dashboard":
        render_dashboard(state)
    elif active_stage["id"] == "upload":
        render_upload(state)
    elif active_stage["id"] == "download":
        render_download(state)
    elif active_stage["id"] == "manage":
        render_manage(state)
    else:
        render_placeholder(active_stage, state)


if __name__ == "__main__":
    main()
