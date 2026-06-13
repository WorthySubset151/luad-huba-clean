"""LUAD-HUBA — interfejs graficzny pipeline'u (Streamlit).

Jeden plik, ręczny sidebar, wymuszanie kolejności etapów pipeline'u.
Uruchomienie: uv run streamlit run app/main.py
"""

__author__ = "Łukasz Połaski"

import sys
from pathlib import Path

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

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim" / "star_counts"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

# --- Definicja etapów pipeline'u (kolejność + zależności) ---
# klucz: id etapu, label: nazwa w sidebarze, requires: id etapu wymaganego wcześniej
STAGES = [
    {"id": "download", "label": "Pobieranie", "requires": None, "always": False},
    {"id": "upload", "label": "Wgrywanie", "requires": None, "always": True},
    {"id": "browse", "label": "Przeglądanie", "requires": None, "always": True},
    {"id": "parse", "label": "Parsowanie", "requires": "raw_ready", "always": False},
    {"id": "validate", "label": "Walidacja", "requires": "parsed", "always": False},
    {"id": "build_matrix", "label": "Macierz ekspresji", "requires": "parsed", "always": False},
    {"id": "build_survival", "label": "Zbiór przeżywalności", "requires": "matrix_built", "always": False},
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
def render_browse(state: dict) -> None:
    """Podgląd plików w data/ (raw, interim, processed)."""
    st.header("Przeglądanie danych")
    st.caption("Przeglądaj pliki pipeline'u: surowe dane, parquety, wyniki.")

    dir_choice = st.selectbox(
        "Katalog",
        options=[
            ("data/raw", DATA_RAW),
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

    st.write(f"Plików: **{len(files)}**")

    file_choice = st.selectbox(
        "Plik do podglądu",
        options=files,
        format_func=lambda f: f"{f.name}  ({f.stat().st_size / 1024:.1f} KB)",
    )

    suffix = file_choice.suffix.lower()
    if suffix == ".parquet":
        try:
            df = pl.read_parquet(file_choice)
            st.write(f"Wymiary: **{df.height} wierszy × {df.width} kolumn**")
            st.dataframe(df.head(20).to_pandas(), use_container_width=True)
            with st.expander("Nazwy kolumn"):
                st.write(list(df.columns))
        except Exception as exc:
            st.error(f"Nie udało się wczytać parquet: {exc}")
    elif suffix in (".tsv", ".csv", ".txt"):
        sep = "\t" if suffix == ".tsv" else ","
        try:
            df = pl.read_csv(file_choice, separator=sep, infer_schema_length=1000)
            st.write(f"Wymiary: **{df.height} wierszy × {df.width} kolumn**")
            st.dataframe(df.head(20).to_pandas(), use_container_width=True)
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
        help="Z konfiguracji: " + str(default_metric) + ". tpm/fpkm znormalizowane, counts/unstranded surowe.",
    )
    dup_strategy = col2.selectbox(
        "Strategia duplikatów",
        options=["deepest", "first", "fail"],
        index=0,
        help="deepest = wybierz aliquot z największą sumą ekspresji (zalecane dla TCGA).",
    )
    biotype = st.text_input(
        "Filtr biotype",
        value=default_biotype or "",
        help="np. protein_coding (zatrzymuje ~33% genów). Puste = wszystkie geny.",
    )

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
    if sheet_path is None or not clinical_path.exists():
        st.error("Brak sample sheet lub `clinical.tsv` w `data/raw/`.")
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


def render_placeholder(stage: dict, state: dict) -> None:
    """Tymczasowa treść dla sekcji jeszcze niezaimplementowanych."""
    st.header(f"{stage['label']}")
    st.info(
        f"Sekcja **{stage['label']}** zostanie zaimplementowana w kolejnej sesji."
    )
    # Krótki opis co tu będzie
    descriptions = {
        "download": "Pobranie danych TCGA-LUAD z GDC API (manifest, pliki STAR, clinical).",
        "upload": "Ręczne wgranie plików STAR-Counts i clinical.tsv.",
        "validate": "Kontrola jakości kohorty (QC) i raport.",
    }
    if stage["id"] in descriptions:
        st.caption(descriptions[stage["id"]])


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
    else:
        render_placeholder(active_stage, state)


if __name__ == "__main__":
    main()
