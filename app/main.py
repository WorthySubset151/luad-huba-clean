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
    method = st.selectbox(
        "Metryka (normalization.method)",
        options=["tpm", "log2cpm"],
        index=["tpm", "log2cpm"].index(norm.get("method", "tpm"))
        if norm.get("method") in ("tpm", "log2cpm") else 0,
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


def render_placeholder(stage: dict, state: dict) -> None:
    """Tymczasowa treść dla sekcji jeszcze niezaimplementowanych."""
    st.header(f"{stage['label']}")
    st.info(
        f"Sekcja **{stage['label']}** zostanie zaimplementowana w kolejnej sesji.\n\n"
        "Sesja 1 obejmuje fundament (nawigacja, stan, wymuszanie kolejności) "
        "oraz sekcje Przeglądanie i Konfiguracja."
    )
    # Krótki opis co tu będzie
    descriptions = {
        "download": "Pobranie danych TCGA-LUAD z GDC API (manifest, pliki STAR, clinical).",
        "upload": "Ręczne wgranie plików STAR-Counts i clinical.tsv.",
        "parse": "Parsowanie plików STAR-Counts (TSV) do formatu parquet.",
        "validate": "Kontrola jakości kohorty (QC) i raport.",
        "build_matrix": "Budowa macierzy ekspresji (geny × próbki) z filtrem biotype.",
        "build_survival": "Budowa zbioru do analizy przeżywalności (z filtrami warunkowymi).",
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
    else:
        render_placeholder(active_stage, state)


if __name__ == "__main__":
    main()
