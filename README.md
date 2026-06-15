# LUAD-HUBA

**Lung Adenocarcinoma – Hybrid Unified Batch Analyzer**

Samowystarczalny pipeline pobierania, walidacji, transformacji i analizy danych
sekwencjonowania RNA oraz danych klinicznych z projektu TCGA-LUAD (The Cancer
Genome Atlas) dla analizy przeżywalności w gruczolakoraku płuca. Obejmuje
warstwę ETL (CLI), interaktywny interfejs graficzny (Streamlit) z dashboardem
analitycznym oraz zestaw notebooków dokumentujących każdy etap przetwarzania.

## Spis treści

- [Opis](#opis)
- [Architektura](#architektura)
- [Instalacja](#instalacja)
- [Szybki start](#szybki-start)
- [Interfejs graficzny (GUI)](#interfejs-graficzny-gui)
- [Interfejs CLI](#interfejs-cli)
- [Omówienie modułów i funkcji](#omówienie-modułów-i-funkcji)
- [Decyzje metodologiczne](#decyzje-metodologiczne)
- [Konfiguracja](#konfiguracja)
- [Notebooki](#notebooki)
- [Struktura katalogów](#struktura-katalogów)
- [Wyjście pipeline'u](#wyjście-pipelineu)
- [Wymagania](#wymagania)

## Opis

LUAD-HUBA komunikuje się bezpośrednio z REST API portalu Genomic Data Commons
(GDC), pobiera pełną kohortę (pliki STAR-Counts, sample sheet, clinical,
metadata) z weryfikacją sum kontrolnych MD5, przeprowadza kontrolę jakości
na poziomie kohorty, transformuje dane do gotowego do analizy zbioru
*survival dataset* (próbki × geny + kolumny `time`, `event`, kowarianty
kliniczne), a następnie pozwala eksplorować wyniki w interaktywnym dashboardzie
(krzywe Kaplana-Meiera, sygnatury wielogenowe, ekspresja markerów).

Pełny przepływ — od surowych danych GDC po wykresy przeżywalności — jest
odtwarzalny i nie wymaga ręcznego klikania w portalu: `git clone` + `uv sync`
+ komendy CLI dają kompletną lokalną kohortę, a `streamlit run app/main.py`
uruchamia interfejs.

Kohorta referencyjna TCGA-LUAD po pełnym przetworzeniu: **601 plików
STAR-Counts → 590 próbek w macierzy** (po deduplikacji aliquotów) **→ 523 próbki
w zbiorze przeżywalności** (po odfiltrowaniu próbek bez danych klinicznych
i artefaktów czasowych), **19 962 genów kodujących białka**, 188 zdarzeń
(zgonów), 64% cenzurowania.

## Architektura

```
┌─────────────────────────────────────────────────────────┐
│                      WARSTWA ETL (src/)                  │
│                                                          │
│   ingest/  →  validate/  →  transform/  →  export/       │
│   parsery     QC kohorty    macierz +      manifest      │
│   + GDC API                 survival       reproduco-    │
│                             dataset        walności      │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              WARSTWA PREZENTACJI (app/)                  │
│                                                          │
│   main.py            – interfejs Streamlit (8 etapów)    │
│   dashboard_viz.py   – wizualizacje Plotly (KM, EDA)     │
└─────────────────────────────────────────────────────────┘
```

Cztery warstwy ETL, każda z jednoznaczną odpowiedzialnością:

- **ingest** — parsery STAR-Counts, sample sheet, clinical, metadata.cart.json;
  klient REST GDC API (endpointy `/files`, `/cases`, `/data`)
- **validate** — reguły QC na poziomie kohorty (brakujące pliki, osierocone
  pliki, duplikaty próbek, brak danych klinicznych), raport JSON ze stemplem
  czasowym UTC
- **transform** — budowanie macierzy ekspresji (z obsługą duplikatów próbek
  i filtrem biotypów), integracja z kowariantami klinicznymi, filtry survival
- **export** — manifest odtwarzalności (hash treści, lista plików źródłowych,
  metryka, znacznik czasowy)

Warstwa prezentacji (`app/`) udostępnia te same funkcje co CLI przez interfejs
graficzny oraz dodaje dashboard analityczny budowany bezpośrednio na zbiorze
przeżywalności.

## Instalacja

```bash
git clone https://github.com/WorthySubset151/luad-huba-clean.git
cd luad-huba-clean
uv sync
```

Wymaga Pythona 3.12+ i [uv](https://github.com/astral-sh/uv).

## Szybki start

Pełny pipeline od zera do dashboardu:

```bash
# 1. Pobranie kohorty z GDC (jednorazowo, ~2.5 GB dla pełnej TCGA-LUAD)
uv run python -m src.cli download --output-dir data/raw/

# 2. Parsowanie plików STAR-Counts do parquet (jeden plik per próbka)
uv run python -m src.cli parse-star

# 3. Kontrola jakości kohorty (raport JSON w logs/qc/)
uv run python -m src.cli validate-cohort

# 4. Budowa macierzy ekspresji (geny × próbki)
uv run python -m src.cli build-matrix --config configs/default.yaml --duplicate-strategy deepest

# 5. Budowa survival dataset (próbki × geny + clinical)
uv run python -m src.cli build-survival --config configs/default.yaml

# 6. Uruchomienie interfejsu graficznego z dashboardem
uv run streamlit run app/main.py
```

> **Uwaga o uruchamianiu CLI.** Komendy wywołuje się przez
> `uv run python -m src.cli <komenda>`. Choć `pyproject.toml` definiuje
> entry point `luad-huba`, w środowisku deweloperskim najpewniejsze jest
> wywołanie modułowe `python -m src.cli`.

## Interfejs graficzny (GUI)

Interfejs Streamlit (`app/main.py`) udostępnia cały pipeline w formie wizualnej
z wymuszaniem kolejności etapów (każdy etap odblokowuje się dopiero gdy
poprzedni jest gotowy). Uruchomienie:

```bash
uv run streamlit run app/main.py
```

Aplikacja otwiera się na `http://localhost:8501`. Nawigacja odbywa się przez
panel boczny, który pokazuje status każdego etapu (gotowy / zablokowany).

### Etapy GUI

| Etap | Warunek wstępny | Opis |
|------|-----------------|------|
| **Pobieranie** | — | Pobranie danych TCGA-LUAD z GDC API |
| **Wgrywanie** | — | Ręczne wgranie plików STAR-Counts i clinical.tsv |
| **Przeglądanie** | — | Podgląd plików (parquet, TSV, YAML) w repozytorium |
| **Parsowanie** | surowe pliki | Parsowanie STAR-Counts do parquet z paskiem postępu |
| **Walidacja** | sparsowane parquety | Kontrola jakości kohorty z raportem QC |
| **Macierz ekspresji** | sparsowane parquety | Budowa macierzy genów × próbki z paskiem postępu |
| **Zbiór przeżywalności** | macierz | Integracja ekspresji z danymi klinicznymi |
| **Dashboard analityczny** | zbiór przeżywalności | Wizualizacje KM, sygnatury, ekspresja markerów |
| **Konfiguracja** | — | Edycja parametrów pipeline'u (YAML) |

> **Status sekcji Pobieranie i Wgrywanie.** Pełny pipeline jest dostępny
> z poziomu CLI. Sekcje Pobieranie (klient GDC API) i Wgrywanie w GUI są
> przewidziane jako uzupełnienie i mogą być dokończone niezależnie — rdzeń
> analityczny (parsowanie → macierz → survival → dashboard) działa w całości.

### Dashboard analityczny

Dashboard (odblokowany po zbudowaniu zbioru przeżywalności) prezentuje kluczowe
wizualizacje biologiczne w dwóch pod-zakładkach. Wszystkie wykresy są
interaktywne (Plotly: najechanie kursorem, przybliżanie).

**Zakładka Przeżywalność:**

- **Kaplan-Meier — cała kohorta** + statystyki (mediana przeżycia całkowitego,
  przeżycie 1/3/5-letnie)
- **Kaplan-Meier — per stadium** (rozbicie na grupy I-IV) + test log-rank
- **Kaplan-Meier — sygnatura wielogenowa** (panel ekspresyjny a priori 7 genów)
  + test log-rank
- **Kaplan-Meier — pojedynczy gen** z wyborem genu z listy markerów; nad
  wykresem wyświetla się precyzyjna charakterystyka roli genu w LUAD, pod
  wykresem stratyfikacja high/low względem mediany ekspresji + test log-rank

**Zakładka Ekspresja:**

- **Rozkład ekspresji** log2(TPM+1) dla wybranej próbki; po wyborze próbki
  wyświetla się panel z jej danymi klinicznymi (pacjent, stadium, wiek, płeć,
  czas obserwacji, status, typ tkanki) — łączy ekspresję z kontekstem klinicznym
- **Ekspresja markerów LUAD** — rozkład klasycznych markerów raka płuca po
  wszystkich próbkach (wykres pudełkowy)

## Interfejs CLI

Pipeline ma **5 komend CLI**, które wykonują się w określonej kolejności.
Każda komenda ma `--help` z pełną listą opcji.

```bash
uv run python -m src.cli <komenda> [opcje]
```

| Komenda | Funkcja |
|---------|---------|
| `download` | Pobiera kohortę z GDC (pliki STAR, sample sheet, clinical, metadata) |
| `parse-star` | Parsuje pliki STAR-Counts do parquet (jeden plik per próbka) |
| `validate-cohort` | Kontrola spójności kohorty, raport QC w `logs/qc/` |
| `build-matrix` | Buduje macierz ekspresji (geny × próbki) |
| `build-survival` | Buduje zbiór przeżywalności (próbki × geny + clinical) |

### Częste przypadki użycia

```bash
# Tylko pobierz metadane bez plików STAR (np. żeby zobaczyć ilu jest pacjentów)
uv run python -m src.cli download --skip-files

# Pobierz tylko 5 plików dla testów
uv run python -m src.cli download --size 5

# Macierz z metryką TPM i obsługą duplikatów (zalecane dla pełnej kohorty)
uv run python -m src.cli build-matrix --metric tpm --duplicate-strategy deepest

# Survival dataset z własnym progiem follow-up
uv run python -m src.cli build-survival --config my_experiment.yaml

# Walidacja w trybie strict (kod wyjścia 1 przy jakimkolwiek błędzie krytycznym)
uv run python -m src.cli validate-cohort --strict
```

## Omówienie modułów i funkcji

### `src/ingest/` — parsery i klient GDC

**`star_parser.py` — `parse_star_counts(path)`**
Wczytuje pojedynczy plik STAR-Counts (surowy TSV z GDC, format GENCODE v36)
i zwraca oczyszczony DataFrame. Pomija komentarze (`#`), usuwa 4 wiersze meta
STAR (`N_unmapped`, `N_multimapping`, `N_noFeature`, `N_ambiguous`), waliduje
identyfikatory Ensembl, rzutuje kolumny zliczeń na `Int64`. Wejście ~60 666
linii → wyjście 60 660 genów × 9 kolumn.

**`sample_sheet_parser.py` — `parse_sample_sheet(path)`**
Parsuje arkusz próbek GDC (`gdc_sample_sheet*.tsv`), mapując pliki na
identyfikatory próbek (`sample_id`) i pacjentów (`case_id`) oraz typ tkanki.
Dostarcza powiązanie plik → próbka → pacjent używane w budowie macierzy.

**`clinical_parser.py` — `parse_clinical(path)`**
Parsuje dane kliniczne (`clinical.tsv`), wyciągając czas obserwacji i status
przeżycia per pacjent oraz kowarianty (wiek, płeć, stadium AJCC). Stanowi
źródło kolumn `time`, `event` i klinicznych dla zbioru przeżywalności.

**`gdc_client.py`, `cases_client.py`** — klient REST GDC API: budowa zapytań do
endpointów `/files`, `/cases`, `/data`, pobieranie z weryfikacją MD5.

**`file_naming.py`** — obsługa dwóch konwencji nazewniczych plików STAR
(`STAR_FILE_PATTERNS`, `STAR_FILE_SUFFIXES`); wykrywanie i ujednolicanie nazw
plików wyjściowych niezależnie od konwencji wejściowej.

### `src/validate/` — kontrola jakości

**`runner.py` — `run_cohort_qc(sample_sheet, clinical, available_stems)`**
Orkiestruje pełną kontrolę jakości kohorty: uruchamia wszystkie reguły QC
i zwraca raport `QCReport` z listą problemów. Funkcje pomocnicze:
`discover_stems(directory)` skanuje katalog parquetów w poszukiwaniu dostępnych
próbek, `save_qc_report(report, output_dir)` zapisuje raport JSON ze stemplem
czasowym UTC.

**`cohort_checks.py`** — cztery reguły spójności:
- `check_samples_have_star_files` — czy każda próbka z sample sheet ma plik STAR
- `check_orphan_star_files` — czy każdy plik STAR ma dopasowanie w sample sheet
- `check_cases_have_clinical` — czy każdy pacjent ma dane kliniczne
- `check_duplicate_samples` — wykrywanie zduplikowanych próbek (aliquoty)

**`qc_result.py`** — struktury raportu: `QCReport` (kontener problemów
z metodą `summary()`), `QCIssue` (pojedynczy problem: severity, category,
message, context), enumy `Severity` i `QCCategory`.

### `src/transform/` — transformacje

**`expression_matrix.py` — `build_expression_matrix(...)`**
Łączy pliki parquet sparsowanych próbek w jedną macierz ekspresji
(geny × próbki). Z każdej próbki wyciąga wybraną metrykę (domyślnie
`unstranded`, dla analiz zalecane `tpm_unstranded`), opcjonalnie filtruje
po biotypie (`protein_coding`) i obsługuje duplikaty próbek. Parametry:
- `metric` — metryka ekspresji do wyciągnięcia
- `duplicate_strategy` — `fail` (domyślnie, rzuca wyjątek), `deepest` (wybiera
  plik o największej sumie ekspresji), `first` (pierwszy alfabetycznie)
- `biotype_filter` — filtr GENCODE (np. `protein_coding` → 19 962 geny)
- `progress_callback` — opcjonalna funkcja raportująca postęp (używana przez GUI)

Towarzyszące funkcje: `build_manifest`/`save_manifest` (manifest odtwarzalności),
`_deduplicate` (logika strategii duplikatów).

**`survival_dataset.py` — `build_survival_dataset(...)`**
Integruje macierz ekspresji, arkusz próbek i dane kliniczne w jeden zbiór gotowy
do analizy w bibliotece `lifelines` (Kaplan-Meier, model Coxa). Każdy wiersz
to jedna próbka; kolumny: identyfikatory, dane przeżycia (`time` w dniach,
`event`), kowarianty kliniczne, ekspresja genów. Parametry:
- `tumor_only` — zachowuje tylko próbki nowotworowe (domyślnie True)
- `min_follow_up_days` — próg krótkiego czasu obserwacji (patrz
  [Decyzje metodologiczne](#decyzje-metodologiczne))
- `drop_zero_time` — usuwa próbki z `time ≤ 0` (artefakt PHI)

### `src/cli_config.py` — konfiguracja

**`load_config(path)`** wczytuje plik YAML, **`get_nested(config, *keys)`**
pobiera zagnieżdżone wartości z bezpiecznym fallbackiem, **`resolve_metric`**
mapuje aliasy metryk (np. `tpm` → `tpm_unstranded`).

### `app/` — interfejs i wizualizacje

**`main.py`** — aplikacja Streamlit: wykrywanie stanu z dysku (`detect_state`),
wymuszanie kolejności etapów (`stage_unlocked`), funkcje renderujące dla każdej
sekcji (`render_browse`, `render_parse`, `render_build_matrix`,
`render_build_survival`, `render_validate`, `render_dashboard`, `render_config`)
oraz pomocnicze (`_render_qc_report` — raport walidacji z podziałem na problemy
obsługiwane automatycznie vs wymagające uwagi; `_render_sample_clinical` — panel
kliniczny wybranej próbki).

**`dashboard_viz.py`** — wizualizacje Plotly budowane na zbiorze
przeżywalności. Funkcje wykresów: `km_overall` (KM całej kohorty + statystyki),
`km_per_stage` (KM per stadium + log-rank), `km_signature` (KM sygnatury
wielogenowej), `km_single_gene` (KM pojedynczego genu high/low), `histogram_tpm`
(rozkład ekspresji próbki), `markers_expression` (boxplot markerów). Dane
referencyjne: `SIGNATURE_PANEL` (7 genów ze znakami), `LUAD_MARKERS` (markery
diagnostyczne i drivery), `GENE_INFO` (charakterystyki 12 genów),
`collapse_stage` (grupowanie stadiów), `_ensure_time_years` (przeliczenie
`time` z dni na lata).

## Decyzje metodologiczne

Pipeline zawiera kilka świadomych decyzji dotyczących jakości danych, które
mają wpływ na poprawność analizy przeżywalności:

**Filtr biotypów (`protein_coding`).** Do analizy ekspresji i przeżywalności
zachowywane są tylko geny kodujące białka (19 962 z 60 660). Pozostałe biotypy
(lncRNA, pseudogeny, miRNA) niosą inny rodzaj sygnału i rozdmuchują
wymiarowość.

**Obsługa duplikatów (aliquoty).** Niektóre próbki TCGA mają wiele aliquotów
(plików STAR dla tej samej próbki). Strategia `deepest` wybiera aliquot
o największej głębokości sekwencjonowania (sumie ekspresji). W kohorcie
referencyjnej deduplikacja redukuje 601 plików do 590 próbek.

**Ochrona wczesnych zgonów (`min_follow_up_days`).** Próg krótkiego czasu
obserwacji usuwa próbki **cenzurowane** z `time` poniżej progu (krótkie cenzury
to utracone obserwacje bez informacji o przeżyciu). Ale próbki ze **zdarzeniem**
(zgon) i krótkim czasem są **zachowywane** — wczesne zgony to realny sygnał
przeżyciowy, nie szum. Usunięcie ich zawyżyłoby optymizm modelu (forma
immortal time bias).

**Artefakt czasowy PHI (`drop_zero_time`).** Próbki z `time ≤ 0` są usuwane.
W TCGA `time = 0` to artefakt anonimizacji (daty podawane z dokładnością
do miesiąca, zaokrąglane, dają wahanie 0-16 dni wg Liu et al. 2018, TCGA-CDR).
Wartości ≤ 0 są też nieużywalne w modelu Coxa.

**Próbki bez danych klinicznych.** Pacjenci z plikiem RNA-seq, ale bez
kompletnych danych klinicznych (brak czasu obserwacji) są pomijani przy budowie
zbioru przeżywalności. W kohorcie referencyjnej to 9 pacjentów. Walidacja
raportuje ich jawnie jako rozjazd obsługiwany automatycznie (nie wymagający
ręcznej interwencji).

## Konfiguracja

Wszystkie komendy akceptują opcjonalną flagę `--config PATH` do pliku YAML.
Domyślne configi w `configs/`:

- `configs/default.yaml` — parametry pipeline'u (projekt, workflow, metryka
  normalizacji, filtr biotypów, próg `min_follow_up_days`, `drop_zero_time`)
- `configs/filters.yaml` — progi QC

Pierwszeństwo wartości: **flaga CLI > config > hardcoded default**. Bez flagi
`--config` pipeline działa z wartościami domyślnymi (backward compatible).

Parametry można też edytować wizualnie w sekcji **Konfiguracja** w GUI.

## Notebooki

Notebooki w `notebooks/` dokumentują i weryfikują kolejne etapy pipeline'u.
Pełnią rolę testów akceptacyjnych (sanity checks) oraz materiału obronnego —
pokazują biologiczną i metodologiczną poprawność przetwarzania.

- **`01_explore_first_samples.ipynb`** — sanity check biologiczny na 2 próbkach
  (geny housekeeping, markery LUAD, pułapka counts vs TPM)
- **`02_full_cohort_pipeline.ipynb`** — operacyjny sanity check pełnej kohorty
  (funnel 601 → 523, raport QC, deep dive: tumor/normal po deduplikacji,
  outliery, pacjenci z wieloma próbkami)
- **`03_gdc_api_sanity_check.ipynb`** — weryfikacja klienta GDC API
  (porównanie danych z API vs referencyjny clinical.tsv z portalu)
- **`04_end_to_end_demo.ipynb`** — demonstracja pełnego przepływu end-to-end
- **`05_expression_eda.ipynb`** — eksploracyjna analiza ekspresji: konstrukcja
  macierzy TPM, log2(TPM+1), PCA z nakładkami klinicznymi, efekt batch TSS,
  markery LUAD
- **`06_baseline_survival.ipynb`** — bazowa analiza przeżywalności: Kaplan-Meier
  (cała kohorta, per stadium), model Coxa, sygnatura wielogenowa (panel
  ekspresyjny a priori), multivariate Cox z porównaniem C-index
- **`07_data_journey.ipynb`** — dydaktyczna podróż pojedynczego pliku STAR-Counts
  przez kolejne transformacje pipeline'u (surowy plik → parsowanie → metryka →
  filtr biotypów → macierz), z śledzeniem konkretnych markerów

## Struktura katalogów

```
luad-huba-clean/
├── data/
│   ├── raw/          # surowe pliki z GDC (.gitignore)
│   ├── interim/      # parquet po parse-star
│   ├── processed/    # macierz ekspresji, survival dataset
│   └── external/     # adnotacje (jeśli używane)
├── src/
│   ├── ingest/       # parsery + klient GDC API
│   ├── validate/     # QC kohorty
│   ├── transform/    # macierz, survival dataset
│   ├── export/       # manifest odtwarzalności
│   ├── cli.py        # komendy CLI (Typer)
│   └── cli_config.py # wczytywanie YAML
├── app/
│   ├── main.py       # interfejs Streamlit (8 etapów)
│   └── dashboard_viz.py  # wizualizacje Plotly
├── notebooks/        # 7 notebooków - sanity checks, EDA, survival
├── configs/          # YAML (default, filters)
├── logs/qc/          # raporty walidacji
├── .streamlit/       # konfiguracja motywu GUI
├── pyproject.toml
└── README.md
```

## Wyjście pipeline'u

| Plik | Opis |
|------|------|
| `data/interim/star_counts/*.parquet` | Sparsowane pliki STAR (jeden per próbka) |
| `data/processed/expression_matrix.parquet` | Macierz ekspresji (geny × próbki) |
| `data/processed/expression_matrix_manifest.json` | Manifest odtwarzalności |
| `data/processed/survival_dataset.parquet` | Zbiór przeżywalności (próbki × geny + clinical) |
| `logs/qc/qc_report_<UTC>.json` | Raport QC kohorty |

## Wymagania

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (lub pip)
- Dostęp do internetu dla `download` (anonimowy, bez autoryzacji dla
  open-access TCGA)

Kluczowe zależności: `polars` (przetwarzanie danych), `typer` (CLI),
`streamlit` (GUI), `plotly` (wizualizacje), `lifelines` i `scikit-survival`
(analiza przeżywalności), `pyarrow` (parquet).
