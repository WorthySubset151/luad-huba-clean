# LUAD-HUBA

**Lung Adenocarcinoma – Hybrid Unified Batch Analyzer**

Samowystarczalny pipeline pobierania, walidacji, transformacji i analizy danych
sekwencjonowania RNA oraz danych klinicznych z projektu TCGA-LUAD (The Cancer
Genome Atlas) dla analizy przeżywalności w gruczolakoraku płuca. Obejmuje
warstwę ETL (CLI), interaktywny interfejs graficzny (Streamlit) z dashboardem
analitycznym oraz zestaw notebooków dokumentujących każdy etap przetwarzania.

> **Aplikacja na żywo:** [luad-app-clean.streamlit.app](https://luad-app-clean.streamlit.app)
> — pełny interfejs działa w chmurze (Streamlit Cloud). Uwaga: na hostowanej
> instancji dane są efemeryczne (resetują się przy restarcie aplikacji), a
> limity zasobów serwera mogą ograniczać operacje na dużych kohortach.

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
- [Testy](#testy)
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
│         WSPÓLNY RDZEŃ ANALIZY (src/analysis/)           │
│   survival_report.py · expression_report.py             │
│   jedno źródło prawdy: KM, Cox, FDR, ekspresja, PCA      │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────┐
        │        GUI (app/, Streamlit)       │
        │   dashboard, panele, eksport ML    │
        └────────────────────────────────────┘
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

Pipeline ma **dwa interfejsy**: CLI (skrypty i automatyzacja) oraz GUI Streamlit
(`app/`, dashboard wizualny z panelami i eksportem zbioru pod ML). GUI liczy
analizy z rdzenia `src/analysis/` (survival_report.py, expression_report.py) i
wspólnej logiki zarządzania `src/manage/` — jedno źródło prawdy, spójne
wyniki. Klient GDC pobiera pliki współbieżnie (httpx.AsyncClient + semafor)
z ponawianiem błędów przejściowych.

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
| **Pobieranie** | — | Pobranie danych TCGA-LUAD z GDC API (podzbiory) |
| **Wgrywanie** | — | Wgranie clinical.tsv, sample sheet i plików STAR przez archiwum ZIP |
| **Przeglądanie** | — | Podgląd plików (parquet, TSV, YAML) w repozytorium |
| **Parsowanie** | surowe pliki | Parsowanie STAR-Counts do parquet z paskiem postępu |
| **Walidacja** | sparsowane parquety | Kontrola jakości kohorty z raportem QC |
| **Macierz ekspresji** | sparsowane parquety | Budowa macierzy genów × próbki z paskiem postępu |
| **Zbiór przeżywalności** | macierz | Integracja ekspresji z danymi klinicznymi |
| **Dashboard analityczny** | zbiór przeżywalności | Wizualizacje KM, sygnatury, ekspresja markerów |
| **Konfiguracja** | — | Edycja parametrów pipeline'u (YAML) |
| **Zarządzanie danymi** | — | Archiwizacja (backup ZIP) i kasowanie plików pipeline'u |

Wszystkie sekcje są w pełni funkcjonalne i działają zarówno lokalnie, jak i na
hostowanej instancji (Streamlit Cloud). Pobieranie używa współbieżnego klienta
API w Pythonie (httpx.AsyncClient + semafor + ponawianie błędów przejściowych —
w praktyce sprawne nawet dla setek/tysięcy plików), a Wgrywanie przyjmuje pliki
przez archiwum ZIP — oba podejścia
działają niezależnie od środowiska (nie wymagają dostępu do systemu plików
serwera ani zewnętrznych narzędzi binarnych).

### Dashboard analityczny

Dashboard (odblokowany po zbudowaniu zbioru przeżywalności) ma **trzy pod-zakładki**:
dwie z interaktywnymi wizualizacjami biologicznymi (Plotly: najechanie kursorem,
przybliżanie) oraz ilościową ocenę gotowości danych pod ML wraz z eksportem zbioru
treningowego.

**Zakładka Przeżywalność:**

- **Kaplan-Meier — cała kohorta** + statystyki (mediana przeżycia całkowitego,
  przeżycie 1/3/5-letnie)
- **Kaplan-Meier — per stadium** (rozbicie na grupy I-IV) + test log-rank
- **Kaplan-Meier — sygnatura wielogenowa** (panel ekspresyjny a priori 7 genów)
  + test log-rank
- **Kaplan-Meier — pojedynczy gen** z wyborem genu z listy markerów; nad
  wykresem wyświetla się precyzyjna charakterystyka roli genu w LUAD, pod
  wykresem stratyfikacja high/low względem mediany ekspresji + test log-rank
- **Model Coxa (wielowymiarowy)** — klinika (wiek, płeć, stadium) + panel genów,
  z HR, przedziałami ufności i C-index
- **Rygor statystyczny** — jawny próg α = 0,05, korekcja wielokrotnego testowania
  Benjamini-Hochberg FDR na panelu 7 genów (single-gene log-rank oraz Cox) oraz
  sensitivity analysis Schoenfelda (minimalny wykrywalny HR przy danej liczbie
  zdarzeń, zamiast nieważnego post-hoc power)

**Zakładka Ekspresja:**

- **Rozkład ekspresji** log2(TPM+1) dla wybranej próbki; po wyborze próbki
  wyświetla się panel z jej danymi klinicznymi (pacjent, stadium, wiek, płeć,
  czas obserwacji, status, typ tkanki) — łączy ekspresję z kontekstem klinicznym
- **Ekspresja markerów LUAD** — rozkład klasycznych markerów raka płuca po
  wszystkich próbkach (wykres pudełkowy)

**Zakładka Gotowość ML:**

Ilościowa ocena, czy dane nadają się pod modele ML (docelowo multimodalne), z
kontrolkami ![OK](https://img.shields.io/badge/OK-7a9b5e) ![uwaga](https://img.shields.io/badge/uwaga-c2a24a) ![działanie](https://img.shields.io/badge/dzia%C5%82anie-bd7a6a) i zbiorczym werdyktem. Grupy metryk: wymiary i reżim uczenia
(liczba próbek i cech, p/n, zdarzeń na cechę EPV), etykieta (zdarzenia, cenzura,
kompletność klinicznych per pole: wiek/płeć/stadium), balans klas (nierównowaga
stadiów), jakość cech (metryka normalizacji, rzadkość, geny o zerowej wariancji),
batch/wyciek (nierównowaga ośrodków TSS oraz test „PC1 — biologia czy batch?" jako
η²(PC1~stadium) vs η²(PC1~ośrodek)) i integralność (duplikaty próbek na pacjenta).

Zakładka zawiera też **Eksport zbioru gotowego pod ML** — z audytu robi remediację:
deduplikacja do 1 próbki/pacjenta, podział train/test grupowany po pacjencie
(`case_id`) ze stratyfikacją po zdarzeniu, selekcja cech (top-K wg wariancji) i
standaryzacja fitowane **wyłącznie na train** (bez wycieku do testu), log2(x+1) i
odrzucenie genów o zerowej wariancji. Wynik (`X_train`/`X_test`, `y_train`/`y_test`,
lista wybranych genów, `manifest.json` z pełną dokumentacją kroków i rekomendowanymi
ustawieniami modelu) pobiera się z GUI jako archiwum ZIP. Tabele zapisywane są
domyślnie w **Parquet** — zachowuje typy kolumn i wartości brakujące, czyta się
kilkukrotnie szybciej niż CSV i przy typowych rozmiarach daje mniejsze pliki;
formatem alternatywnym jest CSV (do otwarcia w arkuszu).

### Zarządzanie danymi

Sekcja domyka cykl życia danych w aplikacji — pozwala zarchiwizować dane przed
skasowaniem i wyczyścić pliki pipeline'u, by zacząć od nowa z inną kohortą.
Dane podzielone są na **cztery niezależne kategorie**, by ciężkie pliki STAR
(gigabajty) można było archiwizować i kasować osobno od lekkich metadanych:

- **Pliki STAR-Counts** (`data/raw/uploaded_star`) — surowe pliki ekspresji,
  zwykle gigabajty
- **Metadane kohorty** (`data/raw`, bez podkatalogu STAR) — clinical.tsv,
  sample sheet, metadata.cart.json
- **Parquety pośrednie** (`data/interim`) — sparsowane pliki STAR
- **Wyniki finalne** (`data/processed`) — macierz ekspresji, zbiór przeżywalności

**Archiwizacja (backup).** Pakuje wybrane kategorie (dowolna kombinacja) do
archiwum ZIP i udostępnia do pobrania. Typowy przypadek: zarchiwizować same
pliki STAR, by nie pobierać gigabajtów ponownie. Działa przez przeglądarkę,
więc również na hostowanej instancji; dla dużych zbiorów pojawia się ostrzeżenie
o limitach pamięci (pakowanie odbywa się w RAM).

**Kasowanie.** Granularne, z naciskiem na bezpieczeństwo, ponieważ operacja
jest nieodwracalna:
- wybór kategorii (te same cztery; np. usunięcie samych plików STAR, by zwolnić
  miejsce, zachowując metadane i wyniki)
- podgląd dokładnego zakresu przed wykonaniem (ile plików, jaki rozmiar)
- twarde potwierdzenie przez wpisanie słowa `USUŃ` (przycisk pozostaje
  zablokowany, dopóki słowo się nie zgadza)
- gwarancja bezpieczeństwa ścieżek — operacja działa wyłącznie wewnątrz
  katalogu `data/` i odmawia usunięcia czegokolwiek poza nim

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

# Naprawa pustych kolumn klinicznych w gotowym zbiorze — dogrywa wiek/płeć/stadium
# z clinical.tsv po case_id, bez przebudowy macierzy (etykiety i geny nietknięte)
uv run python -m src.cli repair-clinical --clinical data/raw/clinical.tsv
```

## Omówienie modułów i funkcji

### `src/modality.py` — czym jest cecha

Jedno miejsce definiujące, jak rozpoznać kolumny cech w zbiorze i jak o nich mówić.
Wcześniej rdzeń zakładał wprost, że cecha to gen z prefiksem `ENSG` — sprawdzenie
`startswith("ENSG")` powtarzało się w pięciu modułach, więc dodanie kolejnej
modalności GDC (miRNA, ekspresja białek) wymagałoby zmiany rdzenia w każdym z nich.

`Modality` to niemutowalny opis: identyfikator, etykieta, prefiks kolumn cech,
rzeczownik opisujący cechę i typowa metryka. Metoda `feature_columns(frame)` zwraca
kolumny cech (działa dla ramek polars i pandas). Dodanie modalności to dopisanie
instancji do `REGISTRY`, nie zmiana rdzenia.

```python
RNASEQ = Modality(
    id="rnaseq",
    label="Ekspresja genów (RNA-seq)",
    feature_prefix="ENSG",
    feature_noun="gen",
    feature_noun_plural="geny",
    default_metric="TPM",
)
```

Analizy generyczne (`cohort_summary`, `ml_readiness_report`, `prepare_ml_dataset`)
przyjmują modalność jako parametr z domyślnym RNA-seq. Analizy oparte na panelu genów
LUAD (sygnatura wielogenowa, model Coxa na panelu) pozostają swoiste dla RNA-seq —
panel siedmiu genów nie ma odpowiednika w miRNA, więc parametr modalności byłby tam
pozorny.

Modalność opisuje też prefiks kolumn cech (dla miRNA to kilka prefiksów — `hsa-mir`,
`hsa-let` — bo nazwy miRBase nie mają wspólnego rdzenia) oraz filtry pobierania z GDC
(`gdc_filters()`: `data_type` i `workflow_type` różnią się między modalnościami). Druga
zarejestrowana modalność, `MIRNA`, korzysta z tej samej infrastruktury co RNA-seq;
parser miRNA jest w `src/ingest/mirna_parser.py`.

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
endpointów `/files`, `/cases`, `/data`. Pobieranie jest **współbieżne**
(`httpx.AsyncClient` + `asyncio.gather` z semaforem, domyślnie 15 plików naraz),
z **ponawianiem** błędów przejściowych (tenacity, exponential backoff — timeouty,
5xx/429, niezgodność MD5; błędy 4xx nie są ponawiane) i strumieniową weryfikacją
sum MD5. Zapytania są **parametryzowane projektem** (`project_id`, domyślnie
TCGA-LUAD) — ten sam kod pobiera dowolny projekt TCGA (BRCA, GBM, …). Sygnatura
`download_files` pozostaje synchroniczna (z guardem na działającą pętlę zdarzeń),
więc GUI korzysta ze współbieżności bez żadnych zmian po swojej
stronie.

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

### `src/analysis/` — wspólny rdzeń analizy (jedno źródło dla GUI)

**`survival_report.py`** — bezgłowy rdzeń analizy przeżywalności zwracający czyste
struktury danych (dict), z których GUI (Plotly) renderuje wyniki. Funkcje: `cohort_summary`, `km_report`/`km_summary`,
`cox_clinical_report`, `cox_genes_report`, `signature_km_report`,
`single_gene_km_report`, `multi_gene_km_report`. **Rygor statystyczny:** `ALPHA`
(0,05), `benjamini_hochberg`/`multiple_testing_report` (korekcja FDR),
`gene_panel_fdr_report` (FDR panelu 7 genów: single-gene log-rank oraz Cox),
`schoenfeld_min_hr`/`schoenfeld_power`/`sensitivity_report` (sensitivity analysis
Schoenfelda), `statistical_rigor_report` (raport zbiorczy). Stałe: `SIGNATURE_PANEL`
(7 genów ze znakami), `STAGE_NUMERIC_MAP`, `COX_CLINICAL_LABELS`.

**`expression_report.py`** — `expression_summary(matrix_path)`: wymiary macierzy,
autodetekcja metryki (TPM vs counts), rozkład ekspresji, efekt batch TSS, PCA
(wariancja głównych składowych), markery LUAD.

**`readiness_report.py`** — `ml_readiness_report(ds, esum)`: metryki gotowości pod
ML pogrupowane w kategorie (wymiary/reżim, etykieta, balans klas, jakość cech,
batch/wyciek, integralność), każda ze statusem green/yellow/red, plus zbiorczy
werdykt i lista zalecanych kroków. Liczy m.in. p/n, EPV, kompletność etykiet per
pole, nierównowagę stadiów, geny o zerowej wariancji, nierównowagę ośrodków TSS oraz
test PC1 (biologia vs batch) metodą η² (correlation ratio).

**`ml_export.py`** — remediacja i eksport zbioru pod ML. `prepare_ml_dataset(ds, …)`:
deduplikacja do 1 próbki/pacjenta (deepest), podział train/test **grupowany po
pacjencie** ze stratyfikacją po zdarzeniu, selekcja cech (top-K wg wariancji) i
standaryzacja **fitowane wyłącznie na train** (dyscyplina braku wycieku), log2(x+1),
odrzucenie zerowej wariancji. Zwraca `X_train`/`X_test`, `y_train`/`y_test`, listę
genów i manifest; `build_ml_bundle(result, fmt="parquet")` pakuje całość (z
`manifest.json` i README) w archiwum ZIP — `fmt` przyjmuje `parquet` (domyślnie)
albo `csv`, a użyty format trafia do manifestu.

### `src/manage/` — zarządzanie danymi

**`data_ops.py`** — wspólna logika archiwizacji i bezpiecznego kasowania zakresów
danych. Definicje zakresów `MANAGE_SCOPES` (STAR, metadane, parquety pośrednie,
wyniki finalne), twardy guard `is_within_data` (operacje wyłącznie wewnątrz
`data/`, neutralizuje też `..`), tryby shallow/recursive, `scope_stats`,
`delete_scope`, `build_archive_zip` (do pamięci — dla pobrania w GUI) oraz
`build_archive_to_path` (strumieniowo na dysk — bezpieczne dla
gigabajtów surowych STAR).

### `src/cli_config.py` — konfiguracja

**`load_config(path)`** wczytuje plik YAML, **`get_nested(config, *keys)`**
pobiera zagnieżdżone wartości z bezpiecznym fallbackiem, **`resolve_metric`**
mapuje aliasy metryk (np. `tpm` → `tpm_unstranded`).

### `app/` — interfejs i wizualizacje

**`main.py`** — aplikacja Streamlit: wykrywanie stanu z dysku (`detect_state`),
wymuszanie kolejności etapów (`stage_unlocked`), funkcje renderujące dla każdej
sekcji (`render_browse`, `render_parse`, `render_build_matrix`,
`render_build_survival`, `render_validate`, `render_download`, `render_upload`,
`render_manage`, `render_dashboard` — w tym sekcja rygoru statystycznego —
`render_config`) oraz pomocnicze (`_render_qc_report` — raport walidacji z podziałem na problemy
obsługiwane automatycznie vs wymagające uwagi; `_render_sample_clinical` — panel
kliniczny wybranej próbki).

**`dashboard_viz.py`** — wizualizacje Plotly budowane na zbiorze przeżywalności.
Funkcje KM **delegują obliczenia** do `src/analysis/survival_report.py` (jedno
źródło prawdy) i tylko rysują wynik (`km_single_gene`, `km_multi_gene`,
`tss_batch_figure`, `pca_variance_figure` i in.) — dzięki temu wykresy w GUI
pochodzą z tych samych liczb.

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

**Rygor statystyczny analizy.** Analiza przeżywalności deklaruje jawny próg
istotności **α = 0,05** (dwustronnie). Panele 7-genowe (single-gene log-rank
oraz wielowymiarowy Cox) przechodzą **korekcję wielokrotnego testowania metodą
Benjaminiego-Hochberga** (FDR) — mniej konserwatywną niż Bonferroni, kontrolującą
odsetek fałszywych odkryć. Zamiast nieważnego *post-hoc power* (liczonego
z obserwowanego efektu — wadliwego wg Hoeniga i Heisey, 2001) raportowana jest
**sensitivity analysis wzorem Schoenfelda**: minimalny wykrywalny HR przy danej
liczbie **zdarzeń** i zadanej mocy. Dla analizy przeżycia liczą się zdarzenia,
nie liczba pacjentów (n jest dane — cała kohorta TCGA), więc prospektywna analiza
mocy byłaby bezprzedmiotowa. Na kohorcie referencyjnej (147 zdarzeń) minimalny
wykrywalny HR to ≈ 1,59 przy mocy 80% i ≈ 1,71 przy 90%.

**Model Coxa — zbieżność.** Cox liczony jest domyślnie bez penalizera (artefakty
kolinearności widoczne świadomie), ale gdy Newton-Raphson nie zbiega (silna
kolinearność skorelowanych genów, macierz osobliwa), automatycznie włącza się ridge
penalizer (L2), zwiększany aż do stabilnego dopasowania — panel liczy wynik zamiast
się wywalać. Kolumny o zerowej wariancji są odrzucane przed dopasowaniem.

**Przygotowanie danych pod ML — dyscyplina braku wycieku.** Warstwa eksportu
(zakładka Gotowość ML) świadomie rozdziela to, co wolno wpiec w dane, od tego, co
należy do modelu. W danych: deduplikacja do jednej próbki na pacjenta, podział
train/test **grupowany po pacjencie** (`case_id`, nie po próbce — inaczej ten sam
pacjent trafiłby do train i test), stratyfikacja po zdarzeniu, a selekcja cech
(top-K wg wariancji, nienadzorowana) i standaryzacja **fitowane wyłącznie na train**.
W manifeście (nie w danych): regularyzacja przy p≫n/EPV, wagi klas przy nierównowadze
stadiów, korekta batchu (ComBat / TSS jako kowarianta) i GroupKFold do walidacji — to
decyzje modelowe, których wpiekanie w zbiór dałoby przeciek albo utratę informacji.

**Kontrola kompletności danych klinicznych.** Budowa zbioru przeżywalności ma
strażnika: po złączeniu kliniki sprawdza wypełnienie kowariantów (wiek, płeć,
stadium). Gdy kolumna jest praktycznie pusta (typowy objaw niepełnego eksportu z API
GDC), build **przerywa się z jasnym komunikatem** zamiast po cichu wyprodukować
wadliwy zbiór; przy częściowych brakach ostrzega. Pełny, kompletny eksport uzyskuje
się z `clinical.tsv` (eksport „cart" z portalu GDC); istniejący zbiór z pustą kolumną
naprawia komenda `repair-clinical`.

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
│   ├── ingest/       # parsery + współbieżny klient GDC API
│   ├── validate/     # QC kohorty
│   ├── transform/    # macierz, survival dataset
│   ├── modality.py   # czym jest cecha (RNA-seq dziś, miRNA/RPPA docelowo)
│   ├── analysis/     # rdzeń: survival, expression, readiness (Gotowość ML), ml_export
│   ├── manage/       # archiwizacja + bezpieczne kasowanie (data_ops)
│   ├── export/       # manifest odtwarzalności
│   ├── cli.py        # komendy CLI (Typer)
│   └── cli_config.py # wczytywanie YAML
├── app/
│   ├── main.py       # interfejs Streamlit (10 sekcji)
│   └── dashboard_viz.py  # wizualizacje Plotly (delegują do src/analysis)
├── tests/            # pytest: rdzeń pipeline'u, nacisk na brak wycieku
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

## Testy

Repozytorium ma zestaw testów `pytest` obejmujący rdzeń pipeline'u — od parsera
klinicznego po eksport zbioru pod ML.

```bash
uv run pytest              # całość
uv run pytest -v           # z nazwami testów
uv run pytest tests/test_ml_export.py   # pojedynczy moduł
```

Testy są **samowystarczalne**: dane wejściowe (klinika w formacie eksportu „cart"
GDC, macierz ekspresji, sample sheet) generowane są syntetycznie w `tests/conftest.py`,
więc uruchamiają się na świeżym klonie bez pobierania kohorty i bez plików w `data/`.

Co jest pokryte:

| Plik | Zakres |
|---|---|
| `test_clinical_parser.py` | filtrowanie diagnoz podstawowych, wyliczanie `time`/`event`, markery braków GDC |
| `test_survival_dataset.py` | integracja ekspresji z kliniką, kolejność kolumn, strażnik kompletności kowariantów |
| `test_readiness_report.py` | mapowanie metryki normalizacji, kompletność per pole, odporność PCA, η², werdykt |
| `test_ml_export.py` | **brak wycieku**, fit wyłącznie na train, deduplikacja, determinizm, format archiwum |
| `test_repair_clinical.py` | naprawa kowariantów bez ruszania etykiet i genów |
| `test_modality.py` | rozpoznawanie cech, rejestr modalności, gotowość rdzenia na inne prefiksy |
| `test_cli.py` | sample sheet (w tym regresja `Project ID`), mapowanie kodów TCGA |
| `test_mirna_parser.py` | parser miRNA Expression Quantification (kolumny, typy, walidacja) |
| `test_smoke.py` | składnia, importy, rejestracja komend CLI, higiena pakietu |

**Nacisk na brak wycieku.** Błąd w tym miejscu nie wywala się głośno — po cichu
zawyża wyniki modelu, więc jedyną obroną jest asercja. Testy sprawdzają rozłączność
pacjentów między train i test (również przy wyłączonej deduplikacji, gdy grupowanie
jest jedynym zabezpieczeniem) oraz to, że selekcja cech i standaryzacja nie zaglądają
do zbioru testowego: zmiana wartości wyłącznie w teście nie może zmienić macierzy
treningowej. Asercje standaryzacji są liczone **per gen** — globalna średnia po całej
macierzy wychodzi bliska zeru nawet przy wycieku, więc przepuszczałaby błąd.

## Wymagania

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (lub pip)
- Dostęp do internetu dla `download` (anonimowy, bez autoryzacji dla
  open-access TCGA)

Kluczowe zależności: `polars` (przetwarzanie danych), `typer` (CLI),
`streamlit` (GUI), `httpx` + `tenacity`
(współbieżny klient GDC z ponawianiem), `plotly` (wizualizacje), `lifelines`
i `scikit-survival` (analiza przeżywalności), `scipy` (statystyka: FDR,
Schoenfeld), `pandas` + `pyarrow` (parquet — magazyn danych i eksport zbiorów ML).
Zależności deweloperskie (`pip install -e ".[dev]"`): `pytest` (testy), `ruff`,
`black`.


## Przewodnik po dashboardzie analitycznym

Dashboard ma trzy zakładki: **Przeżywalność** (jakość etykiety/targetu), **Ekspresja**
(jakość cech/features) i **Gotowość ML** (ilościowa ocena przydatności danych pod modele).
Na ekranie pod każdym wykresem jest tylko jednozdaniowy podpis, żeby zachować czytelność —
pełne wyjaśnienia (co pokazuje, po co jest, jak je czytać, co znaczy dla ML) są tutaj.

### Zakładka Przeżywalność — jakość etykiety (target)

**Kaplan-Meier — cała kohorta.** Krzywa przeżycia całej kohorty (odsetek żyjących w czasie),
z 95% pasem ufności. *Po co:* baseline przed podziałem na grupy; mediana OS i przeżycia
1/3/5-letnie porównywalne z literaturą. *Jak czytać:* każdy schodek to zgon; im wolniej opada,
tym lepsze rokowanie; pas rozszerza się na końcu (mniej obserwowanych). *Pod ML:* to rozkład
zmiennej celu — zdarzenia, nie cenzury, niosą sygnał uczący, więc ich liczba wyznacza realną moc.

**Kaplan-Meier — per stadium.** Krzywe rozbite na stadia AJCC I–IV + test log-rank. *Po co:*
sanity check etykiety — najsilniejszy znany czynnik (stadium) musi rozdzielać przeżycie.
*Jak czytać:* wachlarz krzywych (I najwyżej, IV najniżej); p<0,05 = istotne. *Pod ML:* dowód,
że target niesie wyuczalny sygnał; sensowny model na ekspresji powinien pobić ten kliniczny baseline.

**Model Coxa — kowarianty kliniczne.** Proporcjonalne hazardy (wiek, płeć, stadium): wkład
każdego czynnika w ryzyko zgonu, forest plot HR z CI. *Po co:* baseline predykcyjny na samej
klinice — poprzeczka dla ekspresji. *Jak czytać:* HR>1 ryzyko, HR<1 ochronny; CI przez 1 =
nieistotny; C-index 0,5 losowy, 1,0 idealny (klinika ~0,6–0,7). *Pod ML:* benchmark oraz wskazówka,
które kowarianty kontrolować, by sygnał genów nie był przebraniem za stadium.

**Kaplan-Meier — sygnatura wielogenowa.** Podział high/low wg score 7-genowego panelu. *Po co:*
czy ekspresja (a nie klinika) rozdziela przeżycie. *Jak czytać:* rozjazd krzywych = sygnatura
różnicuje rokowanie; p<0,05 istotne. *Pod ML:* dowód, że cechy korelują z targetem; score liczony
a priori — bez douczania na tych danych, więc bez wycieku informacji.

**Kaplan-Meier — pojedynczy gen.** Dla wybranego genu: przeżycie high/low względem mediany +
charakterystyka genu. *Po co:* eksploracja pojedynczych cech. *Jak czytać:* rozjazd + log-rank p;
pojedynczy gen rzadko silny, wiele testów = ryzyko fałszywych trafień (stąd FDR niżej). *Pod ML:*
univariate screening — ale nie selekcjonuj cech po tych p (przeuczenie); od tego jest walidacja krzyżowa.

**Kaplan-Meier — porównanie wielu genów.** Krzywe grup „high" kilku genów naraz + tabela log-rank.
*Po co:* szybkie porównanie siły prognostycznej. *Jak czytać:* niżej leżąca krzywa „high" = wyższa
ekspresja wiąże się z gorszym rokowaniem; testy wciąż uniwariackie. *Pod ML:* wstępny ranking cech;
zgodność z biologią (proliferacyjne = ryzyka, różnicowania = ochronne) buduje zaufanie do featurów.

**Model Coxa — klinika + panel genów.** Łączy klinikę i geny, porównuje C-index (klinika vs
klinika+geny), forest plot HR genów po korekcie o klinikę. *Po co:* kluczowe pytanie — czy ekspresja
dodaje wartość PONAD klinikę. *Jak czytać:* patrz na Δ C-index; pojedyncze geny bywają nieistotne,
a model i tak lepszy (sygnał w kombinacji); uwaga na kolinearność. *Pod ML:* najmocniejszy argument
za sensem modelu na ekspresji; Δ C-index to uczciwa miara wartości dodanej.

**Rygor statystyczny.** (a) Korekcja wielokrotnego testowania Benjamini-Hochberg (FDR) na panelu
7 genów; (b) sensitivity analysis Schoenfelda — jaki HR da się wykryć przy danej liczbie zdarzeń.
*Po co:* uczciwość — część „istotnych" trafień z 7 testów to przypadek; sensitivity zastępuje nieważny
post-hoc power. *Jak czytać:* q(FDR)<α = istotne po korekcie; min. wykrywalny HR to próg czułości.
*Pod ML:* realistyczny sufit (wykryjesz HR≥~1,5, słabszych nie); FDR to ta sama dyscyplina co przy
feature selection.

### Zakładka Ekspresja — jakość cech (features)

**Podsumowanie macierzy.** Wymiar (geny × próbki), wykryta metryka normalizacji, mediana, % zer,
głębokość. *Po co:* format i jakość cech; metryka decyduje o porównywalności między próbkami. *Jak
czytać:* TPM porównywalne między próbkami, FPKM zbliżone, counts surowe; zera ~13% to normalna
rzadkość; filtr biotypów = mniej szumu. *Pod ML:* niezależnie od metryki log2(x+1) stabilizuje
wariancję; wysoka rzadkość → modele odporne na zera/regularyzacja; FPKM → rozważ TPM.

**Rozkład ekspresji (histogram).** Histogram log2(TPM+1) jednej próbki + jej dane kliniczne. *Po co:*
kontrola jakości pojedynczej próbki. *Jak czytać:* zdrowy profil jest bimodalny (pik przy 0 = geny
wyłączone, garb wyżej = aktywne); brak drugiego garbu = próbka podejrzana. *Pod ML:* wykrywanie
odstających próbek zanim trafią do treningu; log-transformacja to typowy preprocessing.

**Ekspresja markerów LUAD (boxplot).** Rozkład markerów (EGFR, KRAS, TP53, ALK, NKX2-1, SFTPC…) po
próbkach. *Po co:* walidacja biologiczna — czy markery zachowują się zgodnie z wiedzą. *Jak czytać:*
NKX2-1/SFTPC zwykle wysokie, ALK/ROS1 (działają przez fuzje) zwykle niskie; rażące odstępstwa =
problem z macierzą/anotacją. *Pod ML:* sanity check cech — potwierdza, że kolumny to realnie geny,
a nie artefakt mapowania.

**Batch — ośrodki TSS.** Liczba próbek per ośrodek (TSS z barkodu TCGA), słupki posortowane. *Po co:*
wykrycie efektu batch — największego ryzyka confoundingu w danych wieloośrodkowych. *Jak czytać:*
silna nierównowaga = ryzyko, zwłaszcza gdy ośrodek koreluje z targetem. *Pod ML:* batch może wyciekać
do modelu (batch leakage); środki zaradcze: ComBat/limma, TSS jako kowarianta, sprawdzenie PCA.

**PCA — wariancja składowych.** Ile zmienności tłumaczy każda składowa (PC1–5), po log2 + z-score. *Po
co:* mapa głównych osi zmienności; czy PC1 to biologia czy artefakt. *Jak czytać:* sam wykres nie mówi,
co to jest — trzeba nałożyć kolor (stadium/ośrodek); PC1 wg ośrodka = batch (źle), wg stadium = biologia
(dobrze). *Pod ML:* diagnoza confoundingu; PCA bywa też redukcją wymiaru przed modelem.

### Zakładka Gotowość ML — ocena przydatności pod modele

Zakładka liczy zestaw metryk diagnostycznych i przy każdej pokazuje kontrolkę ![OK](https://img.shields.io/badge/OK-7a9b5e) ![wymaga uwagi](https://img.shields.io/badge/wymaga%20uwagi-c2a24a) ![wymaga działania](https://img.shields.io/badge/wymaga%20dzia%C5%82ania-bd7a6a), plus zbiorczy werdykt i listę zalecanych kroków przygotowania danych. Grupy
metryk: **wymiary i reżim uczenia** (liczba próbek i cech, p/n, zdarzeń na cechę EPV), **etykieta**
(zdarzenia, cenzura, kompletność klinicznych), **balans klas** (nierównowaga stadiów), **jakość cech**
(metryka normalizacji, rzadkość, geny o zerowej wariancji), **batch/wyciek** (nierównowaga ośrodków TSS
oraz test „PC1 — biologia czy batch?" liczony jako η²(PC1~stadium) vs η²(PC1~ośrodek)) i **integralność**
(duplikaty próbek na pacjenta). Metryki liczy `src/analysis/readiness_report.py` (jedno źródło), więc są
odtwarzalne i niezależne od interfejsu.
