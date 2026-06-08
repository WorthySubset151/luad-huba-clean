# LUAD-HUBA

**Lung Adenocarcinoma – Hybrid Unified Batch Analyzer**

Samowystarczalny pipeline pobierania, walidacji i transformacji danych
sekwencjonowania RNA oraz danych klinicznych z projektu TCGA-LUAD (The Cancer
Genome Atlas) dla modeli predykcji przeżywalności w gruczolakoraku płuca.

## Opis

LUAD-HUBA komunikuje się bezpośrednio z REST API portalu Genomic Data Commons,
pobiera pełną kohortę (pliki STAR-Counts, sample sheet, clinical, metadata)
z weryfikacją sum kontrolnych MD5, przeprowadza kontrolę jakości na poziomie
kohorty i transformuje dane do gotowego do uczenia zbioru *survival dataset*
(próbki × geny + kolumny `time`, `event`, kowarianty kliniczne).

Nie wymaga ręcznego klikania w portalu - `git clone` + `uv sync` +
`luad-huba download` daje pełną lokalną kopię kohorty.

## Architektura

```
ingest/ → validate/ → transform/ → export/
```

Cztery warstwy, każda z jednoznaczną odpowiedzialnością:

- **ingest** - parsery STAR-Counts, sample sheet, clinical, metadata.cart.json;
  klient REST GDC API (endpointy `/files`, `/cases`, `/data`)
- **validate** - reguły QC na poziomie kohorty (missing files, orphan files,
  duplicate samples, missing clinical), raport JSON ze stemplem czasowym UTC
- **transform** - budowanie macierzy ekspresji (z obsługą duplikatów próbek),
  integracja z kowariantami klinicznymi, filtry survival (`min_follow_up_days`)
- **export** - manifest reproducibility (hash treści, lista plików źródłowych,
  metryka, znacznik czasowy)

Warstwa `app/` (interfejs Streamlit) - planowana, po dokończeniu warstwy modeli.

## Instalacja

```bash
git clone https://github.com/WorthySubset151/luad-huba-clean.git
cd luad-huba-clean
uv sync
```

Wymaga Pythona 3.12+ i [uv](https://github.com/astral-sh/uv).

## Użycie

Pipeline ma 5 komend CLI, które wykonują się w określonej kolejności:

```bash
# 1. Pobranie kohorty z GDC (jednorazowo, ~2.5 GB dla pełnej TCGA-LUAD)
luad-huba download --output-dir data/raw/

# 2. Parsowanie plików STAR-Counts do parquet (jeden plik per próbka)
luad-huba parse-star

# 3. Kontrola jakości kohorty (raport JSON w logs/qc/)
luad-huba validate-cohort

# 4. Budowa macierzy ekspresji (geny × próbki)
luad-huba build-matrix --config configs/default.yaml

# 5. Budowa survival dataset (próbki × geny + clinical)
luad-huba build-survival --config configs/default.yaml
```

Każda komenda ma `--help` z pełną listą opcji.

### Konfiguracja przez YAML

Wszystkie 5 komend akceptuje opcjonalną flagę `--config PATH` do pliku
konfiguracyjnego YAML. Domyślne configi znajdują się w `configs/`:

- `configs/default.yaml` - parametry pipeline'u (projekt, workflow, metryka
  normalizacji TPM/counts, próg `min_follow_up_days`)
- `configs/filters.yaml` - progi QC (planowane do rozbudowy)

Pierwszeństwo wartości: flaga CLI > config > hardcoded default.

Bez flagi `--config` pipeline działa identycznie jak dotąd
(backward compatible).

### Częste przypadki użycia

```bash
# Tylko pobierz metadane bez plików STAR (np. żeby zobaczyć ile pacjentów jest)
luad-huba download --skip-files

# Pobierz tylko 5 plików dla testów
luad-huba download --size 5

# Macierz z metryką TPM zamiast raw counts
luad-huba build-matrix --metric tpm_unstranded

# Survival dataset z innym progiem follow-up
luad-huba build-survival --config my_experiment.yaml

# Walidacja w trybie strict (exit 1 przy jakimkolwiek ERROR)
luad-huba validate-cohort --strict
```

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
│   ├── export/       # manifest reproducibility
│   ├── cli.py        # komendy CLI (Typer)
│   └── cli_config.py # wczytywanie YAML
├── notebooks/        # 4 notebooki - sanity check, eksploracja
├── configs/          # YAML
├── logs/qc/          # raporty walidacji
├── pyproject.toml
└── README.md
```

## Notebooki

- `01_explore_first_samples.ipynb` - sanity check biologiczny na 2 próbkach
  (housekeeping genes, markery LUAD, pułapka counts vs TPM)
- `02_full_cohort_pipeline.ipynb` - operacyjny sanity check pełnej kohorty
  (funnel 601→533, QC report, deep dive: tumor/normal po dedupe, outliery,
  multi-sample patients)
- `03_gdc_api_sanity_check.ipynb` - weryfikacja klienta GDC API
  (porównanie API vs referencyjny clinical.tsv z portalu)

## Wyjście pipeline'u

| Plik | Opis |
|------|------|
| `data/processed/expression_matrix.parquet` | Macierz ekspresji (geny × próbki) |
| `data/processed/expression_matrix_manifest.json` | Manifest reproducibility |
| `data/processed/survival_dataset.parquet` | Gotowy survival dataset (próbki × geny + clinical) |
| `logs/qc/qc_report_<UTC>.json` | Raport QC kohorty |

## Wymagania

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (lub pip)
- Dostęp do internetu dla `luad-huba download` (anonimowy, bez autoryzacji
  dla open-access TCGA)
