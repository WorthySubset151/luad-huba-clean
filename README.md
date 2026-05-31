# LUAD-HUBA

**Lung Adenocarcinoma – Hybrid Unified Batch Analyzer**

Pipeline przygotowania danych sekwencjonowania RNA oraz danych klinicznych
z rejestru The Cancer Genome Atlas (projekt TCGA-LUAD) dla modeli predykcji
przeżywalności w gruczolakoraku płuca.

## Opis

LUAD-HUBA pobiera dane z interfejsu programistycznego Genomic Data Commons,
przeprowadza kontrolę jakości, normalizuje ekspresję genów do jednostek
Transcripts Per Million i eksportuje gotowy do trenowania zbiór danych
wraz z interaktywnym interfejsem graficznym (Streamlit).

## Architektura

```
ingest/ → validate/ → transform/ → export/ → app/
```

Każda warstwa to niezależny moduł Pythona z własnym interfejsem wiersza
poleceń, testami jednostkowymi i konfiguracją YAML.

## Uruchomienie

```bash
# Instalacja zależności (uv)
uv sync

# Pełny pipeline
luad-huba run-all --config configs/default.yaml

# Interfejs graficzny
streamlit run app/main.py
```

## Struktura katalogów

```
luad-huba/
├── data/
│   ├── raw/          # surowe pliki z GDC (gitignore)
│   ├── interim/      # po walidacji
│   ├── processed/    # gotowe macierze (parquet, csv)
│   └── external/     # adnotacje GENCODE
├── src/
│   ├── ingest/       # GDC API, pobieranie, MD5
│   ├── validate/     # kontrola jakości, filtry
│   ├── transform/    # normalizacja TPM, integracja
│   ├── export/       # Parquet, CSV, manifest
│   └── cli.py        # punkt wejścia (typer)
├── app/              # interfejs Streamlit
├── configs/          # konfiguracje YAML
├── logs/             # logi JSON
├── tests/            # testy pytest
├── notebooks/        # eksploracja danych
└── pyproject.toml
```

## Wyjście pipeline'u

| Plik | Opis |
|------|------|
| `expression_matrix.parquet` | Macierz ekspresji (próbki × geny), format kolumnowy |
| `clinical.csv` | Dane kliniczne: czas przeżycia, status, stadium |
| `manifest.json` | Metadane uruchomienia: wersja, skróty konfiguracji i danych |

## Wymagania

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (menedżer pakietów)
