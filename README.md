# LUAD-HUBA

**Lung Adenocarcinoma – Hybrid Unified Batch Analyzer**

Pipeline przygotowania danych sekwencjonowania RNA oraz danych klinicznych
z rejestru The Cancer Genome Atlas (projekt TCGA-LUAD) dla modeli predykcji
przeżywalności w gruczolakoraku płuca.

## Architektura

```
ingest/ → validate/ → transform/ → export/ → app/
```

Każda warstwa to niezależny moduł Pythona z własnym interfejsem wiersza
poleceń, testami jednostkowymi i konfiguracją YAML.

## Wymagania

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (menedżer pakietów)
