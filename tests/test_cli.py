# -*- coding: utf-8 -*-
"""CLI: budowa sample sheet i mapowanie kodów TCGA."""
from __future__ import annotations

import polars as pl
import pytest

from src.cli import _tcga_code_to_sample_type, _tcga_code_to_tissue_type, _write_sample_sheet


@pytest.fixture
def files_metadata() -> pl.DataFrame:
    return pl.DataFrame({
        "file_id": ["f1", "f2"],
        "file_name": ["a.tsv", "b.tsv"],
        "data_type": ["Gene Expression Quantification"] * 2,
        "experimental_strategy": ["RNA-Seq"] * 2,
        "case_submitter_id": ["TCGA-44-0001", "TCGA-44-0002"],
        "sample_id": ["TCGA-44-0001-01A", "TCGA-44-0002-11A"],
    })


def test_sample_sheet_zapisuje_podany_projekt(files_metadata, tmp_path):
    """Regresja: Project ID był wpisany na sztywno jako TCGA-LUAD.

    Skutkiem było ciche uszkodzenie metadanych każdej kohorty innej niż LUAD —
    pobranie TCGA-BRCA dawało sample sheet twierdzący, że to LUAD.
    """
    out = tmp_path / "gdc_sample_sheet.tsv"
    _write_sample_sheet(files_metadata, out, "TCGA-BRCA")

    sheet = pl.read_csv(out, separator="\t")
    assert set(sheet["Project ID"].to_list()) == {"TCGA-BRCA"}


def test_sample_sheet_ma_format_portalu_gdc(files_metadata, tmp_path):
    out = tmp_path / "sheet.tsv"
    _write_sample_sheet(files_metadata, out, "TCGA-LUAD")

    sheet = pl.read_csv(out, separator="\t")
    assert sheet.columns == [
        "File ID", "File Name", "Data Type", "Data Category",
        "Project ID", "Case ID", "Sample ID", "Tissue Type", "Sample Type",
    ]
    assert sheet.height == files_metadata.height


def test_sample_sheet_klasyfikuje_guz_i_tkanke_prawidlowa(files_metadata, tmp_path):
    out = tmp_path / "sheet.tsv"
    _write_sample_sheet(files_metadata, out, "TCGA-LUAD")

    sheet = pl.read_csv(out, separator="\t")
    tissue = dict(zip(sheet["Sample ID"], sheet["Tissue Type"]))
    assert tissue["TCGA-44-0001-01A"] == "Tumor"
    assert tissue["TCGA-44-0002-11A"] == "Normal"


@pytest.mark.parametrize(("code", "expected"), [
    ("01A", "Tumor"),
    ("09A", "Tumor"),
    ("10A", "Normal"),
    ("11A", "Normal"),
])
def test_kod_tcga_na_tissue_type(code, expected):
    assert _tcga_code_to_tissue_type(code) == expected


def test_kod_tcga_na_sample_type():
    assert _tcga_code_to_sample_type("01A") == "Primary Tumor"
    assert _tcga_code_to_sample_type("11A") == "Solid Tissue Normal"


def test_gui_wola_write_sample_sheet_zgodnie_z_sygnatura():
    """Regresja: GUI ma własne wywołanie _write_sample_sheet (import z src.cli).

    Zmiana sygnatury w cli.py bez poprawy wywołania w app/main.py dawała w API
    TypeError (brak argumentu project_id). Test wiąże obie strony: liczba
    pozycyjnych argumentów w wywołaniu GUI musi pasować do definicji.
    """
    import ast
    import inspect
    from pathlib import Path

    required = sum(
        1 for p in inspect.signature(_write_sample_sheet).parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )

    main_src = Path(__file__).resolve().parent.parent / "app" / "main.py"
    calls = [
        node for node in ast.walk(ast.parse(main_src.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_write_sample_sheet"
    ]
    assert calls, "brak wywołania _write_sample_sheet w app/main.py"
    for call in calls:
        assert len(call.args) >= required, (
            f"wywołanie GUI ma {len(call.args)} argumentów, wymagane {required}"
        )


# --- pobieranie: modalność steruje workflow i data_type ------------------------

from src.cli import _resolve_download_target  # noqa: E402
from src.ingest.gdc_client import build_files_filter  # noqa: E402


def test_download_rnaseq_domyslne_parametry():
    mod, workflow, data_type = _resolve_download_target("rnaseq", None, {})
    assert mod.id == "rnaseq"
    assert workflow == "STAR - Counts"
    assert data_type == "Gene Expression Quantification"


def test_download_mirna_ustawia_workflow_i_data_type():
    """Regresja: download wpinał tylko workflow, data_type zostawał RNA-seq.

    miRNA wymaga innego data_type (miRNA Expression Quantification) i workflow
    (BCGSC miRNA Profiling) — inaczej filtr GDC zwróciłby pliki RNA-seq.
    """
    mod, workflow, data_type = _resolve_download_target("mirna", None, {})
    assert mod.id == "mirna"
    assert workflow == "BCGSC miRNA Profiling"
    assert data_type == "miRNA Expression Quantification"


def test_download_jawny_workflow_ma_pierwszenstwo():
    _, workflow, data_type = _resolve_download_target("mirna", "CUSTOM", {})
    assert workflow == "CUSTOM"
    assert data_type == "miRNA Expression Quantification"  # data_type wciąż z modalności


def test_download_filtr_niesie_mirna_data_type():
    _, workflow, data_type = _resolve_download_target("mirna", None, {})
    filt = build_files_filter(project_id="TCGA-LUAD", workflow_type=workflow, data_type=data_type)
    values = [cond["content"]["value"][0] for cond in filt["content"]]
    assert "miRNA Expression Quantification" in values
    assert "BCGSC miRNA Profiling" in values


def test_download_nieznana_modalnosc_zglasza_blad():
    with pytest.raises(KeyError):
        _resolve_download_target("rppa", None, {})
