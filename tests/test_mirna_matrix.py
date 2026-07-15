# -*- coding: utf-8 -*-
"""Budowa macierzy miRNA — dzieli rdzeń z RNA-seq, różni się cechą i metrykami."""
from __future__ import annotations

import polars as pl
import pytest

from src.transform.expression_matrix import ExpressionMatrixError, build_mirna_matrix

MIRNA_SUFFIX = ".mirbase21.mirnas.quantification.txt"
MIRNAS = ["hsa-let-7a-1", "hsa-mir-21", "hsa-mir-155"]


def _write_mirna_parquet(directory, uuid: str, rpm: list[float]):
    path = directory / f"{uuid}.parquet"
    pl.DataFrame({
        "miRNA_ID": MIRNAS,
        "read_count": [int(v * 10) for v in rpm],
        "reads_per_million_miRNA_mapped": rpm,
    }).write_parquet(path)
    return path


@pytest.fixture
def mirna_matrix_inputs(tmp_path):
    uuids = ["aaaa-1111", "bbbb-2222", "cccc-3333"]
    paths = [
        _write_mirna_parquet(tmp_path, uuids[0], [100.0, 50.0, 10.0]),
        _write_mirna_parquet(tmp_path, uuids[1], [110.0, 55.0, 11.0]),
        _write_mirna_parquet(tmp_path, uuids[2], [120.0, 60.0, 12.0]),
    ]
    sheet = pl.DataFrame({
        "file_name": [f"{u}{MIRNA_SUFFIX}" for u in uuids],
        "sample_id": ["TCGA-44-0001-01A", "TCGA-44-0002-01A", "TCGA-44-0003-11A"],
    })
    return paths, sheet


def test_macierz_ma_mirna_w_wierszach(mirna_matrix_inputs):
    paths, sheet = mirna_matrix_inputs
    matrix = build_mirna_matrix(paths, sheet)
    assert matrix.columns[0] == "miRNA_ID"
    assert matrix.height == len(MIRNAS)
    assert set(matrix.columns[1:]) == set(sheet["sample_id"].to_list())


def test_domyslna_metryka_to_rpm(mirna_matrix_inputs):
    paths, sheet = mirna_matrix_inputs
    matrix = build_mirna_matrix(paths, sheet)
    assert matrix["TCGA-44-0001-01A"].to_list() == [100.0, 50.0, 10.0]


def test_metryka_read_count(mirna_matrix_inputs):
    paths, sheet = mirna_matrix_inputs
    matrix = build_mirna_matrix(paths, sheet, metric="read_count")
    assert matrix["TCGA-44-0001-01A"].to_list() == [1000, 500, 100]


def test_niedozwolona_metryka_mirna(mirna_matrix_inputs):
    paths, sheet = mirna_matrix_inputs
    with pytest.raises(ExpressionMatrixError):
        build_mirna_matrix(paths, sheet, metric="tpm_unstranded")  # to metryka RNA-seq


def test_niespojne_mirna_id_zglasza_blad(mirna_matrix_inputs, tmp_path):
    paths, sheet = mirna_matrix_inputs
    odd = tmp_path / "dddd-4444.parquet"
    pl.DataFrame({
        "miRNA_ID": ["hsa-mir-99", "hsa-mir-21", "hsa-mir-155"],  # inny pierwszy
        "read_count": [1, 2, 3],
        "reads_per_million_miRNA_mapped": [1.0, 2.0, 3.0],
    }).write_parquet(odd)
    sheet2 = sheet.vstack(pl.DataFrame({
        "file_name": [f"dddd-4444{MIRNA_SUFFIX}"], "sample_id": ["TCGA-44-0009-01A"],
    }))
    with pytest.raises(ExpressionMatrixError):
        build_mirna_matrix(paths + [odd], sheet2)


def test_pusta_lista_plikow():
    with pytest.raises(ExpressionMatrixError):
        build_mirna_matrix([], pl.DataFrame({"file_name": [], "sample_id": []}))


def test_brak_mapowania_pliku(mirna_matrix_inputs):
    paths, _ = mirna_matrix_inputs
    with pytest.raises(ExpressionMatrixError):
        build_mirna_matrix(paths, pl.DataFrame({"file_name": ["x.txt"], "sample_id": ["X"]}))


def test_duplikaty_deepest_wybiera_glebszy(tmp_path):
    a = _write_mirna_parquet(tmp_path, "eeee-5555", [10.0, 5.0, 1.0])
    b = _write_mirna_parquet(tmp_path, "ffff-6666", [200.0, 100.0, 20.0])  # głębszy
    sheet = pl.DataFrame({
        "file_name": [f"eeee-5555{MIRNA_SUFFIX}", f"ffff-6666{MIRNA_SUFFIX}"],
        "sample_id": ["TCGA-44-0001-01A", "TCGA-44-0001-01A"],  # ten sam sample
    })
    matrix = build_mirna_matrix([a, b], sheet, duplicate_strategy="deepest")
    assert matrix["TCGA-44-0001-01A"].to_list() == [200.0, 100.0, 20.0]
