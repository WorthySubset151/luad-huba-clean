# -*- coding: utf-8 -*-
"""Budowa macierzy ekspresji — RNA-seq (siatka bezpieczeństwa przed refaktorem)."""
from __future__ import annotations

import polars as pl
import pytest

from src.transform.expression_matrix import ExpressionMatrixError, build_expression_matrix

STAR_SUFFIX = ".rna_seq.augmented_star_gene_counts.tsv"
GENES = ["ENSG00000000003.15", "ENSG00000000005.6", "ENSG00000000419.13"]
GENE_TYPES = ["protein_coding", "protein_coding", "lncRNA"]


def _write_sample_parquet(directory, uuid: str, values: list[float]):
    path = directory / f"{uuid}.parquet"
    pl.DataFrame({
        "gene_id": GENES,
        "gene_type": GENE_TYPES,
        "unstranded": [int(v) for v in values],
        "tpm_unstranded": values,
    }).write_parquet(path)
    return path


@pytest.fixture
def star_matrix_inputs(tmp_path):
    uuids = ["11111111-aaaa", "22222222-bbbb", "33333333-cccc"]
    paths = [
        _write_sample_parquet(tmp_path, uuids[0], [10.0, 20.0, 30.0]),
        _write_sample_parquet(tmp_path, uuids[1], [11.0, 21.0, 31.0]),
        _write_sample_parquet(tmp_path, uuids[2], [12.0, 22.0, 32.0]),
    ]
    sheet = pl.DataFrame({
        "file_name": [f"{u}{STAR_SUFFIX}" for u in uuids],
        "sample_id": ["TCGA-44-0001-01A", "TCGA-44-0002-01A", "TCGA-44-0003-11A"],
    })
    return paths, sheet


def test_macierz_ma_geny_w_wierszach_probki_w_kolumnach(star_matrix_inputs):
    paths, sheet = star_matrix_inputs
    matrix = build_expression_matrix(paths, sheet, metric="tpm_unstranded")
    assert matrix.columns[0] == "gene_id"
    assert matrix.height == len(GENES)
    assert set(matrix.columns[1:]) == set(sheet["sample_id"].to_list())


def test_wartosci_trafiaja_pod_wlasciwa_probke(star_matrix_inputs):
    paths, sheet = star_matrix_inputs
    matrix = build_expression_matrix(paths, sheet, metric="tpm_unstranded")
    assert matrix["TCGA-44-0001-01A"].to_list() == [10.0, 20.0, 30.0]


def test_niedozwolona_metryka_zglasza_blad(star_matrix_inputs):
    paths, sheet = star_matrix_inputs
    with pytest.raises(ExpressionMatrixError):
        build_expression_matrix(paths, sheet, metric="nie_ma_takiej")


def test_filtr_biotype_zostawia_tylko_pasujace(star_matrix_inputs):
    paths, sheet = star_matrix_inputs
    matrix = build_expression_matrix(paths, sheet, metric="tpm_unstranded",
                                     biotype_filter="protein_coding")
    assert matrix.height == 2  # dwa protein_coding, lncRNA odpada


def test_filtr_biotype_bez_trafien_zglasza_blad(star_matrix_inputs):
    paths, sheet = star_matrix_inputs
    with pytest.raises(ExpressionMatrixError):
        build_expression_matrix(paths, sheet, metric="tpm_unstranded",
                                biotype_filter="nie_istnieje")


def test_pusta_lista_plikow_zglasza_blad():
    with pytest.raises(ExpressionMatrixError):
        build_expression_matrix([], pl.DataFrame({"file_name": [], "sample_id": []}))


def test_brak_mapowania_pliku_zglasza_blad(star_matrix_inputs):
    paths, _ = star_matrix_inputs
    empty_sheet = pl.DataFrame({"file_name": ["inny.tsv"], "sample_id": ["X"]})
    with pytest.raises(ExpressionMatrixError):
        build_expression_matrix(paths, empty_sheet, metric="tpm_unstranded")
