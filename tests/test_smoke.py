# -*- coding: utf-8 -*-
"""Testy dymne: składnia, importy, rejestracja komend CLI, higiena pakietu."""
from __future__ import annotations

import ast
import importlib
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.cli import app

ROOT = Path(__file__).resolve().parent.parent
MODULES = [
    "src.analysis.survival_report",
    "src.analysis.expression_report",
    "src.analysis.readiness_report",
    "src.analysis.ml_export",
    "src.transform.survival_dataset",
    "src.transform.expression_matrix",
    "src.ingest.clinical_parser",
    "src.ingest.star_parser",
    "src.validate.runner",
    "src.cli",
]
EXPECTED_COMMANDS = {
    "download", "parse-star", "build-matrix",
    "build-survival", "validate-cohort", "repair-clinical",
}


@pytest.mark.parametrize("module", MODULES)
def test_modul_sie_importuje(module):
    importlib.import_module(module)


def test_wszystkie_pliki_parsuja():
    errors = []
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "app").rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            errors.append(f"{path}: {exc}")
    assert not errors, errors


def test_komendy_cli_zarejestrowane():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    registered = {c.name for c in app.registered_commands}
    assert EXPECTED_COMMANDS <= registered


def test_skrypty_pakietu_wskazuja_istniejace_moduly():
    # regresja: po usunięciu TUI został wpis luad-huba-tui = "tui.app:main"
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for name, target in config["project"].get("scripts", {}).items():
        module = target.split(":")[0]
        assert importlib.util.find_spec(module) is not None, f"{name} -> brak modułu {module}"


def test_brak_pozostalosci_po_tui():
    assert not (ROOT / "tui").exists()
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = " ".join(config["project"]["dependencies"]).lower()
    assert "textual" not in deps
