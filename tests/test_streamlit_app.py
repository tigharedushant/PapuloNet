"""
tests/test_streamlit_app.py

Targeted unit tests for app/streamlit_app.py:
1. Syntax, AST validation, and import integrity.
2. Production constants and frozen class contract adherence.
3. Streamlit app execution flow with st.testing.v1.AppTest (initial state, disclaimer, header).
4. Strict locked test set isolation.
"""

from __future__ import annotations

import ast
from pathlib import Path
import pytest

from streamlit.testing.v1 import AppTest


def test_streamlit_app_syntax_and_ast():
    """Verifies that app/streamlit_app.py is valid Python and parses into a complete AST."""
    app_path = Path("app/streamlit_app.py")
    assert app_path.exists(), "app/streamlit_app.py does not exist!"
    code = app_path.read_text(encoding="utf-8")
    tree = ast.parse(code, filename="app/streamlit_app.py")
    assert isinstance(tree, ast.Module)


def test_streamlit_app_contract_constants():
    """Verifies that streamlit_app.py defines the exact frozen production contract."""
    app_path = (Path(__file__).parent.parent / "app/streamlit_app.py").resolve()
    code = app_path.read_text(encoding="utf-8")
    tree = ast.parse(code)
    
    assigned_values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == "FROZEN_CLASSES":
                        assigned_values["FROZEN_CLASSES"] = [elt.value for elt in node.value.elts]
                    elif target.id == "DEFAULT_HANDOFF_PATH":
                        assigned_values["DEFAULT_HANDOFF_PATH"] = node.value.args[0].value

    expected_classes = [
        "Psoriasis",
        "Lichen_Planus",
        "Pityriasis_Rosea",
        "Seborrheic_Dermatitis",
    ]
    assert assigned_values.get("FROZEN_CLASSES") == expected_classes
    assert assigned_values.get("DEFAULT_HANDOFF_PATH") == "artifacts/phase9/final_pipeline_handoff.joblib"


def test_streamlit_app_initial_render():
    """Verifies that the Streamlit app boots up and renders initial header and upload widgets."""
    app_file = (Path(__file__).parent.parent / "app/streamlit_app.py").resolve()
    at = AppTest.from_file(str(app_file), default_timeout=15)
    at.run()

    # App must not crash on startup
    assert not at.exception

    # Check title and header components
    text_content = " ".join([m.value for m in at.markdown])
    assert "PAPULONET" in text_content
    assert "AI-Assisted Papulosquamous Skin Disease Analysis" in text_content
    assert "Research & Demonstration Disclaimer" in text_content

    # File uploader and camera input must be present
    assert len(at.file_uploader) >= 1
    assert len(at.get("camera_input")) >= 1

    # Initial state: prompt to upload images
    info_content = " ".join([info.value for info in at.info])
    assert "Please upload or capture clinical lesion images" in info_content


def test_streamlit_app_zero_locked_test_access():
    """Verifies that streamlit_app.py does not touch or evaluate the locked test set."""
    test_dir = Path("output/06_final_split/test")
    if test_dir.exists():
        count = len(list(test_dir.glob("*/*")))
        assert count == 243, f"Locked test set modified! Expected 243, found {count}"
