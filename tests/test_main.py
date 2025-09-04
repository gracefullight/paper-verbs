import sys
from io import StringIO

import pytest

import paper_verbs


def test_main_reports_no_pdfs_with_empty_assets(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    output = StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    # Point module asset dir to an empty temporary directory
    empty_dir = tmp_path / "empty_assets"
    empty_dir.mkdir()
    monkeypatch.setattr(paper_verbs, "ASSET_DIR", empty_dir, raising=True)
    paper_verbs.main()
    assert "No PDFs found" in output.getvalue()
