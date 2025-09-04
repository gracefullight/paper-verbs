import sys
from io import StringIO

import pytest

from main import main


def test_main_runs_cli_and_reports_no_pdfs(monkeypatch: pytest.MonkeyPatch) -> None:
    output = StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    main()
    assert "No PDFs found" in output.getvalue()
