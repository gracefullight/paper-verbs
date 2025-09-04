# Repository Guidelines

## Project Structure & Module Organization
- Source code lives in `src/`. Entry point: `src/main.py` exposing `main()`.
- Modules live under `src/<module>/` (e.g., `src/demo/`, `src/week1/`). With `pythonpath = ["src"]`, import as `import demo.main`.
- Tests live in `tests/` (e.g., `tests/test_main.py`, `tests/test_demo.py`).

## Build, Test, and Development Commands
- Run app: `uv run poe run` (alias for `python src/main.py`).
- Run a module: `uv run python src/demo/main.py`.
- Install deps: `uv sync`.
- Tests: `uv run poe test`.
- Coverage: `uv run poe test-cov` (uses `--cov=src`).
- Lint/Format: `uv run poe lint` / `uv run poe format`.
- Type check: `uv run poe type-check`.
- All checks: `uv run poe all-checks`.

## Coding Style & Naming Conventions
- Python 3.13 with strict type hints (`mypy` in strict mode).
- Formatter + linter: `ruff` (run `uv run poe format` before commits).
- Style: line length 100, double quotes, spaces for indent.
- Imports: sorted by Ruff (isort); first‑party package root is `src`.
- Naming: modules/files `snake_case.py`; functions/vars `snake_case`; classes `PascalCase`.

## Testing Guidelines
- Framework: `pytest` with coverage over `src/`.
- Location: tests in `tests/` as `test_*.py` with functions `test_*`.
- Run fast loop: `uv run pytest tests/`.
- Assertions allowed; tests ignore security rule `S101`.

## Commit & Pull Request Guidelines
- Commits follow Conventional Commits (e.g., `feat:`, `fix:`, `chore:`); subject <72 chars, imperative mood.
- Before pushing, ensure `uv run poe all-checks` passes.
- PRs: include purpose, concise summary of changes, relevant logs/output, and link issues (e.g., `Closes #123`). Request at least one review.

## Security & Configuration Tips
- Requirements: Python ≥3.13 and `uv` (`pip install uv`).
- Manage deps with `uv add <pkg>` / `uv remove <pkg>`; commit `pyproject.toml` and `uv.lock`.
- Do not commit secrets or large data; respect `.gitignore`.
