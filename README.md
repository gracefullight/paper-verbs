# paper-verbs

Analyze English verb usage from PDFs. Place your papers under `src/assets/` and run the analyzer to extract verb lemma frequencies, simple verb phrases, and tense/voice distributions. Results are printed to console and saved as CSVs under `src/`.

## Quick Start

1) Ensure Python 3.12 is used (3.13 is not supported due to spaCy/NumPy ABI constraints):

```sh
uv python install 3.12     # once
uv sync                     # install dependencies + model
```

2) Put PDFs under `src/assets/` (recursive; `.pdf`/`.PDF` both supported).

3) Run the analyzer:

```sh
uv run python src/main.py
```

## What It Does

- Scans `src/assets/` for PDFs and extracts text (PyMuPDF).
- Removes trailing References/Bibliography/Acknowledgements heuristically.
- Uses spaCy (en_core_web_sm) to parse and collect:
  - Verb lemmas (AUX/modals excluded by default; e.g., can/would/have)
  - Simple verb phrases around each counted verb
  - Tense and voice distributions
- Prints the top 100 verb lemmas to console.
- Saves CSVs under `src/`:
  - `src/verbs.csv` with columns: `rank, verb, count`
  - `src/phrases.csv` with columns: `rank, verb_phrase, count`

Notes
- The lemma `preprint` is excluded from counts.
- No CLI flags: defaults are baked in. Just run the script.

## Development

Common tasks (via poe):

```sh
uv run poe run         # run app
uv run poe test        # tests (pytest)
uv run poe test-cov    # tests with coverage
uv run poe lint        # ruff check
uv run poe format      # ruff format
uv run poe type-check  # mypy (strict)
uv run poe all-checks  # all of the above
```

## Troubleshooting

- spaCy model not found / load fails:
  - This project pins and installs `en_core_web_sm` via `pyproject.toml`. Run `uv sync`.
  - Verify: `uv run python -c "import en_core_web_sm; en_core_web_sm.load(); print('OK')"`
- NumPy/Thinc/Blis errors (e.g., dtype size changed or build failures):
  - Use Python 3.12. Run `uv python install 3.12` then `uv sync --reinstall`.
- PDFs with only scanned images will have little/no text extracted. OCR is required to analyze such files.

## Requirements

- Python >=3.12,<3.13
- uv (install via `pip install uv`)

## Project Layout

- `src/main.py` – entrypoint that runs `paper_verbs.main()`
- `src/paper_verbs.py` – analyzer logic
- `src/assets/` – put your PDFs here
- `src/verbs.csv`, `src/phrases.csv` – outputs
- `tests/` – pytest suite
