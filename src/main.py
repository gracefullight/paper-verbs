from __future__ import annotations

"""Project entrypoint that delegates to the paper verbs CLI."""

import sys

from paper_verbs import main as paper_verbs_main


def main() -> None:
    """Run the paper verbs analyzer CLI (see `paper_verbs.main`)."""
    argv_backup = list(sys.argv)
    try:
        # Avoid pytest CLI flags leaking into argparse when imported.
        sys.argv = [argv_backup[0]]
        paper_verbs_main()
    finally:
        sys.argv = argv_backup


if __name__ == "__main__":
    main()
