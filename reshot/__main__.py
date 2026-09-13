"""`python -m reshot …` — the same CLI as the `reshot` command, for environments where the
console script is not on PATH (a notebook, a venv that was not activated)."""

from .cli import main

raise SystemExit(main())
