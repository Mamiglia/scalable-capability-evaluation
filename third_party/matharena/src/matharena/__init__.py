"""Inspect tasks for MathArena's final-answer competitions.

`parser.py`, `parse_manual.py`, `competitions/` and `LICENSE` are symlinks into
the `vendor/matharena` git submodule (https://github.com/eth-sri/matharena,
MIT, Copyright (c) 2024 SRI Lab, ETH Zurich), pinned to a specific upstream
commit; the full upstream package is not installable (vllm, docker, playwright,
...). Re-sync by bumping the submodule.

`matharena.py` is ours: it registers one inspect task per final-answer competition.
"""

from pathlib import Path

if not (Path(__file__).parent / "parser.py").exists():
    raise ImportError(
        "matharena's vendored upstream sources are missing: the vendor/matharena "
        "git submodule is not checked out. Run `git submodule update --init` "
        "(or clone with --recurse-submodules) and retry."
    )

from matharena.matharena import (  # noqa: E402
    HF_REVISIONS,
    aime_2024_I,
    aime_2024_II,
    aime_2025,
    aime_2026,
    apex_2025,
    apex_shortlist,
    arxivmath,
    arxivmath_0126,
    arxivmath_0226,
    arxivmath_0326,
    arxivmath_0426,
    arxivmath_0526,
    arxivmath_1225,
    brumo_2025,
    cmimc_2025,
    competition_task,
    hmmt_feb_2024,
    hmmt_feb_2025,
    hmmt_feb_2026,
    hmmt_nov_2025,
    matharena_scorer,
    matharena_solver,
    smt_2025,
)

__all__ = [
    "HF_REVISIONS",
    "aime_2024_I",
    "aime_2024_II",
    "aime_2025",
    "aime_2026",
    "apex_2025",
    "apex_shortlist",
    "arxivmath",
    "arxivmath_0126",
    "arxivmath_0226",
    "arxivmath_0326",
    "arxivmath_0426",
    "arxivmath_0526",
    "arxivmath_1225",
    "brumo_2025",
    "cmimc_2025",
    "competition_task",
    "hmmt_feb_2024",
    "hmmt_feb_2025",
    "hmmt_feb_2026",
    "hmmt_nov_2025",
    "matharena_scorer",
    "matharena_solver",
    "smt_2025",
]
