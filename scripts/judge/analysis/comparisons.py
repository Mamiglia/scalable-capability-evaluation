"""Analysis config: which pilot comparisons feed which analysis, plus model
metadata not derivable from logs.

Triplets are (judge, major, minor) shortnames matching configs/judge_pilot.yaml.
A comparison may appear under several keys. Digest keeps only runs specified
in configs/judge_pilot.yaml; analyses without a key here (run overview,
controls, calibration, length confound, item difficulty) use all of those.
"""

COMPARISONS = {
    # Arm A matrix across the dLogit range; feeds the sigmoid gap fit.
    "gap_ladder": [
        ("qwen3.5-122b-a10b", "gemini-3.1-flash-lite", "gemma-4-26b-a4b-it"),
        ("gemma-4-31b-it", "gemini-3.1-flash-lite", "gemma-4-26b-a4b-it"),
        ("qwen3.5-122b-a10b", "gemma-4-31b-it", "gemini-2.5-flash"),
        ("gemini-3.1-flash-lite", "gemma-4-31b-it", "gemini-2.5-flash"),
        ("gemma-4-31b-it", "qwen3.5-122b-a10b", "qwen3.6-35b-a3b"),
        ("gemini-3.1-flash-lite", "qwen3.5-122b-a10b", "qwen3.6-35b-a3b"),
        ("qwen3.5-122b-a10b", "gemma-4-31b-it", "gemma-4-26b-a4b-it"),
        ("gemini-3.1-flash-lite", "gemma-4-31b-it", "gemma-4-26b-a4b-it"),
        ("qwen3.5-122b-a10b", "qwen3.6-35b-a3b", "gemini-3.1-flash-lite"),
        ("gemma-4-31b-it", "qwen3.6-35b-a3b", "gemini-3.1-flash-lite"),
        ("qwen3.5-122b-a10b", "qwen3.6-35b-a3b", "qwen3-32b"),
        ("gemini-3.1-flash-lite", "qwen3.6-35b-a3b", "qwen3-32b"),
        ("qwen3.6-35b-a3b", "qwen3.5-122b-a10b", "qwen3-32b"),
        ("gemma-4-31b-it", "qwen3.5-122b-a10b", "qwen3-32b"),
    ],
    # Same model on both sides; judge accuracy must sit at chance.
    "self_pairs": [
        ("qwen3.5-122b-a10b", "gemma-4-26b-a4b-it", "gemma-4-26b-a4b-it"),
        ("gemini-3.1-flash-lite", "gemma-4-26b-a4b-it", "gemma-4-26b-a4b-it"),
        ("qwen3.5-122b-a10b", "gemini-2.5-flash", "gemini-2.5-flash"),
        ("gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash"),
    ],
    # Shared pairs judged by judges of spread ability.
    "judge_strength": [
        # qwen3.5-122b vs qwen3.6-35b: judges .775 / .725 / .657
        ("gemma-4-31b-it", "qwen3.5-122b-a10b", "qwen3.6-35b-a3b"),
        ("gemini-3.1-flash-lite", "qwen3.5-122b-a10b", "qwen3.6-35b-a3b"),
        ("qwen3-32b", "qwen3.5-122b-a10b", "qwen3.6-35b-a3b"),
        # gemma-31b vs gemma-26b: judges .852 / .725 / .657
        ("qwen3.5-122b-a10b", "gemma-4-31b-it", "gemma-4-26b-a4b-it"),
        ("gemini-3.1-flash-lite", "gemma-4-31b-it", "gemma-4-26b-a4b-it"),
        ("qwen3-32b", "gemma-4-31b-it", "gemma-4-26b-a4b-it"),
        # qwen3.5-122b vs qwen3-32b: judges .824 / .775
        ("qwen3.6-35b-a3b", "qwen3.5-122b-a10b", "qwen3-32b"),
        ("gemma-4-31b-it", "qwen3.5-122b-a10b", "qwen3-32b"),
    ],
    # Judge weaker than both pair members.
    "weak_judge": [
        ("qwen3-32b", "qwen3.5-122b-a10b", "qwen3.6-35b-a3b"),
        ("qwen3-32b", "gemma-4-31b-it", "gemma-4-26b-a4b-it"),
    ],
    # Widest-gap pairs (dLogit ~2.5-2.9, beyond the ladder's top rung): positive
    # control -- judges must ace these -- and first Arm B transcripts under our judges.
    "wide_gap": [
        ("gemini-3.1-flash-lite", "gemini-3.5-flash", "gpt-4o"),
        ("qwen3.5-122b-a10b", "gemini-3.5-flash", "gpt-4o"),
        ("gemini-3.1-flash-lite", "gemini-3.5-flash", "llama-3.1-70b"),
        ("qwen3.5-122b-a10b", "gemini-3.5-flash", "llama-3.1-70b"),
    ],
    # Era rungs (2024 / 2025 / 2026 pairs) x judges of different eras.
    "temporal": [
        ("gpt-4o", "claude-3-5-sonnet-0620", "llama-3.1-70b"),
        ("gpt-4o", "gpt-4.1", "mistral-medium"),
        ("gpt-4o", "gpt-5.5-low", "qwen3.6-plus"),
        ("gemini-2.5-flash", "claude-3-5-sonnet-0620", "llama-3.1-70b"),
        ("gemini-2.5-flash", "gpt-4.1", "mistral-medium"),
        ("gemini-2.5-flash", "gpt-5.5-low", "qwen3.6-plus"),
        ("gemini-3.1-flash-lite", "claude-3-5-sonnet-0620", "llama-3.1-70b"),
        ("gemini-3.1-flash-lite", "gpt-4.1", "mistral-medium"),
        ("gemini-3.1-flash-lite", "gpt-5.5-low", "qwen3.6-plus"),
    ],
}

# Judge shortnames in log paths come from the OpenRouter id and can differ from
# the transcript shortname; digest maps them back so one namespace holds everywhere.
JUDGE_ALIASES = {
    "gpt-4o-2024-08-06": "gpt-4o",
}

# release: "YYYY-MM", None = not yet verified (run-plan open item: check log headers).
MODELS = {
    "gemma-4-31b-it": {"label": "Gemma-4-31B", "release": "2026-04"},
    "gemma-4-26b-a4b-it": {"label": "Gemma-4-26B", "release": "2026-04"},
    "gemini-3.1-flash-lite": {"label": "Flash-Lite 3.1", "release": "2026-03"},
    "gemini-2.5-flash": {"label": "Gemini 2.5 Flash", "release": "2025-06"},
    "qwen3.6-35b-a3b": {"label": "Qwen3.6-35B", "release": None},
    "qwen3.5-122b-a10b": {"label": "Qwen3.5-122B", "release": None},
    "qwen3-32b": {"label": "Qwen3-32B", "release": "2025-04"},
    "gemini-3.5-flash": {"label": "Gemini 3.5 Flash (high)", "release": None},
    "gpt-4o": {"label": "GPT-4o", "release": "2024-08"},
    "llama-3.1-70b": {"label": "Llama-3.1-70B", "release": "2024-07"},
    "claude-3-5-sonnet-0620": {"label": "Claude 3.5 Sonnet (06/24)", "release": "2024-06"},
    "gpt-4.1": {"label": "GPT-4.1", "release": "2025-04"},
    "mistral-medium": {"label": "Mistral Medium 2505", "release": "2025-05"},
    "gpt-5.5-low": {"label": "GPT-5.5 (low)", "release": None},
    "qwen3.6-plus": {"label": "Qwen3.6-Plus", "release": None},
}
