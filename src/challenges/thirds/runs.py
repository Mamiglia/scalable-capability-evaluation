"""The thirds ablation's runs: one top judge per domain x semester, its baseline reasoning YAML, and the run dirs."""
from dataclasses import replace
from pathlib import Path

from configs.domain import DomainConfig
from configs.judge import JudgeConfig
from protocols.pairwise import JudgeBT

KEPT = ("1", "2", "3", "12", "13", "23")  # the six non-full subsets of the three character thirds
# (domain, semester) -> configs/judges/reasoning/<stem>.yml of the top judge's full-transcript baseline run
# (fit.yml top[0]; the -full twins and the 10-per-quartile kimi config match the baseline dir that exists).
RUNS = {("science", "2025-II"): "minimax-m1", ("coding", "2025-II"): "minimax-m1",
        ("math", "2025-II"): "deepseek-r1-0528", ("cyber", "2025-II"): "deepseek-r1-0528",
        ("science", "2026-I"): "kimi-k2-thinking", ("math", "2026-I"): "kimi-k2-thinking",
        ("coding", "2026-I"): "minimax-m2-full", ("cyber", "2026-I"): "deepseek-v3.2-full"}


def harnesses(domain: str, semester: str) -> tuple[DomainConfig, dict[str, JudgeBT]]:
    """condition ("full" or "thirds-<kept>") -> the JudgeBT whose log_dir(semester) holds its verdicts."""
    dom = DomainConfig.from_cli(Path(f"configs/domain/{domain}.yml"))
    cfg = JudgeConfig.from_cli(Path(f"configs/judges/reasoning/{RUNS[domain, semester]}.yml"))
    return dom, {"full": JudgeBT(cfg, dom)} | {f"thirds-{k}": JudgeBT(replace(cfg, truncate=f"thirds-{k}"), dom) for k in KEPT}
