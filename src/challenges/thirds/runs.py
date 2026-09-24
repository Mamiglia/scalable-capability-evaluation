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
# the judge with the lowest reference leave-one-out MSE (pairwise linear, reasoning shown, full train set where
# collected; 2026-09-23 refit after merging the judging repo), where it differs from RUNS; science 2026-I is the
# 40-item run (minimax-m2 also leads on 100 items). compare.py plots these.
LOO = RUNS | {("science", "2026-I"): "minimax-m2", ("math", "2026-I"): "kimi-k2-thinking-full",
              ("coding", "2026-I"): "kimi-k2-thinking-full", ("cyber", "2026-I"): "kimi-k2-thinking-full"}


def harnesses(domain: str, semester: str, stem: str | None = None) -> tuple[DomainConfig, dict[str, JudgeBT]]:
    """condition ("full" or "thirds-<kept>") -> the JudgeBT whose log_dir(semester) holds its verdicts;
    `stem` overrides the RUNS judge (e.g. a reference-LOO pick)."""
    dom = DomainConfig.from_cli(Path(f"configs/domain/{domain}.yml"))
    cfg = JudgeConfig.from_cli(Path(f"configs/judges/reasoning/{stem or RUNS[domain, semester]}.yml"))
    return dom, {"full": JudgeBT(cfg, dom)} | {f"thirds-{k}": JudgeBT(replace(cfg, truncate=f"thirds-{k}"), dom) for k in KEPT}
