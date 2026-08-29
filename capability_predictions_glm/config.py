"""Configuration for the standalone capability-prediction analysis."""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAME = "readable-no-common-support-20260828"
SOURCE_MANIFEST_SHA256 = "2870a25d738c2e2a03b438ee4f35bd4d7e8986518a497a75de0f8aeda0eec0c6"
ANNOTATION_BLOCKS = frozenset({"rubric", "intermediate-errors"})
TARGET_QUARTILES = ("Q2", "Q3", "Q4")

MODEL_FOLDS = (
    ("model-fold-0", ("qwen3.8-2.4t-a95b", "qwen3.8-27b", "gpt-5", "glm-4.6",
                      "ling-2.6-flash", "gpt-4.1", "granite-4.1-8b")),
    ("model-fold-1", ("glm-5.3", "gemini-3.5-flash", "deepseek-v3.2",
                      "muse-glimmer-30b", "qwen3.5-4b", "mistral-medium-3",
                      "gemma-3-12b-it")),
    ("model-fold-2", ("kimi-k3", "kimi-k2.6",
                      "nvidia-nemotron-3-ultra-550b-a55b-bf16", "deepseek-r1-0528",
                      "nvidia-nemotron-3.5-lightning-30b-a3b-bf16", "qwen3-32b")),
    ("model-fold-3", ("claude-opus-4.7", "glm-5.2", "o3", "claude-sonnet-4.5",
                      "qwen3.5-9b", "gemma-4-e2b-it")),
)


@dataclass(frozen=True)
class AnalysisConfig:
    """Paths and small fitting grids; full execution is explicit."""

    input_root: Path = ROOT / "judge_results/capability_predictions_glm/no-common-support-inputs"
    output_root: Path = ROOT / "judge_results/capability_predictions_glm/readable-quartiles"
    model_folds: tuple[tuple[str, tuple[str, ...]], ...] = MODEL_FOLDS
    direct_elastic_alphas: tuple[float, ...] = (0.0001, 0.001, 0.01, 0.1)
    pair_elastic_cs: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    elastic_l1_ratios: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0)
    selection_min_fold_share: float = 2 / 3
    ridge_alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    logistic_cs: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    benchmark_deviation_penalty: float = 3.0
    inner_folds: int = 3
    target_clip: float = 0.01
    max_iter: int = 20000
    random_seed: int = 0


CONFIG = AnalysisConfig()
