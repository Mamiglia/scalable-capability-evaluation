"""Edit these defaults to configure the judge-verdict analysis module."""
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Self


ROOT: Path = Path(__file__).resolve().parents[2]  # Repository root used to make every default path absolute.

# Readable blocks used by EDA/PCA unless a command overrides ``--blocks``.
# Opaque/high-dimensional blocks (embedding-dense, transcript/clause SAEs,
# full clause topology and surprisal families) remain available but opt-in.
READABLE_BLOCKS: tuple[str, ...] = (
    "surface",              # Length, style, formatting and simple textual counts.
    "structure",            # Answer layout plus tool-call and tool-result trajectory counts.
    "rubric",               # Per-transcript LLM ratings of errors, rigour, clarity and efficiency.
    "exposition",           # Think-aloud, option-matching and labelled-exposition measures.
    "derivation",           # Equation, justification and explicit answer-derivation measures.
    "intermediate-errors",  # Quoted and typed slips in otherwise correct maths solutions.
    "clause-descriptors",   # Generic graph statistics computed from clause networks.
    "clause-motifs",        # Named clause-graph and tool-use motifs.
    "clause-plan",          # Whether an agent states and executes a plan.
)  # Dense embeddings, SAEs, surprisal and full topology stay opt-in.
JUDGES: tuple[str, ...] = (
    "gpt-oss-120b",            # Strong open-weight judge served through the pinned provider.
    "deepseek-r1-0528",        # Strong reasoning judge; no AIME rows in the current freeze.
    "o4-mini",                 # Strong OpenAI reasoning judge; no AIME rows currently.
    "gemma-4-31b-it",          # Strong Gemma judge.
    "deepseek-v4-flash-0731",  # Strong DeepSeek V4 Flash judge.
    "Llama-3.1-8B-Instruct",   # Weaker local judge with strong position dependence.
)  # Ling is excluded because its former judging endpoint is no longer available.


@dataclass(frozen=True)
class AnalysisConfig:
    design_matrix_config: Path = ROOT / "configs/design_matrices.yaml"  # Existing matrix/source-path config.
    results_dir: Path = ROOT / "judge_results/verdict_interpretability/readable-eda-20260828"  # Canonical output root.

    # Feature selection: empty regressors means every feature in feature_blocks.
    feature_blocks: tuple[str, ...] = READABLE_BLOCKS  # Matrix blocks loaded by default.
    regressors: tuple[str, ...] = ()  # Exact features to retain; empty keeps all selected blocks.
    interactions: tuple[tuple[str, str], ...] = ()  # GLM product terms; unused by EDA/PCA.

    judges: tuple[str, ...] = JUDGES  # Judge models retained in verdict analyses.
    benchmarks: tuple[str, ...] = ("gpqa", "aime", "intercode", "swe_bench_mini")  # Analysis strata.
    splits: tuple[str, ...] = ("representation_train",)  # Rows used for discovery and current plots.

    # Transcript eligibility. The current matrices already apply these rules;
    # future freeze loaders/GLMs must validate that their manifest agrees.
    canonical_only: bool = True  # Exclude superseded and noncanonical rollout logs.
    require_both_correct: bool = True  # Keep only pairs where both final answers are correct.
    min_reasoning_chars: int = 100  # Minimum characters before the trailing ANSWER line.
    max_transcript_chars: int = 262144  # Maximum visible characters per judged transcript.

    # Design-matrix preprocessing and EDA.
    correlation_method: str = "pearson"  # Correlation statistic used for readable-feature grouping.
    correlation_threshold: float = 0.8  # |r| cutoff for correlated pairs and feature groups.
    correlation_transitive: bool = False  # Require every cross-group pair to pass the cutoff.
    common_support: bool = True  # Retain features variable in every benchmark's support split.
    support_splits: tuple[str, ...] = ("representation_train",)  # Splits used for common-support checks.
    representative_rule: str = "shortest_name"  # Initial readable-feature representative policy.
    representative_overrides: tuple[tuple[str, str], ...] = ()  # (group ID, retained feature) exceptions.
    processed_aliases: tuple[tuple[str, str], ...] = ()  # (representative feature, processed name) aliases.
    max_correlation_features: int = 500  # Guard against accidental huge dense matrices.
    pca_components: int = 20  # Leading PCs retained in variance and loading tables.
    pca_common_support: bool = True  # Fit every benchmark PCA on features variable in every benchmark.
    pca_alignment: str = "stronger"  # Loading sign shown in plots: "stronger" or "judge".
    top_verdict_features: int = 12  # Rows displayed in each verdict-correlation heatmap.
    top_capability_features: int = 20  # Rows displayed in the benchmark-score heatmap.

    # Frequentist and resampling fits.
    fit_backends: tuple[str, ...] = ("frequentist", "hibayes")  # Joint-model backends to run.
    item_folds: int = 4  # Item-fold count for held-out new-items evaluations.
    model_folds: tuple[tuple[str, tuple[str, ...]], ...] = (
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
    )  # Strict new-model folds, copied here so fitting owns no cross-package defaults.
    inner_folds: int = 3  # Grouped folds used to tune each outer frequentist fit.
    elastic_net_cs: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)  # Elastic-net inverse penalties.
    elastic_net_l1_ratios: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)  # Elastic-net L1 shares.
    selection_min_fold_share: float = 2 / 3  # Inner-fold share required to retain a base feature.
    ridge_cs: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)  # Hierarchical-ridge inverse penalties.
    judge_deviation_penalty: float = 3.0  # Shrinkage penalty for judge slope deviations.
    benchmark_deviation_penalty: float = 3.0  # Shrinkage penalty for benchmark slope deviations.
    cell_deviation_penalty: float = 10.0  # Shrinkage penalty for judge-by-benchmark deviations.
    item_intercept_penalty: float = 10.0  # Shrinkage penalty for item intercepts.
    lesion_backends: tuple[str, ...] = ("frequentist",)  # Backends included in paired lesions.
    lesion_metric: str = "log_loss"  # Primary held-out lesion metric.
    glm_max_iter: int = 3_000  # Optimiser iteration ceiling for the forthcoming logistic GLM.
    glm_c: float = 0.1  # Inverse ridge penalty for the first frequentist GLM screen.
    bootstrap_iterations: int = 1_000  # Clustered resamples used for intervals.
    random_seed: int = 0  # Reproducible split, resampling and model seed.

    # Hierarchical GLM defaults, used when that runner is added.
    mcmc_warmup: int = 1_000  # Adaptation draws per hierarchical Bayesian chain.
    mcmc_samples: int = 1_000  # Retained posterior draws per Bayesian chain.
    mcmc_chains: int = 4  # Independent chains used for convergence checks.
    mcmc_method: str = "NUTS"  # HiBayes sampler method.
    mcmc_target_accept: float = 0.85  # NUTS target acceptance probability.
    mcmc_max_tree_depth: int = 10  # NUTS tree-depth ceiling.
    mcmc_init_strategy: str = "median"  # HiBayes parameter initialisation strategy.
    mcmc_chain_method: str = "parallel"  # NumPyro chain execution method.
    hibayes_population_prior_scale: float = 0.5  # Population intercept/slope Normal scale.
    hibayes_judge_prior_scale: float = 0.5  # HalfNormal scale for judge deviations.
    hibayes_benchmark_prior_scale: float = 0.5  # HalfNormal scale for benchmark deviations.
    hibayes_cell_prior_scale: float = 0.5  # HalfNormal scale for judge-by-benchmark deviations.
    hibayes_item_prior_scale: float = 0.5  # HalfNormal scale for nested item intercepts.

    @property
    def data_dir(self) -> Path:
        """Directory for numeric CSV outputs."""
        return self.results_dir / "data"

    @property
    def figure_dir(self) -> Path:
        """Directory for PNG and editable-SVG figures."""
        return self.results_dir / "figures"

    @property
    def basic_eda_dir(self) -> Path:
        """Directory for feature summaries and exact-duplicate reports."""
        return self.results_dir / "basic-eda"

    @property
    def processed_matrix_dir(self) -> Path:
        """Directory for the rebuilt readable-feature design matrix."""
        return self.results_dir / "design-matrix"

    @property
    def fit_dir(self) -> Path:
        """Directory for fitted-model artifacts and held-out predictions."""
        return self.results_dir / "fits"

    @property
    def lesion_dir(self) -> Path:
        """Directory for paired held-out feature-lesion results."""
        return self.results_dir / "lesions"

    @property
    def model_figure_dir(self) -> Path:
        """Directory for figures derived from saved fit and lesion artifacts."""
        return self.figure_dir / "models"

    def validate(self) -> Self:
        """Validate editable options early and return this immutable config."""
        if not self.feature_blocks:
            raise ValueError("feature_blocks cannot be empty")
        if not self.judges:
            raise ValueError("judges cannot be empty")
        if self.min_reasoning_chars < 0:
            raise ValueError("min_reasoning_chars must be non-negative")
        positive = {
            "max_transcript_chars": self.max_transcript_chars,
            "max_correlation_features": self.max_correlation_features,
            "pca_components": self.pca_components,
            "top_verdict_features": self.top_verdict_features,
            "top_capability_features": self.top_capability_features,
            "item_folds": self.item_folds,
            "inner_folds": self.inner_folds,
            "glm_max_iter": self.glm_max_iter,
            "glm_c": self.glm_c,
            "bootstrap_iterations": self.bootstrap_iterations,
            "mcmc_warmup": self.mcmc_warmup,
            "mcmc_samples": self.mcmc_samples,
            "mcmc_chains": self.mcmc_chains,
            "mcmc_max_tree_depth": self.mcmc_max_tree_depth,
            "hibayes_population_prior_scale": self.hibayes_population_prior_scale,
            "hibayes_judge_prior_scale": self.hibayes_judge_prior_scale,
            "hibayes_benchmark_prior_scale": self.hibayes_benchmark_prior_scale,
            "hibayes_cell_prior_scale": self.hibayes_cell_prior_scale,
            "hibayes_item_prior_scale": self.hibayes_item_prior_scale,
        }
        if invalid := [
            name for name, value in positive.items()
            if value <= 0 or (name.startswith("hibayes_") and not isfinite(value))
        ]:
            raise ValueError(f"{', '.join(invalid)} must be positive")
        if self.mcmc_method not in {"NUTS", "HMC"}:
            raise ValueError("mcmc_method must be 'NUTS' or 'HMC'")
        if not isfinite(self.mcmc_target_accept) or not 0 < self.mcmc_target_accept < 1:
            raise ValueError("mcmc_target_accept must be between 0 and 1")
        if self.mcmc_init_strategy not in {"median", "mean", "uniform"}:
            raise ValueError("mcmc_init_strategy must be 'median', 'mean', or 'uniform'")
        if self.mcmc_chain_method not in {"parallel", "sequential", "vectorized"}:
            raise ValueError("mcmc_chain_method must be 'parallel', 'sequential', or 'vectorized'")
        if not 0 <= self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be between 0 and 1")
        if self.correlation_method not in {"pearson", "spearman"}:
            raise ValueError("correlation_method must be 'pearson' or 'spearman'")
        if self.representative_rule not in {"shortest_name"}:
            raise ValueError("representative_rule must be 'shortest_name'")
        if not self.support_splits:
            raise ValueError("support_splits cannot be empty")
        if not self.fit_backends or not set(self.fit_backends) <= {"frequentist", "hibayes"}:
            raise ValueError("fit_backends must contain only 'frequentist' or 'hibayes'")
        if not self.lesion_backends or not set(self.lesion_backends) <= {"frequentist", "hibayes"}:
            raise ValueError("lesion_backends must contain only 'frequentist' or 'hibayes'")
        if self.lesion_metric not in {"log_loss", "accuracy"}:
            raise ValueError("lesion_metric must be 'log_loss' or 'accuracy'")
        grids = {
            "elastic_net_cs": self.elastic_net_cs,
            "elastic_net_l1_ratios": self.elastic_net_l1_ratios,
            "ridge_cs": self.ridge_cs,
        }
        if invalid := [name for name, values in grids.items() if not values or any(value <= 0 for value in values)]:
            raise ValueError(f"{', '.join(invalid)} must be non-empty and positive")
        if any(value > 1 for value in self.elastic_net_l1_ratios):
            raise ValueError("elastic_net_l1_ratios must be at most 1")
        if not 0 < self.selection_min_fold_share <= 1:
            raise ValueError("selection_min_fold_share must be between 0 and 1")
        penalties = {
            "judge_deviation_penalty": self.judge_deviation_penalty,
            "benchmark_deviation_penalty": self.benchmark_deviation_penalty,
            "cell_deviation_penalty": self.cell_deviation_penalty,
            "item_intercept_penalty": self.item_intercept_penalty,
        }
        if invalid := [name for name, value in penalties.items() if value <= 0]:
            raise ValueError(f"{', '.join(invalid)} must be positive")
        if not self.model_folds or any(not name or not members for name, members in self.model_folds):
            raise ValueError("model_folds must contain named non-empty model groups")
        fold_names = [name for name, _ in self.model_folds]
        models = [model for _, members in self.model_folds for model in members]
        if len(set(fold_names)) != len(fold_names) or len(set(models)) != len(models):
            raise ValueError("model_folds must have unique fold names and model membership")
        for name, pairs in {
            "representative_overrides": self.representative_overrides,
            "processed_aliases": self.processed_aliases,
        }.items():
            if any(len(pair) != 2 for pair in pairs) or len({pair[0] for pair in pairs}) != len(pairs):
                raise ValueError(f"{name} must contain unique key-value pairs")
        if self.pca_alignment not in {"stronger", "judge"}:
            raise ValueError("pca_alignment must be 'stronger' or 'judge'")
        if len(set(self.regressors)) != len(self.regressors):
            raise ValueError("regressors contains duplicates")
        if any(len(pair) != 2 for pair in self.interactions):
            raise ValueError("interactions must contain feature pairs")
        return self


CONFIG = AnalysisConfig().validate()
