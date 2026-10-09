"""
The analysis code: every protocol's prediction of every target's theta*, scored against the Baseline (the references'
mean theta*), and the analyses built on those predictions. scripts/analysis/run.py runs the analyses in order, with the
settings in scripts/analysis/config.yaml.

Shared modules:
    config            loads config.yaml and points the data readers at its paths
    loading           tasks from the frozen tables; verdict and rating tables from logs/
    matching          judge tables restricted to the frozen train items
    judges            verdicts and ratings to theta-hat; the reference leave-one-out error that picks a judge
    baselines         the Baseline and the accuracy oracle
    calibration       every protocol's prediction of every target of a task (the forecasts table)
    stats             the core condition, best_judges (the judge each task uses), per-target errors
    likelihood_ratio  each target's log likelihood ratio against the Baseline, and its t-tests
    plots             figure style shared by every group

One subpackage per group of analyses:
    capability_signal  whether transcripts of saturated items carry information about a model's capability
    transcript_parts   where in a transcript that information sits
    generalisation     whether the predictions hold beyond the conditions they are fitted in
    sensitivity        how much the results depend on choices made in the analysis
"""
