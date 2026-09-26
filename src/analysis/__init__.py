"""The paper's analyses: every protocol's forecast of every target against the Baseline (the references' mean theta*).
scripts/analysis/run.py runs them in order, with the settings in scripts/analysis/config.yaml.

Shared machinery:
    config            loads config.yaml and points the data readers at its paths
    loading           tasks from the frozen tables; verdict and rating tables from logs/
    matching          judge tables restricted to the frozen train items
    judges            verdicts and ratings to theta-hat; the reference leave-one-out error that picks a judge
    baselines         the Baseline and the accuracy oracle
    calibration       every protocol's forecast of a task (the forecasts table)
    stats             the core condition, best_judges (the judge each task uses), per-target errors
    likelihood_ratio  each target's log likelihood ratio against the Baseline, and its t-tests
    plots             figure style shared by every group

One subpackage per question the paper asks:
    capability_signal  do saturated transcripts carry capability information?
    transcript_parts   where in the transcript is it?
    generalisation     does it hold beyond the reference models?
    sensitivity        how much do the analysis choices matter?
"""
