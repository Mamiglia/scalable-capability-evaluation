"""Results analysis of the forecasting task: every protocol against the chance floor and the
accuracy oracle, per domain and semester (docs/superpowers/specs/2026-09-15-results-analysis-design.md).

One module per step, chained by the cell runner scripts/analyses/results.py:
    loading      semester tasks from the frozen tables; verdict and rating tables from logs/
    matching     restrict judge tables to the frozen train items and count what is unjudged
    baselines    chance floor, oracle and reference mean
    embeddings   ridge on mean transcript embeddings
    calibration  every signal -> theta-hat on the theta* scale (the forecasts table)
    stats        MSE per domain x semester, pooled rows, judge summaries, floor and oracle
    plots        the two results figures
    challenges/  truncation, hidden reasoning, ... reuse the same steps on their own logs
"""
