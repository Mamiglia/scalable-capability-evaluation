"""
Where in a transcript the capability information sits: each protocol's squared error when a third of every transcript,
or one channel of it (reasoning, messages, tool calls), is removed or kept alone. paired_change holds the paired change
in squared error and its figures, thirds and channels compute each target's error per condition; the runner is
scripts/analysis/transcript_parts.py.
"""
