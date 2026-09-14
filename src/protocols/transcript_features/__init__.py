"""Transcript statistics -> capability, calibrated on reference models.

features.py defines the length, regex, and RST-transition feature groups.
reader.py reads structured samples directly into feature tables.
protocol.py selects one of three nested feature sets for standardised ridge.
The shared judging reader is unchanged. The RST/tool labeller and registry
integration are separate remaining steps.
"""

from protocols.transcript_features.protocol import TranscriptRegression

__all__ = ["TranscriptRegression"]
