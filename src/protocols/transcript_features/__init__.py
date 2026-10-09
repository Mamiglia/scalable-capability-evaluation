"""Transcript statistics -> capability, calibrated on reference models.

features.py defines the length, regex, and RST-transition feature groups.
reader.py reads structured samples directly into feature tables.
protocol.py selects one of three nested feature sets for standardised ridge.
"""

from protocols.transcript_features.protocol import TranscriptRegression

__all__ = ["TranscriptRegression"]
