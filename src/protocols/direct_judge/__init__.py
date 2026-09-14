"""Direct protocols: transcript ratings linearly calibrated to reference capability.

    scoring.py   mean ratings and linear calibration on reference theta*
    protocol.py  DirectRegression — fit keeps references, predict calibrates ratings
    judge.py     JudgeDirect — an LLM judge's ratings, collected on demand
"""

from protocols.direct_judge.judge import JudgeDirect, from_logs
from protocols.direct_judge.protocol import DirectRegression

__all__ = ["JudgeDirect", "DirectRegression", "from_logs"]
