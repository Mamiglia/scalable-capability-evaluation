"""Direct protocols: transcript ratings linearly calibrated to training capability.

    scoring.py   mean ratings and linear calibration on references + historical theta*
    protocol.py  DirectRegression — fit keeps task.train, predict calibrates ratings
    judge.py     JudgeDirect — an LLM judge's ratings, collected on demand
"""

from protocols.direct_judge.judge import JudgeDirect, from_logs
from protocols.direct_judge.protocol import DirectRegression

__all__ = ["JudgeDirect", "DirectRegression", "from_logs"]
