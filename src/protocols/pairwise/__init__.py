"""Pairwise protocols: Bradley-Terry over verdicts between the task's models, anchored on the references.

    bradley_terry.py  the fit (rank), with anchors
    protocol.py       PairwiseBT — fit keeps the references, predict places the targets among them
    judge.py          JudgeBT — an LLM judge's verdicts on every (target, reference) and (reference, reference) pair, judged on demand
    calibrated.py     JudgeCalibrated — the same verdicts, calibrated BT + held-out-error shrinkage to the frontier prior
    linear.py         JudgeLinear — the same verdicts, per-bench linear calibration of pair logits on the reference pairs (WLS)
"""

from protocols.pairwise.judge import JudgeBT, from_logs
from protocols.pairwise.protocol import PairwiseBT

__all__ = ["JudgeBT", "PairwiseBT", "from_logs"]
