"""Tag each judge explanation with the reasons it cites, by keyword, and tabulate the mix per judge.

    uv run python scripts/verdict_dissection/tag_explanations.py

Reads judge_explanations.parquet (extract_judge_explanations.py) and writes
judge_results/verdict_dissection/explanation_reasons.parquet (one boolean column per reason per
explanation) and explanation_reason_mix.csv (share of explanations citing each reason, per judge x
benchmark). The reason list is a keyword taxonomy read off a sample of explanations; a reason is
"cited" when any of its patterns appears anywhere in the explanation, regardless of which side.
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "judge_results/verdict_dissection"

REASONS = {
    "detail": r"detailed|thorough|comprehensive|step[- ]by[- ]step|in[- ]depth|elaborat|extensive|fuller|more complete",
    "concision": r"concise|efficient|direct(?:ly)?\b|streamlined|succinct|brief|unnecessary|redundant|fewer steps|"
                 r"straightforward|wasted|laborious|verbose|minimal",
    "verification": r"verif|confirm|double[- ]check|checks?\b|checked|validat|sanity",
    "error_or_flaw": r"\berror|mistake|incorrect|flaw|misstat|wrong|inconsisten|fail|sloppy|confus|misinterpret",
    "explains_reasoning": r"explain|explanation|justif|reasoning|rationale|walks? through|shows? (?:its|the) work",
    "understanding": r"understanding|insight|grasp|deeper|nuanced|conceptual|sophisticat",
    "clarity": r"\bclear|clarity|organi[sz]ed|structured|readable|well[- ]",
    "no_explanation": r"without (?:any )?(?:explanation|reasoning|justification)|no (?:explanation|reasoning|working)|"
                      r"simply (?:provides|states|gives|outputs)|only (?:provides|gives|states) the answer|"
                      r"just (?:the|an) answer|bare answer|merely (?:states|gives)|lacks (?:any )?(?:explanation|reasoning)",
    "elimination": r"eliminat|distractor|other options|alternative|rule[sd]? out|each (?:answer )?(?:option|choice)|why .{0,40}(?:incorrect|wrong)",
    "rigour": r"rigor|precise|precision|formal|mathematical|systematic|careful",
    "tool_use": r"\btool|command|script|automat|bash|python|pipe|grep|shell",
    "context": r"context|background|broader|implication|physical(?:ly)? (?:insight|meaning|intuition)",
    "units_or_numbers": r"\bunits?\b|dimensional|numeric|calculation|comput",
    "formatting": r"format|markdown|bullet|headers?|latex|typeset|presentation",
    "hedging": r"hedg|uncertain|tentative|confident|assert",
}
PATTERNS = {name: re.compile(pattern, re.I) for name, pattern in REASONS.items()}


def main():
    table = pd.read_parquet(OUT / "judge_explanations.parquet", columns=["bench", "judge", "judge_log", "judge_sample", "judge_epoch", "explanation"])
    table = table[table.explanation != ""].copy()
    table["bench"] = table.bench.where(~table.bench.str.startswith("aime"), "aime")
    for name, pattern in PATTERNS.items():
        table[name] = table.explanation.map(lambda s: bool(pattern.search(s)))
    table.drop(columns="explanation").to_parquet(OUT / "explanation_reasons.parquet", index=False)
    mix = table.groupby(["judge", "bench"])[list(REASONS)].mean().round(3)
    mix.to_csv(OUT / "explanation_reason_mix.csv")
    print(mix.T.to_string())
    print("\nno reason tagged:", round(float((~table[list(REASONS)].any(axis=1)).mean()), 3))


if __name__ == "__main__":
    main()
