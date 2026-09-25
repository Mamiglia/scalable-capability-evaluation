"""The protocol and domain palette shared by the figures."""

PROTOCOL = {
    "Length": "#f1a48a",
    "Length+regex": "#eb6834",
    "Embedding": "#4a3aa7",
    "LLM Rater": "#1baf7a",
    "LLM Matchup": "#2a78d6",
}
# Triad around LLM Matchup, for its variants; TENUOUS for weak/greyed-out marks.
MATCHUP = {"Pairwise": "#2a78d6", "Mixed": "#d62a78", "Ref pairs": "#78d62a"}
TENUOUS = "#f8cbbc"
DOMAIN = {"SCIENCE": "o", "MATH": "s", "CODING": "^", "CYBER": "D"}
YMAX = 200  # % of ref-mean; anything above is drawn as a caret at the ceiling
