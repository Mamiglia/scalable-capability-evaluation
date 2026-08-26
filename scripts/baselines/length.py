"""Deterministic length judges for the stronger-model task: pick-longer says the longer transcript
(assistant text, characters) is from the stronger model, pick-shorter the opposite; equal lengths
give 0.5. Nothing is fitted, so each held-out design is just its test pairs.

    uv run python scripts/baselines/length.py <output_dir>

Writes <output_dir>/stronger_model_from_pair/{pick-longer,pick-shorter}/<design>/ in the GLM fits'
layout (see pairs.py).
"""
import argparse
from pathlib import Path

import numpy as np

from pairs import designs, held_out, load_config, load_pairs, load_transcripts, transcript_rows, write_design


def pick_longer(pairs, transcripts, left, right):
    """P(left stronger) = 1 if left is longer, 0 if shorter, 0.5 if equal."""
    length = transcripts.text.str.len().to_numpy()
    return (np.sign(length[left] - length[right]) + 1) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    config = load_config()
    pairs, transcripts = load_pairs(config), load_transcripts(config)
    left, right = transcript_rows(pairs, transcripts, "left"), transcript_rows(pairs, transcripts, "right")
    longer = pick_longer(pairs, transcripts, left, right)
    for design in designs():
        test = held_out(design, pairs.bench, pairs.item, pairs.model_left, pairs.model_right)
        rows = pairs.loc[test, ["bench", "item", "model_left", "model_right"]].assign(observed=pairs.left_stronger[test])
        for name, predicted in (("pick-longer", longer[test]), ("pick-shorter", 1 - longer[test])):
            scores = write_design(args.output_dir, name, design, rows.assign(predicted=predicted))
            print(f"{design} {name}: accuracy {scores.accuracy.iloc[0]:.3f} on {len(rows)} held-out pairs")


if __name__ == "__main__":
    main()
