import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from eda_rolling_means import METRICS, compute_position_adjustment, position_bins


def test_position_bins_follow_short_context_boundaries():
    positions = np.array([0, 31, 32, 63, 64, 127, 128, 8191, 8192])
    assert position_bins(positions).tolist() == [0, 0, 1, 1, 2, 2, 3, 8, 9]


def test_streaming_position_adjustment_centres_each_position_bin(tmp_path):
    rows = []
    for model, shift in (("weak", -1.0), ("strong", 1.0)):
        for token_index, base in ((0, 2.0), (32, 4.0), (64, 6.0)):
            row = {
                "bench": "bench", "model": model, "item": "item", "epoch": 0,
                "log": "log", "sample_epoch": 0, "content_hash": model,
                "scorer": "llama", "token_index": token_index, "assistant": True,
            }
            for offset, metric in enumerate(METRICS):
                row[f"{metric}_bits"] = base + shift + offset
            rows.append(row)
    rows.append(rows[0] | {"assistant": False, "short_bits": 999.0})
    tokens = pd.DataFrame(rows)
    path = tmp_path / "tokens"
    pq.write_to_dataset(pa.Table.from_pandas(tokens, preserve_index=False), path,
                        partition_cols=["scorer", "bench"])

    normalised_tokens = tmp_path / "normalised-tokens"
    baselines, adjusted = compute_position_adjustment(
        path, batch_size=2, normalised_tokens=normalised_tokens)

    short = baselines[(baselines.scorer == "llama") &
                      (baselines.metric == "short")].sort_values("position_bin")
    assert short.position_bin.tolist() == [0, 1, 2]
    assert short.mean_bits.tolist() == [2.0, 4.0, 6.0]
    assert set(short.n_tokens) == {2}
    column = "surprisal_rolling_short_position_adjusted_mean__llama"
    got = adjusted.set_index("model")[column]
    assert got.to_dict() == {"weak": -1.0, "strong": 1.0}
    token_rows = pq.read_table(normalised_tokens).to_pandas()
    assert len(token_rows) == 6
    assert set(token_rows.position_bin) == {0, 1, 2}
    residuals = token_rows.groupby("model").short_position_adjusted_bits.mean()
    assert residuals.to_dict() == {"strong": 1.0, "weak": -1.0}
