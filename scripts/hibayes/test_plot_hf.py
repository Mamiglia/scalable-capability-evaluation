"""The HF rerun produces the same six plot types as the main GLM."""

import numpy as np
import pandas as pd

try:
    from plot_hf import FIGURE_NAMES, make_plots
except ImportError:
    FIGURE_NAMES = ()
    make_plots = None


def test_plot_bundle_is_complete_and_keeps_svg_text_editable(tmp_path):
    assert make_plots is not None, "plot_hf.make_plots is not implemented"
    frame = pd.DataFrame({
        "chosen_left": [1, 0, 1, 0],
        "judge": ["a", "a", "b", "b"],
        "item": ["i1", "i2", "i1", "i2"],
        "transcript_condition": ["gpqa", "mbpp", "gpqa", "mbpp"],
        "length_difference_sd": [1.0, -1.0, 0.5, -0.5],
        "capability_difference_sd": [0.5, -0.5, 1.0, -1.0],
        "same_model_family": [0, 1, 0, 1],
    })
    data = tmp_path / "data.csv"
    posterior = tmp_path / "posterior.npz"
    frame.to_csv(data, index=False)
    shape = (2, 12)
    np.savez(
        posterior,
        position_effect_mean=np.zeros(shape),
        position_effect_scale=np.full(shape, 0.1),
        judge_position_effects_raw=np.zeros(shape + (2,)),
        predictor_effect_mean=np.broadcast_to([0.1, 0.5, -0.2], shape + (3,)),
        predictor_effect_scale=np.full(shape + (3,), 0.1),
        judge_predictor_effects_raw=np.zeros(shape + (2, 3)),
        judge_condition_effects_free=np.zeros(shape + (2, 1)),
        item_sigma=np.full(shape, 0.1),
        item_effects_raw=np.zeros(shape + (2,)),
    )

    make_plots(data, posterior, tmp_path / "plots", loo_batch=2)

    assert {path.stem for path in (tmp_path / "plots").glob("*.svg")} == set(FIGURE_NAMES)
    assert all("<text" in path.read_text() for path in (tmp_path / "plots").glob("*.svg"))
    loo_svg = (tmp_path / "plots" / "judge_loo_accuracy.svg").read_text().splitlines()
    assert any(">a</text>" in line and "rotate(-90" in line for line in loo_svg)
