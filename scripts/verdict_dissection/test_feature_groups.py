import numpy as np
import pandas as pd

from feature_groups import build_groups, project


def test_duplicated_features_merge_and_project_to_one_named_column():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200), rng.normal(size=200)
    frame = pd.DataFrame({"split": "representation_train", "x": a, "x_copy": a + 0.01 * rng.normal(size=200),
                          "x_negative": -a, "y": b})
    groups, loadings = build_groups(frame, ["x", "x_copy", "x_negative", "y"], threshold=0.8)
    assert list(loadings.columns) == ["x", "y"]                   # named after the central feature
    assert groups.set_index("feature").group.to_dict() == {"x": "x", "x_copy": "x", "x_negative": "x", "y": "y"}
    projected = project(frame, loadings)
    assert np.corrcoef(projected.x, a)[0, 1] > 0.99               # sign-aligned to x
    assert np.allclose(projected.y, b)


def test_pair_projection_equals_difference_of_projections():
    loadings = pd.DataFrame([[0.6], [0.8]], index=["x", "y"], columns=["x"])
    left = pd.DataFrame({"x": [1.0], "y": [2.0]})
    right = pd.DataFrame({"x": [0.5], "y": [-1.0]})
    pair = pd.DataFrame({"d_x": left.x - right.x, "d_y": left.y - right.y})
    assert np.allclose(project(pair, loadings, "d_").d_x, project(left, loadings).x - project(right, loadings).x)
