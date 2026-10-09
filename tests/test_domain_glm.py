"""Domain GLM (analysis.capability_signal.domain_glm) on synthetic inputs."""

import numpy as np
import pandas as pd
import pytest

from analysis import stats
from analysis.capability_signal import domain_glm, domain_glm_model


def test_rows_one_per_target_and_floored():
    f = pd.DataFrame([
        dict(domain="math", semester="2026-I", protocol=p, variant="-", judge="-", condition="-", judge_loo_mse=np.nan,
             model="m", theta_hat=h, theta=1.0)
        for p, h in (("easy_items", 1.0), ("glm_regex", 2.0), ("accuracy_hard", 2.0))
    ])
    rows = domain_glm.glm_rows(stats.target_errors(f, {}), 1e-6)
    assert list(rows.protocol) == ["easy_items", "glm_regex"]  # the oracle is not a protocol
    assert rows.y.tolist() == [1e-6, 1.0] and rows.task.iloc[0] == "math 2026-I"


def test_ratio_table_reads_posterior_arrays():
    coords = {"protocol": ["easy_items", "direct"], "domain": ["science", "math"]}
    posterior = {"protocol_effects": np.tile([[0.0, np.log(0.5)]], (2, 3, 1)),  # chains x draws x P
                 "interaction": np.tile([[[0.0, 0.0], [0.0, np.log(2.0)]]], (2, 3, 1, 1)),
                 "sigma_protocol": np.tile([[0.0, 0.3]], (2, 3, 1))}
    t = domain_glm.ratio_table(posterior, coords).set_index("domain")
    np.testing.assert_allclose(t.loc[["science", "math", "all"], "ratio"], [0.5, 1.0, 0.5])
    assert t.loc["all", "sigma_protocol"] == pytest.approx(0.3)
    wide = domain_glm.wide_table(domain_glm.ratio_table(posterior, coords))
    assert list(wide.index) == ["easy_items", "direct"] and wide.loc["easy_items", "math"] == 1.0


def test_model_traces_with_expected_shapes():
    import jax.numpy as jnp
    from numpyro import handlers
    features = {"obs": jnp.ones(6), "protocol_index": jnp.array([0, 1, 0, 1, 0, 1]), "num_protocol": 2,
                "domain_index": jnp.array([0, 0, 1, 1, 2, 2]), "num_domain": 3,
                "model_domain_index": jnp.array([0, 0, 1, 1, 2, 2]), "num_model_domain": 3,
                "protocol_judge_index": jnp.zeros(6, dtype=int), "num_protocol_judge": 1}
    trace = handlers.trace(handlers.seed(domain_glm_model.domain_glm(), 0)).get_trace(features)
    assert trace["interaction"]["value"].shape == (2, 3) and trace["protocol_effects"]["value"].shape == (2,)
    assert float(trace["interaction"]["value"][0].sum()) == 0.0  # the baseline row is pinned
    assert trace["obs"]["fn"].__class__.__name__ == "Gamma"
