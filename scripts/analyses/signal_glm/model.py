"""hibayes model: Gamma GLM of squared forecast error with protocol x domain deviations partially
pooled per protocol and a random intercept per (target model, domain) cell: every protocol
forecasts the same target in the same domain, so the cell's distance from the reference mean
is shared across its rows (the pairing unit of the paired tests), and a random intercept per
judge within judge protocol (level "-", the transcript protocols and the best-judge table, pinned to 0). See docs/superpowers/specs/2026-09-16-signal-glm-design.md."""

import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from hibayes.model import Model, check_features, model
from hibayes.process import Features


def signal_builder(
    interaction: bool = True,
    prior_intercept_scale: float = 2.0,
    prior_effect_scale: float = 1.0,
    prior_sigma_scale: float = 0.5,
    prior_cell_scale: float = 1.0,  # cell spread on the log scale; baseline errors span ~3 log units across cells
) -> Model:
    """log mu = intercept + protocol_effects[p] + domain_effects[d] + interaction_effects[p, d] + cell_effects[c] + judge_effects[j];
    obs ~ Gamma(shape[p], shape[p] / mu), one shape per protocol (a small consistent improvement and a
    large erratic one differ in shape). cell_effects[c] = sigma_cell * z, one per (target model, domain); judge_effects[j] = sigma_judge * z
    for j > 0, one per (judge protocol, judge). Index 0 of protocol (the saturated-item prediction)
    is the reference level, pinned to 0; domain_effects sum to zero, so the intercept is the grand
    mean over domains; interaction[p, d] = sigma_protocol[p] * z, so a protocol's four domain
    deviations share one estimated spread (the (protocol | domain) term). interaction=False drops
    that term (the additive model m1 of the LOO comparison; sigma_protocol is then all zeros)."""

    def statistical_model(features: Features) -> None:
        check_features(features, ["obs", "protocol_index", "num_protocol", "domain_index", "num_domain", "model_domain_index", "num_model_domain", "protocol_judge_index", "num_protocol_judge"])
        P, D, C, J = features["num_protocol"], features["num_domain"], features["num_model_domain"], features["num_protocol_judge"]
        p, d, c, j = features["protocol_index"], features["domain_index"], features["model_domain_index"], features["protocol_judge_index"]

        intercept = numpyro.sample("intercept", dist.Normal(0, prior_intercept_scale))
        protocol_free = numpyro.sample("protocol_free", dist.Normal(0, prior_effect_scale).expand([P - 1]).to_event(1))
        protocol_effects = numpyro.deterministic("protocol_effects", jnp.concatenate([jnp.zeros(1), protocol_free]))
        domain_free = numpyro.sample("domain_free", dist.Normal(0, prior_effect_scale).expand([D - 1]).to_event(1))
        domain_effects = numpyro.deterministic("domain_effects", jnp.concatenate([domain_free, -domain_free.sum(keepdims=True)]))

        if interaction:
            sigma_protocol = numpyro.sample("sigma_protocol_free", dist.HalfNormal(prior_sigma_scale).expand([P - 1]).to_event(1))
            z = numpyro.sample("interaction_z", dist.Normal(0, 1).expand([P - 1, D]).to_event(2))
            deviations = sigma_protocol[:, None] * z
        else:
            sigma_protocol, deviations = jnp.zeros(P - 1), jnp.zeros((P - 1, D))
        interaction_effects = numpyro.deterministic("interaction", jnp.concatenate([jnp.zeros((1, D)), deviations]))
        numpyro.deterministic("sigma_protocol", jnp.concatenate([jnp.zeros(1), sigma_protocol]))

        sigma_cell = numpyro.sample("sigma_cell", dist.HalfNormal(prior_cell_scale))
        cell_z = numpyro.sample("cell_z", dist.Normal(0, 1).expand([C]).to_event(1))
        cell_effects = numpyro.deterministic("cell_effects", sigma_cell * cell_z)

        sigma_judge = numpyro.sample("sigma_judge", dist.HalfNormal(prior_sigma_scale))
        judge_z = numpyro.sample("judge_z", dist.Normal(0, 1).expand([max(J - 1, 1)]).to_event(1))
        judge_effects = numpyro.deterministic("judge_effects", jnp.concatenate([jnp.zeros(1), sigma_judge * judge_z])[:J])

        shape = numpyro.sample("shape", dist.Exponential(1.0).expand([P]).to_event(1))
        log_mu = intercept + protocol_effects[p] + domain_effects[d] + interaction_effects[p, d] + cell_effects[c] + judge_effects[j]
        numpyro.sample("obs", dist.Gamma(shape[p], shape[p] / jnp.exp(log_mu)), obs=features["obs"])

    return statistical_model


@model
def signal_gamma_model(**kwargs) -> Model:
    """m2: with the (protocol | domain) term."""
    return signal_builder(interaction=True, **kwargs)


@model
def signal_gamma_additive(**kwargs) -> Model:
    """m1: protocol + domain only, no (protocol | domain) term."""
    return signal_builder(interaction=False, **kwargs)
