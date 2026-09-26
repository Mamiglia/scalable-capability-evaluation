"""The hibayes models of the domain GLM (domain_glm.py): a Gamma GLM of squared forecast error with protocol x domain
deviations partially pooled per protocol, a random intercept per (target model, domain) cell (every protocol forecasts
the same target in the same domain, so the cell's distance from the reference mean is shared across its rows) and a
random intercept per judge within judge protocol. With each task's best judge only, the judge term has one level
("-", pinned to 0) and changes no estimate; its parameters stay so the sampler's draws are the verified run's.

Importing this module registers the two models with hibayes.
"""

import jax.numpy as jnp
import numpyro
from hibayes.model import Model, check_features, model
from hibayes.process import Features
from numpyro import distributions as dist


def builder(
    interaction: bool = True,
    prior_intercept_scale: float = 2.0,
    prior_effect_scale: float = 1.0,
    prior_sigma_scale: float = 0.5,
    prior_cell_scale: float = 1.0,
) -> Model:
    """log mu = intercept + protocol_effects[p] + domain_effects[d] + interaction[p, d] + cell_effects[c] + judge_effects[j];
    obs ~ Gamma(shape[p], shape[p] / mu), one shape per protocol. Protocol index 0 (the Baseline) is the reference
    level, pinned to 0; domain_effects sum to zero, so the intercept is the grand mean over domains;
    interaction[p, d] = sigma_protocol[p] * z, so a protocol's domain deviations share one spread (the
    (protocol | domain) term). interaction=False drops that term (the additive model of the LOO comparison)."""

    def statistical_model(features: Features) -> None:
        check_features(features, ["obs", "protocol_index", "num_protocol", "domain_index", "num_domain", "model_domain_index",
                                  "num_model_domain", "protocol_judge_index", "num_protocol_judge"])
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
def domain_glm(**kwargs) -> Model:
    """The reported fit: with the (protocol | domain) term."""
    return builder(interaction=True, **kwargs)


@model
def domain_glm_additive(**kwargs) -> Model:
    """protocol + domain only, no (protocol | domain) term: the LOO comparison's alternative."""
    return builder(interaction=False, **kwargs)
