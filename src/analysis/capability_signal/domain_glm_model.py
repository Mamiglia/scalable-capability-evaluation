"""
The hibayes models of the domain GLM (domain_glm.py): a Gamma GLM of squared prediction error with protocol x domain
deviations partially pooled per protocol, a random intercept per (target model, domain) cell (every protocol predicts
the same target in the same domain, so the cell's distance from the references' mean is shared across its rows) and a
random intercept per judge within judge protocol. With each task's best judge only, the judge term has one level
("-", pinned to 0) and changes no estimate; its parameters are still sampled so that the sampler's random draws, and
so the reported numbers, stay those of the run they were checked on.

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
    """
    Builds the numpyro model of squared error: log mu = intercept + protocol_effects[p] + domain_effects[d] +
    interaction[p, d] + cell_effects[c] + judge_effects[j], and obs ~ Gamma(shape[p], shape[p] / mu) with one shape
    per protocol, so mu is the mean. Protocol index 0 (the Baseline) is the reference level, its effect, interaction
    and sigma_protocol pinned to 0; domain_effects sum to zero, so the intercept is the grand mean over domains;
    interaction[p, d] = sigma_protocol[p] * z, so a protocol's domain deviations share one spread (the
    (protocol | domain) term).

    Inputs:
        interaction: whether to include the (protocol | domain) term; False gives the additive model the
            leave-one-out comparison sets against it.
        prior_intercept_scale: the SD of the intercept's normal prior.
        prior_effect_scale: the SD of the protocol and domain effects' normal priors.
        prior_sigma_scale: the scale of the half-normal priors of sigma_protocol and sigma_judge.
        prior_cell_scale: the scale of the half-normal prior of sigma_cell.
    Outputs:
        the model function, which takes hibayes Features (obs, and the index and count of protocol, domain,
        model_domain and protocol_judge) and samples obs.
    """

    def statistical_model(features: Features) -> None:
        """The numpyro model over one set of features."""
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
    """
    The GLM with the (protocol | domain) term, the fit the ratios and contrasts come from; registered with hibayes as
    domain_glm.

    Inputs:
        kwargs: the prior scales, as builder's prior_* arguments.
    Outputs:
        the model function (builder with interaction=True).
    """
    return builder(interaction=True, **kwargs)


@model
def domain_glm_additive(**kwargs) -> Model:
    """
    The additive GLM, protocol + domain with no (protocol | domain) term, which the leave-one-out comparison sets
    against the full model; registered with hibayes as domain_glm_additive.

    Inputs:
        kwargs: the prior scales, as builder's prior_* arguments.
    Outputs:
        the model function (builder with interaction=False).
    """
    return builder(interaction=False, **kwargs)
