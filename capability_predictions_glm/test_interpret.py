from dataclasses import replace

from capability_predictions_glm.config import CONFIG
from capability_predictions_glm.interpret import build_specifications, run_fixed_spec
from capability_predictions_glm.test_run import _inputs


def test_interpretation_specs_cover_lesions_singulars_and_annotation_removal() -> None:
    inputs = _inputs()

    specs = {spec.name: spec for spec in build_specifications(inputs.dictionary)}

    assert specs["no-annotations"].features == ("signal",)
    assert specs["lesion-feature-signal"].features == ("quality",)
    assert specs["singular-feature-signal"].features == ("signal",)
    assert specs["lesion-block-rubric"].features == ("signal",)
    assert specs["singular-family-annotations"].features == ("quality",)


def test_fixed_spec_runs_all_targets_without_retuning() -> None:
    inputs = _inputs()
    config = replace(CONFIG, model_folds=(("fold", ("m5",)),))
    spec = {spec.name: spec for spec in build_specifications(inputs.dictionary)}["no-annotations"]
    penalties = {("fold", target): (1.0, 0.1) for target in ("Q2", "Q3", "Q4")}

    result = run_fixed_spec(inputs, config, spec, penalties)

    assert set(result.direct_predictions.target_quartile) == {"Q2", "Q3", "Q4"}
    assert set(result.pair_item_predictions.target_quartile) == {"Q2", "Q3", "Q4"}
    assert result.direct_predictions.specification.eq("no-annotations").all()
    assert result.pair_item_predictions.specification.eq("no-annotations").all()
