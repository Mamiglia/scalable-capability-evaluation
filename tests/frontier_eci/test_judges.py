from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]


def load(path):
    return yaml.safe_load((ROOT / path).read_text())


def assert_first_party(spec, provider, *, require_parameters=True):
    route = spec["model_args"]["provider"]
    assert route["only"] == [provider]
    assert route["allow_fallbacks"] is False
    assert route["require_parameters"] is require_parameters
    assert route["data_collection"] == "deny"


def test_checkpoint_judges_are_exact_and_first_party():
    gpt4o = load("configs/models/openai/gpt-4o-1120.yml")
    assert gpt4o["slug"] == "openrouter/openai/gpt-4o-2024-11-20"
    assert_first_party(gpt4o, "openai")

    gemini = load("configs/models/google/gemini-2.5-pro.yml")
    assert gemini["slug"] == "openrouter/google/gemini-2.5-pro"
    assert_first_party(gemini, "google-ai-studio")

    gpt52 = load("configs/models/openai/gpt-5.2.yml")
    assert gpt52["slug"] == "openrouter/openai/gpt-5.2"
    assert_first_party(gpt52, "openai", require_parameters=False)


def test_bridge_judges_are_binding_and_private():
    o4 = load("configs/models/openai/o4-mini.yml")
    assert_first_party(o4, "openai")

    flash = load("configs/models/deepseek/deepseek-v4-flash-0731.yml")
    route = flash["model_args"]["provider"]
    assert route["only"] == ["deepinfra/fp8"]
    assert route["allow_fallbacks"] is False
    assert route["require_parameters"] is True
    assert route["data_collection"] == "deny"
