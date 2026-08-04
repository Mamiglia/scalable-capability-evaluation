"""Tests for integrated Runpod collection support."""

import json
import subprocess
from unittest.mock import MagicMock

import pytest

from collect.runpod import main as runpod_main
from collect.runpod import utils
from configs import ModelConfig


def model_config(*, use=False, runpod=True):
    config = None
    if runpod:
        config = {
            "use": use,
            "model": "org/model",
            "gpu_id": "NVIDIA A40",
        }
    return ModelConfig(
        name="org/model",
        slug="openrouter/org/model",
        model_args={"provider": {}},
        runpod=config,
    )


def test_validate_defaults_and_requires_boolean_use():
    config = utils.validate({"model": "org/model", "gpu_id": "NVIDIA A40"})
    assert config["use"] is False
    assert config["cloud_type"] == "COMMUNITY"
    assert config["container_disk_in_gb"] == 20
    assert config["vllm_args"] == []

    with pytest.raises(ValueError, match="runpod.use must be a Boolean"):
        utils.validate({"use": "true", "model": "org/model", "gpu_id": "NVIDIA A40"})


def test_termination_deadline_converts_duration_to_rfc3339():
    value = utils.termination_deadline("4h")
    assert value.endswith("Z")
    assert "T" in value
    assert utils.termination_deadline("2026-08-04T00:00:00Z") == "2026-08-04T00:00:00Z"


def test_run_creates_pod_with_defaults_and_extra_vllm_args(monkeypatch):
    commands = []

    def fake_command(command):
        commands.append(command)
        return MagicMock(stdout=json.dumps({"id": "pod123"}))

    monkeypatch.setattr(utils, "run_command", fake_command)
    pod_id = utils.run(
        {
            "model": "org/model",
            "gpu_id": "NVIDIA A100 80GB PCIe",
            "vllm_args": ["--tensor-parallel-size", 2],
        }
    )

    assert pod_id == "pod123"
    command = commands[0]
    assert command[:3] == ["runpodctl", "pod", "create"]
    assert "--cloud-type=COMMUNITY" in command
    assert "--gpu-id=NVIDIA A100 80GB PCIe" in command
    assert "--gpu-count=1" in command
    assert "--container-disk-in-gb=20" in command
    assert (
        "--docker-args=--model org/model --enable-prefix-caching "
        "--tensor-parallel-size 2"
    ) in command


def test_run_command_bubbles_up_cli_error(monkeypatch):
    error = subprocess.CalledProcessError(
        1, ["runpodctl"], stderr='{"error":"invalid gpu"}'
    )
    monkeypatch.setattr(utils.subprocess, "run", MagicMock(side_effect=error))

    with pytest.raises(utils.RunpodCommandError, match="invalid gpu"):
        utils.run_command(["runpodctl", "pod", "create"])


def test_disabled_model_runs_without_runpod(monkeypatch):
    model = model_config(use=False)
    collected = []
    monkeypatch.setattr(utils, "run", MagicMock())

    assert runpod_main([model], lambda value: collected.append(value) or 0) is None
    assert collected == []
    utils.run.assert_not_called()


def test_yaml_use_true_runs_waits_collects_and_terminates(monkeypatch):
    model = model_config(use=True)
    events = []
    monkeypatch.setattr(utils, "run", lambda config: events.append("run") or "pod123")
    monkeypatch.setattr(
        utils,
        "wait",
        lambda pod_id, port, timeout: events.append("wait")
        or "https://pod123-8000.proxy.runpod.net/v1",
    )
    monkeypatch.setattr(utils, "terminate", lambda pod_id: events.append("terminate"))

    def collect(value):
        events.append("collect")
        assert value.slug == "openai/org/model"
        assert value.model_args["base_url"].endswith("/v1")
        assert "provider" not in value.model_args
        return 0

    assert runpod_main([model], collect) == 0
    assert events == ["run", "wait", "collect", "terminate"]
    assert model.slug == "openrouter/org/model"
    assert model.model_args == {"provider": {}}


def test_force_requires_runpod_mapping():
    with pytest.raises(ValueError, match="needs a runpod mapping"):
        runpod_main([model_config(runpod=False)], lambda model: 0, force=True)


def test_collection_error_bubbles_after_termination(monkeypatch):
    model = model_config(use=True)
    terminated = []
    monkeypatch.setattr(utils, "run", lambda config: "pod456")
    monkeypatch.setattr(
        utils, "wait", lambda pod_id, port, timeout: "https://example.test/v1"
    )
    monkeypatch.setattr(utils, "terminate", terminated.append)

    def fail(_model):
        raise RuntimeError("evaluation failed")

    with pytest.raises(RuntimeError, match="evaluation failed"):
        runpod_main([model], fail)
    assert terminated == ["pod456"]
