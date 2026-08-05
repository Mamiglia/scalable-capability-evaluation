import json
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from configs import ModelConfig
from model import Model
from model import runpod


def test_plain_model_arguments():
    config = ModelConfig(
        name="org/model",
        slug="openai/org/model",
        model_args={"base_url": "https://example.test/v1", "temperature": 0.5},
    )

    with Model(config) as model:
        assert model.eval_kwargs() == {
            "model": "openai/org/model",
            "model_args": {"temperature": 0.5},
            "model_base_url": "https://example.test/v1",
        }


def test_runpod_model_lifecycle(monkeypatch):
    config = ModelConfig(
        name="org/model",
        slug="openrouter/org/model",
        model_args={"provider": {}},
        runpod={"use": True, "model": "org/model", "gpu_id": "NVIDIA A40"},
    )
    events = []
    monkeypatch.setattr(runpod, "create", lambda cfg: events.append("create") or "pod1")
    monkeypatch.setattr(
        runpod,
        "wait",
        lambda pod_id, port, timeout: events.append("wait")
        or "https://pod1-8000.proxy.runpod.net/v1",
    )
    monkeypatch.setattr(runpod, "terminate", lambda pod_id: events.append("terminate"))

    with pytest.raises(RuntimeError, match="eval failed"):
        with Model(config) as model:
            assert model.eval_kwargs() == {
                "model": "vllm/org/model",
                "model_args": {},
                "model_base_url": "https://pod1-8000.proxy.runpod.net/v1",
            }
            raise RuntimeError("eval failed")

    assert events == ["create", "wait", "terminate"]
    assert config.slug == "openrouter/org/model"
    assert config.model_args == {"provider": {}}


def test_startup_failure_terminates_runpod(monkeypatch):
    config = ModelConfig(
        name="org/model",
        runpod={"use": True, "model": "org/model", "gpu_id": "NVIDIA A40"},
    )
    terminated = []
    monkeypatch.setattr(runpod, "create", lambda cfg: "pod1")
    monkeypatch.setattr(
        runpod, "wait", MagicMock(side_effect=TimeoutError("not ready"))
    )
    monkeypatch.setattr(runpod, "terminate", terminated.append)

    with pytest.raises(TimeoutError, match="not ready"), Model(config):
        pass
    assert terminated == ["pod1"]


def test_threads_share_one_runpod(monkeypatch):
    config = ModelConfig(
        name="org/shared-model",
        runpod={
            "use": True,
            "model": "org/shared-model",
            "gpu_id": "NVIDIA A40",
        },
    )
    barrier = threading.Barrier(2)
    created = []
    terminated = []
    monkeypatch.setattr(
        runpod, "create", lambda cfg: created.append(cfg) or "shared-pod"
    )
    monkeypatch.setattr(runpod, "wait", lambda pod_id, port, timeout: "shared-url")
    monkeypatch.setattr(runpod, "terminate", terminated.append)

    def evaluate(_index):
        with Model(config) as model:
            barrier.wait()
            return model.eval_kwargs()["model_base_url"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(evaluate, range(2))) == ["shared-url", "shared-url"]

    assert len(created) == 1
    assert terminated == ["shared-pod"]


def test_readiness_retries_transient_http_error(monkeypatch):
    ready = MagicMock()
    ready.__enter__.return_value.status = 200
    monkeypatch.setattr(
        runpod.urllib.request,
        "urlopen",
        MagicMock(
            side_effect=[
                urllib.error.HTTPError("url", 403, "Forbidden", None, None),
                ready,
            ]
        ),
    )
    monkeypatch.setattr(runpod.time, "sleep", lambda seconds: None)

    assert runpod.wait("pod1", 8000, 30) == ("https://pod1-8000.proxy.runpod.net/v1")


def test_runpod_command(monkeypatch):
    captured = []
    monkeypatch.setattr(
        runpod,
        "_run",
        lambda command: captured.append(command)
        or MagicMock(stdout=json.dumps({"id": "pod1"})),
    )
    config = runpod.validate(
        {
            "model": "org/model",
            "gpu_id": "NVIDIA A40",
            "vllm_args": ["--max-model-len", 8192],
        }
    )

    assert runpod.create(config) == "pod1"
    assert captured[0][:3] == ["runpodctl", "pod", "create"]
    assert "--container-disk-in-gb=20" in captured[0]
    assert (
        "--docker-args=--model org/model --enable-prefix-caching --max-model-len 8192"
        in captured[0]
    )
