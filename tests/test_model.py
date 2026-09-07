import json
import os
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from configs import ModelConfig
from model import Model
from model import runpod
from configs.model import RunpodConfig


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
    monkeypatch.setattr(runpod, "create", lambda cfg: events.append("create") or ("pod1", "gpu"))
    monkeypatch.setattr(
        runpod,
        "wait",
        lambda pod_id, port, timeout, stall_timeout=0: events.append("wait")
        or "https://pod1-8000.proxy.runpod.net/v1",
    )
    monkeypatch.setattr(runpod, "terminate", lambda pod_id: events.append("terminate"))
    monkeypatch.setattr(runpod, "placement", lambda pod_id: {"machine": "m1", "datacenter": "EU-RO-1"})

    with pytest.raises(RuntimeError, match="eval failed"):
        with Model(config) as model:
            assert model.eval_kwargs() == {
                "model": "openai-api/runpod/org/model",
                "model_args": {"api_key": "inspectai", "stream": False},
                "model_base_url": "https://pod1-8000.proxy.runpod.net/v1",
                "max_retries": 2,
                "metadata": {"runpod_gpu": "gpu", "runpod_pod_id": "pod1",
                             "runpod_machine": "m1", "runpod_datacenter": "EU-RO-1"},
            }
            raise RuntimeError("eval failed")

    assert events == ["create", "wait", "terminate"]
    assert config.slug == "openrouter/org/model"
    assert config.model_args == {"provider": {}}


def test_generate_arguments():
    config = ModelConfig(
        name="org/model",
        slug="vllm/org/model",
        generate_args={"temperature": 1.0, "extra_body": {"min_p": 0.0}},
    )

    with Model(config) as model:
        assert model.eval_kwargs()["temperature"] == 1.0
        assert model.eval_kwargs()["extra_body"] == {"min_p": 0.0}


def test_startup_failure_terminates_runpod(monkeypatch):
    config = ModelConfig(
        name="org/model",
        runpod={"use": True, "model": "org/model", "gpu_id": "NVIDIA A40"},
    )
    terminated = []
    monkeypatch.setattr(runpod, "create", lambda cfg: ("pod1", "gpu"))
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
        runpod, "create", lambda cfg: created.append(cfg) or ("shared-pod", "gpu")
    )
    monkeypatch.setattr(runpod, "wait", lambda pod_id, port, timeout, stall_timeout=0: "shared-url")
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
    config = RunpodConfig(
        model="org/model", gpu_id="NVIDIA A40", vllm_args=["--max-model-len", 8192]
    )

    assert runpod.create(config) == ("pod1", "NVIDIA A40")
    assert captured[0][:3] == ["runpodctl", "pod", "create"]
    assert "--container-disk-in-gb=40" in captured[0]
    assert (
        "--docker-args=--model org/model --enable-prefix-caching --max-model-len 8192"
        in captured[0]
    )


def _worker_eval(config_dict):
    config = ModelConfig(
        name="org/mp-model",
        runpod=config_dict,
    )
    with Model(config) as model:
        return model.eval_kwargs()["model_base_url"]


def test_cross_process_runpod_sharing(monkeypatch):
    monkeypatch.setattr(runpod, "create", lambda cfg: ("shared-mp-pod", "gpu"))
    monkeypatch.setattr(runpod, "wait", lambda pod_id, port, timeout, stall_timeout=0: "http://shared-mp-url/v1")
    monkeypatch.setattr(runpod, "is_healthy", lambda base_url: True)
    monkeypatch.setattr(runpod, "terminate", lambda pod_id: None)

    config_dict = {
        "use": True,
        "model": "org/mp-model",
        "gpu_id": "NVIDIA A40",
    }

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(_worker_eval, [config_dict, config_dict]))

    assert results == ["http://shared-mp-url/v1", "http://shared-mp-url/v1"]



def test_gpu_fallback_is_logged_loudly(monkeypatch, caplog):
    """A substituted GPU changes price and comparability, so it must not pass silently."""
    attempts = []

    def fake_create(config, gpu_id):
        attempts.append(gpu_id)
        if gpu_id == "NVIDIA A40":
            raise RuntimeError("runpodctl failed: there are no longer any resources")
        return "pod1"

    monkeypatch.setattr(runpod, "_create_pod_once", fake_create)
    config = RunpodConfig(model="org/model", gpu_id="NVIDIA A40", fallback_gpus=["NVIDIA L40S"])

    with caplog.at_level("WARNING"):
        assert runpod.create(config) == ("pod1", "NVIDIA L40S")

    assert attempts == ["NVIDIA A40", "NVIDIA L40S"]
    assert "not the configured NVIDIA A40" in caplog.text


def test_gpu_fallback_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        runpod,
        "_create_pod_once",
        lambda config, gpu_id: (_ for _ in ()).throw(RuntimeError("failed to create pod")),
    )
    config = RunpodConfig(model="m", gpu_id="NVIDIA A40", fallback_gpus=[])

    with pytest.raises(RuntimeError, match="No capacity"):
        runpod.create(config)


def test_unexpected_create_error_is_not_retried(monkeypatch):
    calls = []

    def fake_create(config, gpu_id):
        calls.append(gpu_id)
        raise RuntimeError("runpodctl is not installed")

    monkeypatch.setattr(runpod, "_create_pod_once", fake_create)
    config = RunpodConfig(model="m", gpu_id="NVIDIA A40")

    with pytest.raises(RuntimeError, match="not installed"):
        runpod.create(config)
    assert calls == ["NVIDIA A40"]


def test_stalled_pod_is_replaced_once(monkeypatch):
    """A pod whose log went silent is a bad host: terminate it and create one more."""
    from model.session import RunpodSession

    events = []
    monkeypatch.setattr(runpod, "create", lambda cfg: events.append("create") or (f"pod{len(events)}", "gpu"))
    monkeypatch.setattr(runpod, "terminate", lambda pod_id: events.append(("terminate", pod_id)))
    waits = iter([runpod.Stalled("silent"), "http://good/v1"])

    def fake_wait(pod_id, port, timeout, stall_timeout=0):
        result = next(waits)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(runpod, "wait", fake_wait)
    monkeypatch.setattr(runpod, "diagnostics", lambda pod_id: events.append(("diagnostics", pod_id)) or "")
    session = RunpodSession("stall-key", RunpodConfig(model="m", gpu_id="g"))
    assert session.acquire() == "http://good/v1"
    assert events == ["create", ("diagnostics", "pod1"), ("terminate", "pod1"), "create"]


def test_wait_raises_stalled_on_silent_log(monkeypatch):
    import time as _time

    monkeypatch.setattr(runpod, "direct_url", lambda pod_id, port: None)
    monkeypatch.setattr(runpod, "is_healthy", lambda url, timeout=3: False)
    monkeypatch.setattr(runpod, "last_log_time", lambda pod_id: _time.time() - 1000)
    clock = [0.0]
    monkeypatch.setattr(runpod.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(runpod.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + 61))

    with pytest.raises(runpod.Stalled, match="no container log output"):
        runpod.wait("pod1", 8000, timeout=3600, stall_timeout=900)


def test_last_log_time_parses_runpod_lines(monkeypatch):
    lines = "\n".join([
        '{"source":"container","line":"Loading model from scratch...","ts":"2026-09-03T15:05:33.810646874Z"}',
        '{"source":"container","line":"[shm_broadcast.py:801] No available shared memory broadcast block","ts":"2026-09-03T16:00:00.000000000Z"}',
        "not json",
    ])
    monkeypatch.setattr(runpod, "_run", lambda cmd: MagicMock(stdout=lines))
    assert runpod.last_log_time("pod1") == 1788447933.810646
