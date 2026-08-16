import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


PATH = Path(__file__).with_name("run_rewrites.py")
SPEC = importlib.util.spec_from_file_location("run_rewrites", PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_call_codex_can_pin_model_and_reasoning_effort(monkeypatch):
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_text("response")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    response, _ = runner.call_codex(
        "prompt", reasoning_effort="medium", model="gpt-5.6-terra"
    )

    assert response == "response"
    assert commands == [[
        "codex", "exec", "-s", "read-only", "--ignore-user-config",
        "-m", "gpt-5.6-terra", "-c", 'model_reasoning_effort="medium"',
        "-o", commands[0][-2], "prompt",
    ]]


def test_custom_instruction_three_shards_keep_stable_assignments_on_resume(tmp_path, monkeypatch):
    originals = tmp_path / "originals.jsonl"
    originals.write_text("".join(
        json.dumps({"model": "m", "item_id": f"i{i}", "transcript": f"text {i}",
                    "n_words": 2}) + "\n"
        for i in range(7)
    ))
    instruction = tmp_path / "custom.md"
    instruction.write_text("experiment notes\n---\nCUSTOM INSTRUCTION")
    outputs = [tmp_path / f"shard-{i}.jsonl" for i in range(3)]
    outputs[0].write_text(json.dumps(
        {"model": "m", "item_id": "i0", "rewritten": "already done"}
    ) + "\n")
    calls = []

    def fake_gpt(prompt, reasoning_effort="", model=""):
        calls.append((prompt, reasoning_effort, model))
        transcript = runner.re.search(r"<<<TRANSCRIPT 1>>>\n(.*?)\n<<<END 1>>>", prompt).group(1)
        return f"<<<REWRITE 1>>>\n{transcript}\n<<<END 1>>>", 0

    monkeypatch.setattr(runner, "ORIGINALS", originals)
    monkeypatch.setitem(runner.CALLERS, "gpt", fake_gpt)
    for index, output in enumerate(outputs):
        monkeypatch.setattr(sys, "argv", [
            str(PATH), "gpt", "--instruction", str(instruction),
            "--shard", f"{index}/3", "--out", str(output),
            "--model", "gpt-5.6-terra", "--reasoning-effort", "medium",
            "--workers", "1", "--batch-size", "1",
        ])
        runner.main()

    keys = [
        [(row["model"], row["item_id"]) for row in map(json.loads, output.read_text().splitlines())]
        for output in outputs
    ]
    assert keys == [
        [("m", "i0"), ("m", "i3"), ("m", "i6")],
        [("m", "i1"), ("m", "i4")],
        [("m", "i2"), ("m", "i5")],
    ]
    assert calls and all(
        prompt.startswith("CUSTOM INSTRUCTION")
        and effort == "medium" and model == "gpt-5.6-terra"
        for prompt, effort, model in calls
    )
