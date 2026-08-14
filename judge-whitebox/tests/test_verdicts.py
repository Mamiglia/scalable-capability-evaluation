import json

import pandas as pd
import torch

import verdicts


class StubTokenizer:
    """Known single-token ids for "1"/"2"."""

    def encode(self, text, add_special_tokens=False):
        return {"1": [11], "2": [22]}[text]


def test_readout_picks_higher_logit():
    logits = torch.zeros(50)
    logits[11] = 3.0
    logits[22] = 1.0
    l1, l2 = verdicts.readout(logits, StubTokenizer())
    assert l1 == 3.0 and l2 == 1.0
    assert l1 > l2  # pick would be 1


class _FakeBatch(dict):
    def to(self, device):
        return self


class _StubRunTokenizer(StubTokenizer):
    """Adds the render/pad surface run() needs, no real model or chat template."""

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kw):
        return "PROMPT"

    def __call__(self, text):
        return {"input_ids": [1, 2, 3]}

    def pad(self, encs, return_tensors):
        return _FakeBatch(n=len(encs))


class _StubModel:
    """Always favors ANSWER token "1" (id 11) over "2" (id 22), so pick == 1 always."""

    def eval(self):
        pass

    def __call__(self, n):
        logits = torch.zeros(n, 1, 50)
        logits[:, 0, 11] = 5.0
        logits[:, 0, 22] = 1.0

        class Out:
            pass
        out = Out()
        out.logits = logits
        return out


def test_run_invert_scores_against_3_minus_label(tmp_path, monkeypatch):
    # no model/tokenizer loaded from disk: readout path is stubbed end to end
    monkeypatch.setattr(verdicts, "load_model",
                        lambda model_name, adapter, c: (_StubModel(), _StubRunTokenizer(), "cpu"))
    df = pd.DataFrame([
        dict(pair_uid="p1", order=0, label=1, question="Q", options_1="", options_2="",
             text_1="t1", text_2="t2"),
        dict(pair_uid="p2", order=0, label=2, question="Q", options_1="", options_2="",
             text_1="t1", text_2="t2"),
    ])
    monkeypatch.setattr(verdicts.wbpairs, "load_pairs", lambda c, split, limit: df)

    c = {"verdicts": {"max_input_tokens": 100, "batch_size": 2}}
    out_path = tmp_path / "out.jsonl"
    verdicts.run("dev", None, "eval", None, out_path, c, invert=True)

    rows = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(rows) == 2
    by_uid = {r["pair_uid"]: r for r in rows}
    assert all(r["pick"] == 1 for r in rows)  # stub model always favors "1"
    assert all(r["invert"] is True for r in rows)
    # label=1 -> target = 3-1 = 2, pick=1 != 2 -> incorrect
    assert by_uid["p1"]["correct"] is False
    # label=2 -> target = 3-2 = 1, pick=1 == 1 -> correct
    assert by_uid["p2"]["correct"] is True
