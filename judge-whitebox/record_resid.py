"""Residual-stream recorder: forward-only, per-prompt capture of the post-block hidden state at
the LAST token position, at every layer boundary (output_hidden_states gives n_layers+1 states,
embeddings included). No per-token storage -- v1 is one vector per (prompt, layer).

Two modes: pairwise (default) records the judging-verdict prompt per pairs-df row, as before.
--single-transcript instead records each unique transcript on its own, in a non-judging read
frame (same discipline as corpus.py: no judging vocabulary), for geometry analysis in
transcript_geometry.py."""
import argparse
import hashlib
import pathlib

import torch

import verdicts
import wbconfig
import wbpairs
import wbprompts

# Non-judging read frame, module constant (mirrors corpus.py's SUMMARISE_SYSTEM/USER discipline):
# no mention of judging, verdicts, "capable", or transcript numbering.
TRANSCRIPT_SYSTEM = "You are a helpful assistant."
TRANSCRIPT_USER = "Read the following solution carefully.\n---\n{text}\n---"


def render_transcript_prompt(text, tokenizer):
    msgs = [{"role": "system", "content": TRANSCRIPT_SYSTEM},
            {"role": "user", "content": TRANSCRIPT_USER.format(text=text)}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)


def build_units(df):
    """Explode pairs-df rows into one row per unique transcript (author's answer to an item),
    deduped by text -- first occurrence kept, in row order (same dedupe discipline as
    corpus.py's build_control). Each unit carries the provenance transcript_geometry.py needs:
    item, author model name, cap score."""
    units, seen_text = [], set()
    for _, row in df.iterrows():
        for suffix in ("1", "2"):
            text = row[f"text_{suffix}"]
            if text in seen_text:
                continue
            seen_text.add(text)
            units.append({"item": row["item"], "author": row[f"model_{suffix}"],
                          "cap": row[f"cap_{suffix}"], "text": text})
    return units


def record_single_transcript(model, tokenizer, units, max_tokens):
    """One forward pass per unique transcript, rendered via render_transcript_prompt -- no verdict
    readout (this frame never asks for an answer). Same cap-and-count skip pattern as record().
    Returns (resid [n_kept, n_layers+1, d_model] float32, kept_unit_idx: list[int] positions
    into units, aligned with the returned rows)."""
    device = next(model.parameters()).device
    resid_rows, kept_unit_idx = [], []
    n_skipped = 0
    for i, unit in enumerate(units):
        prompt = render_transcript_prompt(unit["text"], tokenizer)
        enc = tokenizer(prompt, return_tensors="pt")
        if enc["input_ids"].shape[1] > max_tokens:
            n_skipped += 1
            continue
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs = torch.stack([h[0, -1, :].float().cpu() for h in out.hidden_states])  # [n_layers+1, d]
        resid_rows.append(hs)
        kept_unit_idx.append(i)
    print(f"skipped {n_skipped} transcripts over {max_tokens} tokens; kept {len(kept_unit_idx)}")
    resid = torch.stack(resid_rows) if resid_rows else torch.zeros(0, 0, 0)
    return resid, kept_unit_idx


def record(model, tokenizer, df, invert, max_tokens):
    """One forward pass per kept row (no grad, no batching -- batch size 1 needs no padding).
    Rows over max_tokens are skipped, same cap-and-count pattern as screen.py/verdicts.py.
    Returns (resid [n_kept, n_layers+1, d_model] float32, l1 [n_kept], l2 [n_kept],
    kept_row_idx: list[int] positions into df, aligned with the returned rows)."""
    device = next(model.parameters()).device
    resid_rows, l1s, l2s, kept_row_idx = [], [], [], []
    n_skipped = 0
    for i, (_, row) in enumerate(df.iterrows()):
        prompt = wbprompts.render_verdict_prompt(row, tokenizer, invert=invert)
        enc = tokenizer(prompt, return_tensors="pt")
        if enc["input_ids"].shape[1] > max_tokens:
            n_skipped += 1
            continue
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs = torch.stack([h[0, -1, :].float().cpu() for h in out.hidden_states])  # [n_layers+1, d]
        resid_rows.append(hs)
        l1, l2 = verdicts.readout(out.logits[0, -1, :], tokenizer)
        l1s.append(l1)
        l2s.append(l2)
        kept_row_idx.append(i)
    print(f"skipped {n_skipped} rows over {max_tokens} tokens; kept {len(kept_row_idx)}")
    resid = torch.stack(resid_rows) if resid_rows else torch.zeros(0, 0, 0)
    return resid, torch.tensor(l1s, dtype=torch.float32), torch.tensor(l2s, dtype=torch.float32), kept_row_idx


def main():
    c = wbconfig.cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="name key into models.hf_ids")
    parser.add_argument("--adapter", default=None, help="optional LoRA adapter path")
    parser.add_argument("--invert", action="store_true",
                         help="render the inverted (pick-weaker) verdict prompt")
    parser.add_argument("--single-transcript", action="store_true",
                         help="record unique transcripts in a non-judging read frame instead of "
                              "judging pairs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default=c["verdicts"]["eval_split"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    assert not (args.single_transcript and args.invert), \
        "--invert doesn't apply to --single-transcript (no verdict instruction in that frame)"

    model, tokenizer, device = verdicts.load_model(args.model, args.adapter, c)
    df = wbpairs.load_pairs(c, split=args.split, limit=args.limit)
    max_tokens = c["verdicts"]["max_input_tokens"]

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"split": args.split, "limit": args.limit, "model": args.model, "adapter": args.adapter,
            "pairs_path": c["paths"]["pairs"]}  # resolved (WB_PAIRS-aware) -- pins down which parquet

    if args.single_transcript:
        units = build_units(df)
        resid, kept_unit_idx = record_single_transcript(model, tokenizer, units, max_tokens)
        kept = [units[i] for i in kept_unit_idx]
        torch.save({
            "mode": "single_transcript", "resid": resid,
            "item": [u["item"] for u in kept], "author": [u["author"] for u in kept],
            "cap": [u["cap"] for u in kept],
            "text_hash": [hashlib.sha256(u["text"].encode()).hexdigest() for u in kept],
            "invert": False, **meta,
        }, out_path)
        n_layers = resid.shape[1] - 1 if resid.shape[0] else 0
        print(f"n_transcripts={resid.shape[0]} n_layers={n_layers} -> {out_path}")
        return

    resid, l1, l2, kept_row_idx = record(model, tokenizer, df, args.invert, max_tokens)
    torch.save({
        "mode": "pairwise", "resid": resid, "l1": l1, "l2": l2, "kept_row_idx": kept_row_idx,
        "invert": args.invert, **meta,
    }, out_path)
    n_layers = resid.shape[1] - 1 if resid.shape[0] else 0
    print(f"n_prompts={resid.shape[0]} n_layers={n_layers} -> {out_path}")


if __name__ == "__main__":
    main()
