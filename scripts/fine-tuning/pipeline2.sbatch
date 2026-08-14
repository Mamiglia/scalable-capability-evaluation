#!/bin/bash
#SBATCH --job-name=jft-hard2
#SBATCH --partition=a100,gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Hard-example round 2: find train pairs the round-1 adapter still judges
# wrong, continue training that adapter on them (2 LRs, val-picked), then
# eval the winner. Separate adapters; round-1 outputs untouched.
source hpc_common.sh
A1="$OUT/adapters/final"

echo "=== STAGE train-verdicts $(date) ==="
[ -f "$OUT/train_verdicts_ft.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$A1" --splits train --out "$OUT/train_verdicts_ft.jsonl"

echo "=== STAGE hard-select $(date) ==="
# hard pairs alone teach the model to invert itself (lr 3e-5 run collapsed
# to val 0.14); mix in an equal replay sample of correctly-judged pairs
[ -f "$OUT/pairs_hard2mix.parquet" ] || python - "$IN" "$OUT" <<'EOF'
import json, sys
import numpy as np
import pandas as pd
inp, out = sys.argv[1], sys.argv[2]
v = pd.DataFrame([json.loads(l) for l in open(f"{out}/train_verdicts_ft.jsonl")])
picked = (v.logit_1 > v.logit_2).map({True: 1, False: 2})
bad = set(v.pair_uid[picked != v.label])
p = pd.read_parquet(f"{inp}/pairs.parquet")
train = p[p.split == "train"]
hard = train[train.pair_uid.isin(bad)]
assert (hard.groupby("pair_uid").size() == 2).all()
good_uids = np.array(sorted(set(train.pair_uid) - bad))
rng = np.random.default_rng(0)
replay_uids = set(rng.choice(good_uids, size=hard.pair_uid.nunique(), replace=False))
replay = train[train.pair_uid.isin(replay_uids)]
df = pd.concat([hard, replay, p[p.split == "val"]])
print(f"hard pairs: {hard.pair_uid.nunique()} of {v.pair_uid.nunique()}; "
      f"replay pairs: {replay.pair_uid.nunique()}; "
      f"train label mean {pd.concat([hard, replay]).label.mean():.3f}")
df.to_parquet(f"{out}/pairs_hard2mix.parquet")
EOF

echo "=== STAGE hard-train $(date) ==="
for LR in 3e-5 1e-4; do
  D="$OUT/adapters/hard2mix_lr${LR}"
  [ -f "$D/adapter_config.json" ] && { echo "hard2mix $LR done, skipping"; continue; }
  python train.py --pairs "$OUT/pairs_hard2mix.parquet" --base-model "$MODEL" \
    --resume-adapter "$A1" --lr "$LR" --out-dir "$D"
done

echo "=== STAGE hard-pick $(date) ==="
[ -f "$OUT/hard2mix_best.txt" ] || python - "$OUT" <<'EOF'
import glob, json, sys
out = sys.argv[1]
best = None
for d in sorted(glob.glob(f"{out}/adapters/hard2mix_lr*")):
    rec = [json.loads(l) for l in open(f"{d}/train_log.jsonl")][-1]
    print(d.split("/")[-1], "val_acc", rec["val_acc"])
    if best is None or rec["val_acc"] > best[0]:
        best = (rec["val_acc"], d)
open(f"{out}/hard2mix_best.txt", "w").write(best[1] + "\n")
EOF
read -r A2 < "$OUT/hard2mix_best.txt"
echo "hard2mix best: $A2"

echo "=== STAGE hard-evals $(date) ==="
[ -f "$OUT/verdicts_hard2.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$A2" --splits eval_mm,eval_mixed,eval_items --out "$OUT/verdicts_hard2.jsonl"
[ -f "$OUT/selfpref_hard2.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_selfpref.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$A2" --out "$OUT/selfpref_hard2.jsonl"
[ -f "$OUT/gpqa_hard2.jsonl" ] || \
python pod_gpqa.py --items "$IN/items_ft.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$A2" --out "$OUT/gpqa_hard2.jsonl"

echo "=== PIPELINE2_DONE $(date) ==="
