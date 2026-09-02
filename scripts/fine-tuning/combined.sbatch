#!/bin/bash
#SBATCH --job-name=jft-combined
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Combined SFT set (plain + spoof + rewrite). Grid-sweep lr x rank, pick the
# LOWEST rank whose val is within margin of the best (interpretable), final-train,
# then 3-way held-out eval: plain (eval_mm), spoof, adversarial style.
# Submit per model, e.g.:
#   sbatch --export=ALL,JFT_MODEL=Qwen3-0.6B,JFT_OUT=out_combined06 combined.sbatch
#   sbatch --export=ALL,JFT_MODEL=Qwen3-8B,JFT_OUT=out_combined8b combined.sbatch
source hpc_common.sh
PAIRS="$IN/pairs_combined.parquet"
MARGIN="${JFT_MARGIN:-0.01}"
test -f "$PAIRS" || { echo "MISSING pairs_combined.parquet"; exit 1; }

echo "=== grid $(date) ==="
while read -r LR RANK LIMIT; do
  D="$OUT/grid_lr${LR}_r${RANK}"
  [ -f "$D/adapter_config.json" ] && { echo "cell $D done"; continue; }
  python train.py --pairs "$PAIRS" --base-model "$MODEL" \
    --lr "$LR" --lora-rank "$RANK" --limit "$LIMIT" --out-dir "$D"
done < <(python - <<'EOF'
import yaml
t = yaml.safe_load(open("config.yaml"))["train"]
for lr in t["lr_grid"]:
    for r in t["lora_rank_grid"]:
        print(lr, r, t["grid_subsample"])
EOF
)

echo "=== pick lowest-good rank $(date) ==="
[ -f "$OUT/best_cell.txt" ] || python - "$OUT" "$MARGIN" <<'EOF'
import glob, json, os, re, sys
out, margin = sys.argv[1], float(sys.argv[2])
cells = []
for d in sorted(glob.glob(f"{out}/grid_lr*_r*")):
    rec = [json.loads(l) for l in open(f"{d}/train_log.jsonl")][-1]
    m = re.match(r"grid_lr(.+)_r(\d+)$", os.path.basename(d))
    cells.append((float(rec["val_acc"]), m.group(1), int(m.group(2))))
    print(os.path.basename(d), "val_acc", rec["val_acc"])
best = max(c[0] for c in cells)
ok = [c for c in cells if c[0] >= best - margin]
val, lr, rank = min(ok, key=lambda c: (c[2], -c[0]))   # lowest rank, then best val
print(f"best_val={best:.4f} -> pick lr={lr} rank={rank} (val {val:.4f})")
open(f"{out}/best_cell.txt", "w").write(f"{lr} {rank}\n")
EOF
read -r LR RANK < "$OUT/best_cell.txt"
echo "final cell: lr=$LR rank=$RANK"

echo "=== final train $(date) ==="
A="$OUT/adapters/final"
[ -f "$A/adapter_config.json" ] || \
python train.py --pairs "$PAIRS" --base-model "$MODEL" \
  --lr "$LR" --lora-rank "$RANK" --splits train,val --out-dir "$A"

echo "=== 3-way held-out eval $(date) ==="
[ -f "$OUT/eval_plain.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$A" --splits eval_mm --out "$OUT/eval_plain.jsonl"
[ -f "$OUT/eval_spoof.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_spoof_heldout.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$A" --out "$OUT/eval_spoof.jsonl"
[ -f "$OUT/eval_style.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_style_heldout.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$A" --out "$OUT/eval_style.jsonl"

echo "=== COMBINED_DONE $(date) ==="
