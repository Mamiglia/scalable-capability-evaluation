#!/bin/bash
#SBATCH --job-name=jft-pipeline
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Whole judge-ft pilot in one job: fail-fast smoke train, lr x rank grid,
# auto-pick by val accuracy, final train, verdict + gpqa evals.
# Every stage skips if its output exists, so a requeue resumes.
source hpc_common.sh

echo "=== STAGE preflight $(date) ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import vllm, peft, transformers, pandas; print('vllm', vllm.__version__)"
{ test -f "$MODEL/config.json" && ls "$MODEL"/*.safetensors >/dev/null 2>&1; } || { echo "MISSING WEIGHTS"; exit 1; }
test -f "$IN/pairs.parquet" || { echo "MISSING PAIRS"; exit 1; }

echo "=== STAGE smoke $(date) ==="
[ -f "$OUT/smoke_adapter/adapter_config.json" ] || \
python train.py --pairs "$IN/pairs.parquet" --base-model "$MODEL" \
  --lr 1e-4 --lora-rank 16 --limit 100 --out-dir "$OUT/smoke_adapter"

echo "=== STAGE grid $(date) ==="
while read -r LR RANK LIMIT; do
  D="$OUT/adapters/grid_lr${LR}_r${RANK}"
  [ -f "$D/adapter_config.json" ] && { echo "cell $D done, skipping"; continue; }
  python train.py --pairs "$IN/pairs.parquet" --base-model "$MODEL" \
    --lr "$LR" --lora-rank "$RANK" --limit "$LIMIT" --out-dir "$D"
done < <(python - <<'EOF'
import yaml
t = yaml.safe_load(open("config.yaml"))["train"]
for lr in t["lr_grid"]:
    for r in t["lora_rank_grid"]:
        print(lr, r, t["grid_subsample"])
EOF
)

echo "=== STAGE pick $(date) ==="
[ -f "$OUT/best_cell.txt" ] || python - "$OUT" <<'EOF'
import glob, json, os, re, sys
out = sys.argv[1]
best = None
for d in sorted(glob.glob(f"{out}/adapters/grid_*")):
    rec = [json.loads(l) for l in open(f"{d}/train_log.jsonl")][-1]
    m = re.match(r"grid_lr(.+)_r(\d+)$", os.path.basename(d))
    print(os.path.basename(d), "val_acc", rec["val_acc"])
    if best is None or rec["val_acc"] > best[0]:
        best = (rec["val_acc"], m.group(1), m.group(2))
open(f"{out}/best_cell.txt", "w").write(f"{best[1]} {best[2]}\n")
EOF
read -r LR RANK < "$OUT/best_cell.txt"
echo "best cell: lr=$LR rank=$RANK"

echo "=== STAGE final $(date) ==="
[ -f "$OUT/adapters/final/adapter_config.json" ] || \
python train.py --pairs "$IN/pairs.parquet" --base-model "$MODEL" \
  --lr "$LR" --lora-rank "$RANK" --splits train,val --out-dir "$OUT/adapters/final"

echo "=== STAGE verdicts $(date) ==="
A="$OUT/adapters/final"
[ -f "$OUT/verdicts_base.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs.parquet" --base-model "$MODEL" --tp 1 \
  --splits eval_mm,eval_mixed,eval_items --out "$OUT/verdicts_base.jsonl"
[ -f "$OUT/selfpref_base.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_selfpref.parquet" --base-model "$MODEL" --tp 1 \
  --out "$OUT/selfpref_base.jsonl"
[ -f "$OUT/verdicts_ft.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$A" --splits eval_mm,eval_mixed,eval_items --out "$OUT/verdicts_ft.jsonl"
[ -f "$OUT/selfpref_ft.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_selfpref.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$A" --out "$OUT/selfpref_ft.jsonl"

echo "=== STAGE gpqa $(date) ==="
[ -f "$OUT/gpqa_base.jsonl" ] || \
python pod_gpqa.py --items "$IN/items_ft.parquet" --base-model "$MODEL" --tp 1 \
  --out "$OUT/gpqa_base.jsonl"
[ -f "$OUT/gpqa_ft.jsonl" ] || \
python pod_gpqa.py --items "$IN/items_ft.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$OUT/adapters/final" --out "$OUT/gpqa_ft.jsonl"

echo "=== PIPELINE_DONE $(date) ==="
