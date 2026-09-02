# Eval-variation SWC run (user-driven; nothing runs on the gateway)

Reuses judge_variation's venv and the staged weights read-only; our code +
data stage into the NEW dir `$BASE/code/eval_variation/`. Judge logs come
back by rsync when jobs finish.

```bash
BASE=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge
```

## 1. Stage (already done by Claude via rsync --ignore-existing; re-run after local edits)

```bash
cd /Users/moriohamada/PycharmProjects/ai_safety/small-judge/.claude/worktrees/eval-variation
rsync -av --ignore-existing --exclude .git --exclude logs --exclude data \
  --exclude judge_results --exclude external_data --exclude model-sampler \
  --exclude docs --exclude .env --exclude .venv \
  ./ swc:$BASE/code/eval_variation/repo/
rsync -avL --ignore-existing data/gpqa/63793eb6/ swc:$BASE/code/eval_variation/repo/data/gpqa/63793eb6/
rsync -avL --ignore-existing data/eval-variation/ swc:$BASE/code/eval_variation/repo/data/eval-variation/
```

(code-dir overwrite exception: `rsync -av analyses/eval-variation/ swc:$BASE/code/eval_variation/repo/analyses/eval-variation/`)

## 2. Pilot (one judge, 10 rows/pair; review before the grid)

```bash
cd $BASE/code/eval_variation/repo/analyses/eval-variation/slurm
C=$BASE/code/eval_variation/repo/analyses/eval-variation/configs
sbatch -J ev-pilot --export=ALL,RUN_ARGS="--limit 10" serve_and_judge.sbatch \
  $BASE/models/Qwen3-0.6B Qwen3-0.6B qwen $C/judge_baseline.yaml
```

## 3. Full grid (all six conditions per judge)

```bash
cd $BASE/code/eval_variation/repo/analyses/eval-variation/slurm
C=$BASE/code/eval_variation/repo/analyses/eval-variation/configs
CFGS="$C/judge_baseline.yaml $C/judge_words200.yaml $C/judge_words50-100.yaml \
  $C/judge_confident.yaml $C/judge_matteroffact.yaml"
sbatch -J ev-q06 serve_and_judge.sbatch $BASE/models/Qwen3-0.6B Qwen3-0.6B qwen $CFGS
sbatch -J ev-l3b serve_and_judge.sbatch $BASE/models/Llama-3.2-3B-Instruct Llama-3.2-3B-Instruct llama $CFGS
sbatch -J ev-q8b serve_and_judge.sbatch $BASE/models/Qwen3-8B Qwen3-8B qwen $CFGS
```

The pooled condition is API-complete and pooled configs reference all three
condition dirs, so it is omitted from the SWC grid for now; add
`$C/judge_api_lenmatch-pooled.yaml` to a job's list to replicate it locally
(run.py ignores the openrouter judge rows via --judge).

## 4. Pull logs back + analyze (Mac)

```bash
cd /Users/moriohamada/PycharmProjects/ai_safety/small-judge/.claude/worktrees/eval-variation
rsync -av --ignore-existing swc:$BASE/code/eval_variation/repo/logs/eval-variation/ \
  /Users/moriohamada/PycharmProjects/ai_safety/small-judge/logs/eval-variation/
.venv/bin/python analyses/eval-variation/analyze.py
```

Notes
- SWC judge slugs are `openai-api/local/<served-name>`; the sbatch exports
  LOCAL_API_KEY=none and passes --model-base-url with a per-job port.
- Qwen judges think by default; `--reasoning-parser qwen3` keeps think text
  out of the parsed completion. 32k ctx fits any card in gpu_lowp.
- Requeue-safe: finished pair dirs are skipped on restart.
- Everything writes NEW files only; the one overwrite exception is our own
  code dir (step 1 note).
