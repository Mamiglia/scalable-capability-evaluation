# Staging + launch (user-driven; nothing runs on the gateway)

Base (checked 2026-08-10, all three judges already staged and complete):

```bash
BASE=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge
# weights:  $BASE/models/{Qwen3-0.6B,Qwen3-8B,Llama-3.2-3B-Instruct}   (+ Qwen3-32B spare)
# our new code goes to $BASE/code/judge_variation/  - new dir, nothing existing touched
```

## 1. Weights — already on ceph, nothing to do

## 2. Stage repo + data (new dirs only)

```bash
cd /Users/moriohamada/PycharmProjects/ai_safety/small-judge
rsync -av --ignore-existing --exclude .git --exclude logs --exclude judge_results \
  ./ swc:$BASE/code/judge_variation/repo/
```

(`external_data/` rides along in that rsync — it lives inside the repo.)

Re-staging after local edits: our own code dir is the agreed overwrite exception —

```bash
rsync -av analyses/judge-variation/ swc:$BASE/code/judge_variation/repo/analyses/judge-variation/
```

## 3. Fresh venv (do NOT touch code/judge_ft/venv — live jobs use it)

Run by hand on the gateway (quick installs; if the reaper kills pip, rerun —
pip resumes from its cache):

```bash
/usr/bin/python3.12 -m venv $BASE/code/judge_variation/venv
source $BASE/code/judge_variation/venv/bin/activate
pip install vllm inspect-ai pyyaml
pip install -e $BASE/code/judge_variation/repo
```

## 4. Pilot (one job, ~10 items, review before the grid)

```bash
cd $BASE/code/judge_variation/repo/analyses/judge-variation/slurm
sbatch -J pilot-q8b serve_and_judge.sbatch $BASE/models/Qwen3-8B qwen3-8b native \
  $BASE/code/judge_variation/repo/analyses/judge-variation/configs/gpqa_a_baseline.yaml
```

For the pilot, first edit the `python run.py` line in the sbatch to add
`--limit 10` (remove again before the grid; Slurm snapshots the script at
submit, so edit before each submit).

Pull the pilot logs back for review:

```bash
rsync -av swc:$BASE/code/judge_variation/repo/logs/judging-variations/ logs/judging-variations/
```

## 5. Full grid (priority order; later jobs are the first cut if time is short)

```bash
cd $BASE/code/judge_variation/repo/analyses/judge-variation/slurm
C=$BASE/code/judge_variation/repo/analyses/judge-variation/configs
sbatch -J q06-nat  serve_and_judge.sbatch $BASE/models/Qwen3-0.6B qwen3-0.6b native \
  $C/gpqa_a_baseline.yaml $C/gpqa_e_with_reasoning_native.yaml \
  $C/otis-aime_a_baseline.yaml $C/otis-aime_e_with_reasoning_native.yaml \
  $C/gpqa_b_ignore_superficial.yaml $C/gpqa_d_pick_weaker.yaml \
  $C/gpqa_c_ignore_superficial_confidence.yaml \
  $C/otis-aime_b_ignore_superficial.yaml $C/otis-aime_d_pick_weaker.yaml \
  $C/otis-aime_c_ignore_superficial_confidence.yaml
sbatch -J q06-yarn serve_and_judge.sbatch $BASE/models/Qwen3-0.6B qwen3-0.6b-yarn yarn \
  $C/gpqa_e_with_reasoning.yaml $C/otis-aime_e_with_reasoning.yaml
sbatch -J q8b-nat  serve_and_judge.sbatch $BASE/models/Qwen3-8B qwen3-8b native \
  $C/gpqa_a_baseline.yaml $C/gpqa_e_with_reasoning_native.yaml \
  $C/otis-aime_a_baseline.yaml $C/otis-aime_e_with_reasoning_native.yaml \
  $C/gpqa_b_ignore_superficial.yaml $C/gpqa_d_pick_weaker.yaml \
  $C/gpqa_c_ignore_superficial_confidence.yaml \
  $C/otis-aime_b_ignore_superficial.yaml $C/otis-aime_d_pick_weaker.yaml \
  $C/otis-aime_c_ignore_superficial_confidence.yaml
sbatch -J q8b-yarn serve_and_judge.sbatch $BASE/models/Qwen3-8B qwen3-8b-yarn yarn \
  $C/gpqa_e_with_reasoning.yaml $C/otis-aime_e_with_reasoning.yaml
sbatch -J l3b      serve_and_judge.sbatch $BASE/models/Llama-3.2-3B-Instruct llama-3.2-3b llama \
  $C/gpqa_a_baseline.yaml $C/gpqa_e_with_reasoning.yaml \
  $C/otis-aime_a_baseline.yaml $C/otis-aime_e_with_reasoning.yaml \
  $C/gpqa_b_ignore_superficial.yaml $C/gpqa_d_pick_weaker.yaml \
  $C/gpqa_c_ignore_superficial_confidence.yaml \
  $C/otis-aime_b_ignore_superficial.yaml $C/otis-aime_d_pick_weaker.yaml \
  $C/otis-aime_c_ignore_superficial_confidence.yaml
```

Attribution (last, cut first):

```bash
# append to any native/llama job's config loop, or its own job:
python attribution.py --bench gpqa --judge-model openai-api/local/judge --judge-short <judge>
python attribution.py --bench otis-aime --judge-model openai-api/local/judge --judge-short <judge>
```

Notes
- The `_native` yamls run under native serving only; yarn jobs run the full e
  yamls, which include those ids, so the comparison set is judged by both.
- Qwen judges think by default; `--reasoning-parser qwen3` keeps think-text out
  of the parsed completion. Llama mode serves without a parser.
- Preemption/requeue safe: finished pair dirs are skipped on restart.
- Everything writes NEW files only; nothing on ceph is deleted or overwritten
  (the one agreed exception: our own code dir, step 2).
