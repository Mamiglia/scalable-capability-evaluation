# Staging + launch (user-driven; nothing runs on the gateway)

Everything reuses what the pairwise runs already staged: weights in
$BASE/models/, the judge_variation venv, the judge_ft vllm binary. Only the
repo copy is new.

```bash
BASE=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge
```

## 1. Stage the repo (new dir, nothing existing touched)

```bash
cd /Users/moriohamada/PycharmProjects/ai_safety/small-judge
rsync -av --ignore-existing --exclude .git --exclude logs --exclude judge_results \
  ./ swc:$BASE/code/direct_judge/repo/
```

Re-staging after local edits (our own code dir is the overwrite exception):

```bash
rsync -av scripts/analyses/direct-judge/ swc:$BASE/code/direct_judge/repo/scripts/analyses/direct-judge/
```

## 2. Pilot (one judge, capped items — check parse rate before the grid)

```bash
ssh swc
cd $BASE/code/direct_judge/repo/scripts/analyses/direct-judge/slurm
sbatch -J dj-pilot --export=ALL,RUN_ARGS="--limit 5 --log-root $BASE/code/direct_judge/pilot-logs/gpqa" serve_and_score.sbatch \
  $BASE/models/Qwen3-0.6B qwen3-0.6b native configs/gpqa.yaml
```

The pilot writes under pilot-logs/, not the real log tree — success logs in
the real tree would make the grid's resume skip those models. Pull it down and
check parse rate and score spread before launching the grid:

```bash
rsync -av swc:$BASE/code/direct_judge/pilot-logs/ /tmp/dj-pilot/
cd scripts/analyses/direct-judge && uv run python analyse.py --roots /tmp/dj-pilot --out-dir /tmp/dj-pilot-results
```

(The pilot's `--log-root ...pilot-logs/gpqa` makes the pulled tree `<root>/gpqa/<judge>/<model>`, which is exactly the layout analyse.py's `--roots` expects.)

## 3. Grid (after pilot looks sane)

```bash
sbatch -J dj-q06  serve_and_score.sbatch $BASE/models/Qwen3-0.6B            qwen3-0.6b   native configs/gpqa.yaml configs/otis-aime.yaml
sbatch -J dj-q8   serve_and_score.sbatch $BASE/models/Qwen3-8B              qwen3-8b     native configs/gpqa.yaml configs/otis-aime.yaml
sbatch -J dj-l3   serve_and_score.sbatch $BASE/models/Llama-3.2-3B-Instruct llama-3.2-3b  llama  configs/gpqa.yaml configs/otis-aime.yaml
```

## 4. Pull results

```bash
rsync -av swc:$BASE/code/direct_judge/repo/logs/direct-judge/ logs/direct-judge/
```
