# Output-conditioned token manifolds — HANDOFF 2026-08-17

## One-line state

**Status 2026-08-18 morning:** both corrected recordings are COMPLETE and verified on Ceph. The
first fourteen downstream analysis jobs failed on a launcher bug and produced no output; the
launcher is fixed, and the campaign was **resubmitted as jobs 3392423-3392436** after an
environment smoke passed. See "All 14 downstream jobs failed" below for the post-mortem and the
current job table.

**Critical correction, 2026-08-18:** the original 100-pair activation pilot is superseded for
science because its order-preserving limit contained only four Qwen items and five Llama items.
It remains an engineering/method-calibration artefact only. A tested deterministic item-round-robin
sampler is committed; corrected item-balanced storage estimates and recordings use the new immutable
tag `item_balanced_core100_v2`. No semantic feature or causal claim may use the old pilot.

## Why the first pilot is invalid for inference

`limit_pairs` previously took the first 100 pair IDs after fold selection. The pairs parquet is
item-clustered, so the nominally item-grouped discovery cells actually contained:

- Qwen fold A: 100 pairs / 200 prompts / **4 items**;
- Llama fold B: 100 pairs / 200 prompts / **5 items**.

This explains why nominal cross-validation halves contained only two items. All results below this
correction—including late ridge/CEBRA/BSF/SAE values—are retained as pipeline calibration, not
evidence. The fix is `item_round_robin_v1`: whole pair IDs are selected in deterministic rounds
across every available item while original row order is preserved. New manifests record
`limit_sampling=item_round_robin_v1`; estimates report `n_pairs` and `n_items` explicitly.

The calibration nevertheless exposed and fixed real implementation defects:

- fold-local residual and latent-code normalization;
- nested item-grouped regularization over a sufficiently wide ridge grid;
- CEBRA `delta_normal` conditioning and nonlinear KNN decoding;
- semantic-blind prompt-balanced token sampling;
- separate per-layer state and prompt-local change surfaces;
- exact-token and strictly item-cross-fitted trace exporters;
- genuine BatchTopK rather than the rejected inference-only per-sample TopK control;
- post-hoc capability/length/position/error labels kept out of representation fitting;
- held-out inherited-versus-new cross-layer decomposition.

Runpod pod `s4163rowfu5yc6` was created briefly, received no data, and was terminated. No new Runpod
volume exists and account spend returned to the pre-existing baseline. SWC remains the execution
environment.

## Corrected item-balanced core now recording

Compute estimates completed cleanly with the new sampler:

| Model/fold | estimate job | pairs | prompts | items | tokens | residual bytes |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B / A | 3392176 | 100 | 200 | 20 | 325,606 | 32,008,372,224 |
| Llama-3.2-3B-Instruct / B | 3392177 | 100 | 200 | 20 | 334,728 | 24,678,825,984 |

Corrected immutable recording jobs:

| Model/fold | job | destination |
|---|---:|---|
| Qwen3-8B / A | 3392178 | `token_manifold/item_balanced_core100_v2/record_Qwen3-8B_A` |
| Llama-3.2-3B-Instruct / B | 3392179 | `token_manifold/item_balanced_core100_v2/record_Llama-3.2-3B-Instruct_B` |

Check these jobs roughly every 30 minutes, per user preference. Do not delete or overwrite old or
new Ceph artefacts.

Both recordings **COMPLETED** and are verified on Ceph: 30 GB (Qwen, 12 layers
`0,3,7,10,13,16,20,23,26,29,33,36`) and 23 GB (Llama, 12 layers `0,3,5,8,10,13,15,18,20,23,25,28`),
each 100 pairs / 200 prompts / 20 items with `limit_sampling=item_round_robin_v1`. The expensive
stage of the campaign is done and does not need repeating.

## All 14 downstream jobs failed — launcher bug, no science lost

| Model | timeline | CEBRA all 12 | BSF all 12 | BatchTopK all 12 | raw / PCA / random controls |
|---|---:|---:|---:|---:|---:|
| Qwen | 3392182 | 3392183 | 3392184 | 3392185 | 3392186 / 3392187 / 3392188 |
| Llama | 3392189 | 3392190 | 3392191 | 3392192 | 3392193 / 3392194 / 3392195 |

Every one of these is dead. Two independent causes, both in the launcher, neither in the analysis
code or the recordings:

**1. `wb_common.sh` was never sourced (all 14 jobs).** The campaign was submitted from
`code_snapshot_4a46b76/`, but that directory only carried `sbatch/wb_common.sh`, and the sbatch
sources it as a bare relative path. The source failed, which also meant `set -euo pipefail` was
never enabled, so each job continued silently with `OUT` unset, computed for as long as 1h50m, and
only then died trying to write to `/token_manifold/...` (permission denied at the filesystem root).
The twelve method/control jobs completed their full fits and threw every result away at the final
write.

**2. The two timeline jobs were OOM-killed** at ~3 minutes with 96G/80G requested. This is a
resource request, not a defect: `timeline` holds all twelve layers in host RAM at once, and the
comparable pilot job 3391796 peaked at 111 GB RSS against a 180G request. Timeline needs `--mem`
around 200G.

A third latent problem surfaced while diagnosing: `WB_MANIFOLD_VENV` pointed at a venv that does not
exist anywhere on Ceph. `deps/token_manifold_overlay_v1` is a `pip install --target` overlay holding
cebra, bsf, sklearn and scipy but **no torch and no numpy** — it is meant to layer on top of the
judge_ft venv, not to be activated. The jobs only found torch at all because `--export=ALL` carried
interpreter state in from the submitting shell. That is unreproducible, so the sbatch now sets the
overlay explicitly via `WB_MANIFOLD_OVERLAY` on `PYTHONPATH` and lets `wb_common.sh` supply the
judge_ft venv underneath.

### Launcher fixes applied

- All eight `judge-whitebox/sbatch/*.sbatch` now use
  `source ./wb_common.sh || { ...; exit 3; }`, so a job submitted from the wrong directory dies in
  about one second instead of after two hours of GPU time. This was the shared line in every
  whitebox launcher, not just the token-manifold one.
- `token_manifold.sbatch` replaces the four `source "$WB_MANIFOLD_VENV/bin/activate"` calls with an
  explicit `PYTHONPATH` prepend of `WB_MANIFOLD_OVERLAY`.
- The timeline usage comment now records the `--mem=200G` requirement.
- `wb_common.sh` and the fixed `token_manifold.sbatch` are staged into the snapshot root
  `code_snapshot_4a46b76/`; the snapshot's frozen analysis code is untouched.

**Maintainer review point:** the `WB_MANIFOLD_VENV` → `WB_MANIFOLD_OVERLAY` change alters how the
manifold stages resolve dependencies. It matches what the successful runs actually did, but it is an
environment decision rather than a bug fix, so it is worth a look before the campaign is trusted.

### Resubmission — DONE 2026-08-18, jobs 3392423-3392436

The campaign was resubmitted after the launcher fix. The environment smoke (job 3392421) passed
first, on a CPU node, confirming the stack the analysis stages actually need:

```
/…/code/judge_ft/venv/bin/python
OK torch 2.11.0+cu130 numpy 2.3.5 sklearn 1.7.2 cebra 0.6.0 scipy 1.16.3
OK discover module imports
```

| Model | timeline | CEBRA all 12 | BSF all 12 | BatchTopK all 12 | raw / PCA / random controls |
|---|---:|---:|---:|---:|---:|
| Qwen | 3392423 | 3392425 | 3392426 | 3392427 | 3392428 / 3392429 / 3392430 |
| Llama | 3392424 | 3392431 | 3392432 | 3392433 | 3392434 / 3392435 / 3392436 |

Their logs are clean of the launcher error that killed the previous fourteen. The submitting script
is staged at `code_snapshot_4a46b76/submit_campaign.sh`.

Two further traps found while resubmitting, both worth remembering:

- **`sbatch --wrap` runs under `/bin/sh` (dash), which has no `source`.** The first smoke attempt
  (3392418) died on `source: not found`. Anything needing bash must be a real script file.
- **`--export` splits its own argument on commas**, so `--export=ALL,WB_LAYERS=0,3,7,…` sets
  `WB_LAYERS=0` and treats `3`, `7`, … as separate variable names. Comma-valued variables
  (`WB_LAYERS`, `WB_SEEDS`) must be exported in the submitting shell and picked up by a plain
  `--export=ALL`, which is what `submit_campaign.sh` does. **The original campaign passed
  `WB_LAYERS` through `--export`, so its "all 12 layers" jobs were very probably running a single
  layer**; the failed BSF job logged 36 fits, consistent with 4 layers × 3 seeds × 3 folds rather
  than the 108 that twelve layers would give. Either way the previous run's layer coverage was not
  what the directory names claimed. The resubmission passes all twelve layers, verified by a
  stubbed-`sbatch` dry run before anything was submitted.

### First corrected result: no linear verdict information inside Qwen transcripts; a weak one in Llama

Both timelines finished (3392423 in 10m25s, 3392424 in 8m48s) and are pulled to
`judge-whitebox/out/token_manifold/item_balanced_core100_v2/timeline_{qwen,llama}.csv`.

Read the readout group first and then set it aside. `answer_readout` is the last span before the
model emits its answer, so a ridge decoding the signed gap there is reading the output, not
localising a decision. Its layer profile is not a finding, and the fact that it now peaks at the last
recorded layer is what a readout should do. The only thing the readout numbers are good for is
sizing how badly four items inflated the pilot: Qwen physical layer 29 falls from R2 0.416 to 0.067,
and the sharp 26/29/33 plateau flattens away. Llama's readout is stable, 0.038 to 0.066 at layer 25.
That is a warning about the pilot's variance, not a localisation result, and no layer should be
frozen on the strength of it.

The Stage 1 question the design actually poses is where in **token time** verdict information first
becomes decodable, and whether a decision is present before, during or after each transcript. That
lives in the `transcript_1` and `transcript_2` groups across ten relative-position deciles and twelve
layers, scored against the prompt-level shuffled-gap null. Counting cells that beat their own null:

| model | span | pilot, 4-5 items | corrected, 20 items | median margin over null | raw R2 > 0 |
|---|---|---:|---:|---:|---:|
| Qwen | transcript_1 | 0/120 | 0/120 | -0.050 | 0/120 |
| Qwen | transcript_2 | 0/120 | 0/120 | -0.048 | 0/120 |
| Llama | transcript_1 | 0/120 | **109/120** | +0.012 | 7/120 |
| Llama | transcript_2 | 0/120 | **118/120** | +0.017 | 0/120 |

- **Qwen carries no linear verdict information in either transcript, at any layer or token position.**
  Every one of its 240 transcript cells sits below its own shuffled null, before and after the
  correction. Whatever drives a Qwen verdict is not linearly readable from the transcript tokens.
- **Item balancing created the only transcript-level signal in the study, and it is Llama's.** The
  pilot showed 0/240 cells above null; the corrected run shows 227/240. This is the change that
  matters — not the readout deflation — because it is the first evidence of decision-relevant
  structure while the judge is still reading.
- **It does not yet clear the design's claim-1 bar.** Held-out R2 stays at or below zero in almost
  every transcript cell (7/120 and 0/120 positive). Llama beats a shuffled-gap control while still
  predicting worse than the mean, so this is a consistent edge over the control, not decodability.
  Calling it decodability would skip a rung on the claim ladder.

The Llama margin has real token-time shape rather than being a flat offset:

- In `transcript_1` it is largest at the opening and early-middle deciles (0, 3-5, up to +0.040 at
  physical layer 20) and decays toward the end of the span, going slightly negative at decile 9 in the
  late layers (-0.009 at layers 23-28).
- In `transcript_2` it concentrates in the middle deciles (4-7) and *strengthens with depth*, reaching
  +0.042 at layer 28.

So in Llama the weak signal is present while the judge reads, is strongest mid-span in each
transcript, and fades at the tail of the first transcript rather than accumulating there. That is the
"present before, during or after each transcript" trajectory the spec asks Stage 1 to produce.

This is precisely the situation the nonlinear/sparse gate exists for: a linear locator finds nothing
usable inside Qwen's transcripts and only a sub-threshold edge inside Llama's. The twelve running
CEBRA/BSF/BatchTopK trajectory jobs test whether nonlinear or sparse structure recovers what ridge
cannot, against raw/PCA/random matched controls. The layer set needs no extension: the spec fixes
twelve evenly spaced layers with early, middle and late controls and forbids removing them.

### Original resubmission commands, for reference

The commands below reproduce the same campaign by hand; no `afterok` dependencies are needed now
that both recordings are complete. Note the comma caveat above — prefer `submit_campaign.sh`.

```bash
C=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge
S=$C/staging/whitebox/code_snapshot_4a46b76
OVERLAY=$C/staging/whitebox/deps/token_manifold_overlay_v1
R=$C/staging/whitebox/token_manifold/item_balanced_core100_v2
MQ=$R/record_Qwen3-8B_A/manifest.json
ML=$R/record_Llama-3.2-3B-Instruct_B/manifest.json
QL=0,3,7,10,13,16,20,23,26,29,33,36
LL=0,3,5,8,10,13,15,18,20,23,25,28
cd $S

# 0. environment smoke: must print torch, numpy, sklearn, cebra and bsf versions.
sbatch --partition=cpu --gres= --mem=8G --time=00:10:00 --wrap \
  "cd $S && source ./wb_common.sh && export PYTHONPATH=$OVERLAY:\$PYTHONPATH && \
   python -c 'import torch,numpy,sklearn,cebra,bsf;print(torch.__version__,numpy.__version__,sklearn.__version__,cebra.__version__)'"

# 1. timelines (large host RAM, no GPU work of consequence)
sbatch --mem=200G --export=ALL,WB_STAGE=timeline,WB_TAG=item_balanced_core100_v2,WB_RUN=qwen,WB_MANIFEST=$MQ,WB_MANIFOLD_OVERLAY=$OVERLAY token_manifold.sbatch
sbatch --mem=200G --export=ALL,WB_STAGE=timeline,WB_TAG=item_balanced_core100_v2,WB_RUN=llama,WB_MANIFEST=$ML,WB_MANIFOLD_OVERLAY=$OVERLAY token_manifold.sbatch

# 2. Qwen methods and matched controls, all twelve layers
for M in cebra bsf batchtopk; do
  sbatch --export=ALL,WB_STAGE=trajectory,WB_TAG=item_balanced_core100_v2,WB_RUN=qwen_${M}_all12_v1,WB_METHOD=$M,WB_MANIFEST=$MQ,WB_LAYERS=$QL,WB_SEEDS=0,1,2,WB_POSITIONS=32,WB_CV_FOLDS=3,WB_MANIFOLD_OVERLAY=$OVERLAY token_manifold.sbatch
done
for M in raw pca random; do
  sbatch --export=ALL,WB_STAGE=trajectory,WB_TAG=item_balanced_core100_v2,WB_RUN=qwen_controls_all12_v1,WB_METHOD=$M,WB_MANIFEST=$MQ,WB_LAYERS=$QL,WB_SEEDS=0,1,2,WB_POSITIONS=32,WB_CV_FOLDS=3,WB_MANIFOLD_OVERLAY=$OVERLAY token_manifold.sbatch
done

# 3. Llama methods and matched controls, all twelve layers
for M in cebra bsf batchtopk; do
  sbatch --export=ALL,WB_STAGE=trajectory,WB_TAG=item_balanced_core100_v2,WB_RUN=llama_${M}_all12_v1,WB_METHOD=$M,WB_MANIFEST=$ML,WB_LAYERS=$LL,WB_SEEDS=0,1,2,WB_POSITIONS=32,WB_CV_FOLDS=3,WB_MANIFOLD_OVERLAY=$OVERLAY token_manifold.sbatch
done
for M in raw pca random; do
  sbatch --export=ALL,WB_STAGE=trajectory,WB_TAG=item_balanced_core100_v2,WB_RUN=llama_controls_all12_v1,WB_METHOD=$M,WB_MANIFEST=$ML,WB_LAYERS=$LL,WB_SEEDS=0,1,2,WB_POSITIONS=32,WB_CV_FOLDS=3,WB_MANIFOLD_OVERLAY=$OVERLAY token_manifold.sbatch
done
```

The `WB_RUN` values above reproduce the original destination directory names exactly, recovered from
the failed jobs' tracebacks, so the intended output layout under `item_balanced_core100_v2` is
preserved. CEBRA still uses all eight latent dimensions, three seeds and three item folds; the
reconstructive-method and matched-control screens use the same semantic-blind 32-relative-position
sample and item folds.

Note when reading the results: `trajectory.csv` scores are grouped by
`(structural_group, relative_decile)`, and the finest of those cells can hold a single row, which
makes `r2` undefined and produces the `UndefinedMetricWarning` flood seen in the logs. The `n_rows`
column is there to filter those cells; the fits themselves are unaffected.

## Research questions that bind the corrected analysis

1. Which exact tokens change anonymous internal decision representations?
2. Which post-hoc external transcript properties those objects encode, especially independent
   capability/strength versus length, position/order and other biases?
3. In errors where the judge selects the longer but weaker transcript, when does capability
   information appear, when does the bias representation appear, and why does it take over?
4. Can layer-X dimensions predict output-associated layer-Y geometry? Decompose layer Y into an
   inherited/predictable component and a residual; define new information by incremental held-out
   gap prediction beyond layer X, not residual predictivity alone.
5. Which objects causally change the judge's own logit gap, and how do pairs interact non-additively?
6. Which objects replicate across Qwen/Llama, conditional on each model exhibiting the corresponding
   behaviour?

Capability, length, position, style and correctness labels are prohibited during representation
fitting. They enter only after anonymous objects are frozen. Semantic names require an independently
measurable rule, confirmation-fold re-extraction/alignment and causal controls.

## Scientific question

Discover residual-stream geometry that predicts each base judge's own continuous signed logit gap
without supplying semantic feature names. Then localise the discovered objects to exact transcript
tokens/phrases, name their external correlates post hoc, independently re-extract and align them, and
test causal effects and interactions. Qwen3-8B and Llama-3.2-3B-Instruct are reciprocal discovery and
replication models. Fine-tuned models come only after the base analysis is frozen.

Design: `docs/superpowers/specs/2026-08-17-output-conditioned-token-manifolds-design.md`

Plan: `docs/superpowers/plans/2026-08-17-token-manifold-bakeoff.md`

## Superseded pilot timeline results — engineering only

Both jobs used 200 prompts (100 pair IDs, both orders), but only four/five items. They used immutable float16 residual shards at 12
layers, item-grouped five-fold OOF ridge, equal prompt weights, fold-local mean/RMS scaling, and a
prompt-level shuffled-gap null.

| Model | job | peak token | frozen physical layer | held-out R2 | shuffled R2 | interpretation |
|---|---:|---|---:|---:|---:|---|
| Qwen3-8B base, fold A | 3391796 | answer readout | 29 | 0.4162 | -0.0170 | strong late readout locator |
| Llama-3.2-3B-Instruct base, fold B | 3391797 | answer readout | 25 | 0.0381 | -0.0411 | weak late readout locator |

Qwen readout R2 rises from -0.118 at layer 23 to 0.403 at 26, peaks at 0.416 at 29, stays 0.411 at
33, then falls to 0.036 at 36. Llama rises from -0.070 at layer 15 to -0.005 at 18, crosses zero at
23, and peaks at 0.038 at 25. All transcript-token cells remain below zero in both models. Therefore
the current evidence localises an output decision near the readout but does not yet explain which
transcript tokens determine it; that is the purpose of the nonlinear/sparse method gate now running.

Compact local summaries:

- `judge-whitebox/out/token_manifold/base_manifold_pilot100_v1/timeline_qwen_v2.csv`
- `judge-whitebox/out/token_manifold/base_manifold_pilot100_v1/timeline_llama_v2.csv`

The corresponding OOF tables remain on Ceph because they are about 331 MB and 306 MB.

## Superseded mixed-token bake-offs — engineering only

| Model | job | layer | new Ceph output directory |
|---|---:|---:|---|
| Qwen3-8B base | 3391805 | 29 | `bakeoff_qwen_v2_scientific_L29` |
| Llama-3.2-3B-Instruct base | 3391806 | 25 | `bakeoff_llama_v2_scientific_L25` |

Each job ran the pre-declared full grid: five seeds; CEBRA dimensions 1/2/4/8/16 at 10,000
iterations; BSF block sizes 1/2/4/8; and matched BatchTopK SAE. Outputs are `bakeoff.csv` and
`bakeoff.pt`. Do not interpret the earlier `*_quick` directories: they used smoke settings and layers
selected from the superseded timeline.

User preference: check these long jobs no more than once per hour.

## SWC paths and commands

```bash
C=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge
ROOT=$C/staging/whitebox/token_manifold/base_manifold_pilot100_v1

ssh swc 'sacct -j 3391805,3391806 --format=JobID,State,Elapsed,ExitCode,NodeList -X'

# After each job is COMPLETED, pull only the compact CSV first.
rsync -a swc:$ROOT/bakeoff_qwen_v2_scientific_L29/bakeoff.csv \
  judge-whitebox/out/token_manifold/base_manifold_pilot100_v1/bakeoff_qwen_v2_scientific_L29.csv
rsync -a swc:$ROOT/bakeoff_llama_v2_scientific_L25/bakeoff.csv \
  judge-whitebox/out/token_manifold/base_manifold_pilot100_v1/bakeoff_llama_v2_scientific_L25.csv
```

Never overwrite or delete Ceph files. Check local pull destinations are absent before `rsync`.

## Code and verification

Latest manifold commit: `d4fb7d7` (`token manifolds: normalize ridge folds`). The staged SWC
`token_manifold_discover.py` SHA-256 matches local:
`ff00a6bdd378da8edd53b7278f2509660b45f8db5fb092402065390cf5f0a677`.

- Focused manifold suite: 30 passed.
- Full whitebox suite: 199 passed, 4 optional skips, 2 existing constant-input warnings.

The isolated `.venv-manifold` intentionally lacks general project dependencies such as
`python-dotenv`; use it for manifold tests and `uv run pytest judge-whitebox/tests -q` for the full
project suite.

## Superseded and failed artefacts — do not delete

The first timeline CSV/OOF pairs are invalid for interpretation because ridge normalization was not
fitted within each CV fold:

- `$ROOT/timeline_qwen.csv`
- `$ROOT/timeline_qwen_oof.csv`
- `$ROOT/timeline_llama.csv`
- `$ROOT/timeline_llama_oof.csv`

Earlier failed/cancelled jobs left logs only:

- `$C/logs/wb-tm-est-llama-3391635.out`
- `$C/logs/wb-tm-envcheck-3391639.out`
- `$C/logs/wb-tm-overlaycheck-3391643.out`
- `$C/logs/wb-token-manifold-3391650.out`
- `$C/logs/wb-token-manifold-3391651.out`

List these for user cleanup if wanted; never remove them autonomously.

## Next gate

When both bake-offs finish, compare held-out gap decoding, reconstruction, active-code budget and
seed stability. Retain a complementary pair if CEBRA best captures decision geometry while BSF/SAE
provides the better inverse residual map. Do not assign semantic names yet. Only after this empirical
review should Plan 2 implement phrase localisation, post-hoc naming, independent re-extraction,
alignment, causal interventions and interactions.
