# Frontier ECI Side-Analysis Execution Plan

**Goal:** Predict Q2/Q3/Q4 accuracy from Q1 successful-transcript judgments for one quarterly ECI
frontier shared by GPQA and SWE-mini.

**Execution:** All substantive work runs on RunPod. All artifacts remain under the isolated
`frontier_eci` / `frontier-eci` / `eci-semiannual` paths from the design spec.

## 1. Freeze design and difficulty

- [x] Freeze eight quarterly candidates, three checkpoint judges, two bridge judges, releases,
  ECI values, prompts, and 25 strictly historical pairs in `configs/frontier_eci/experiment.yaml`.
- [x] Freeze Q1/Q2/Q3/Q4 from canonical-pool difficulty only in
  `configs/frontier_eci/item_quartiles.csv`.
- [x] Verify Q1+Q2 exactly equals the pre-existing frozen easy half.
- [x] Test excluded effort variants and strict release inequalities.

## 2. Prepare sources on RunPod

- [x] Copy the frozen canonical identity table with a recorded SHA-256.
- [x] Import GPQA GPT-4o, Claude 3.7, Gemini 3 Pro, and Opus 4.6 Epoch logs.
- [x] Download and verify the public 4.0 GB Gemini 3 Pro SWE archive; trim it to the 46 SWE-mini
  items and 100-message cap.
- [x] Copy exact canonical-pool logs required by the pair graph.
- [x] Pin and selectively mirror HF revision
  `6b2b9658f30de3aeade93d9192c32f4a0c19ce6e`.
- [x] Export 110,748 protocol-compatible HF verdict rows; exclude incompatible conditions.

## 3. Generate and preflight judging

- [x] Generate Q1-only per-benchmark/per-judge configs, skipping only bridge cells already complete
  in HF.
- [x] Exhaustively render 12,212 calls and save per-pair input/expected/worst-case cost.
- [x] Snapshot all live OpenRouter endpoint records.
- [x] Run all ten GPQA/SWE smoke cells; verify parsing, counterbalancing, resume behavior, and
  per-response served-provider provenance.
- [x] Diagnose GPT-5.2 routing: keep OpenAI-only/fallback-off/data-deny, disable only the inaccurate
  strict parameter registry check, and test the route end to end.

## 4. Run full paid matrix

- [x] Add a tested pair-level budget guard using actual OpenRouter key usage and preflight reserves.
- [ ] Run all missing Q1 cells sequentially in durable RunPod tmux session `frontier-eci-full`.
- [ ] Rejudge only genuinely unanswered provider failures, if any; never recompute completed cells.
- [ ] Record exact API cost, RunPod duration/cost, success counts, and served providers.

## 5. Aggregate without future leakage

- [x] Test rolling release prefixes: future edges cannot alter an earlier candidate.
- [x] Test accuracy calibration: the target's own Q2/Q3/Q4 outcome is never an input.
- [x] Test equal-judge consensus weighting independent of upload volume.
- [ ] Parse completed new logs into side-analysis verdict and rollout tables.
- [ ] Combine new rows with compatible HF rows, preferring frozen canonical rollouts wherever
  available.
- [ ] For each judge/candidate/quartile, fit BT on the release prefix, calibrate using strictly older
  models, and bootstrap Q1 items.

## 6. Diagnose and plot

- [x] Add transcript-visibility extraction for visible length, reasoning removed, and
  redacted/encrypted reasoning.
- [ ] Export predictions, metrics, model residuals, position/consistency/length diagnostics, and
  transcript visibility tables.
- [ ] Render checkpoint, bridge, residual heatmap, and visibility-versus-error figures as 300 dpi
  PNG and editable PDF.
- [ ] Visually inspect every figure at original resolution and cross-check plotted points to CSV.

## 7. Verify and hand off

- [ ] Run all `tests/frontier_eci` tests and `git diff --check` on RunPod.
- [ ] Write `docs/results/2026-08-28-frontier-eci-side-analysis.md` with provenance, results,
  caveats, and links.
- [ ] Transfer only isolated code/config/results plus intended judge model configs back to the
  workspace.
- [ ] Delete the remote secrets file, transfer final artifacts, delete the RunPod, and confirm it is
  absent from the account.
