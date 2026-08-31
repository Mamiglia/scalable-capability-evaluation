# Pairwise capability Singular correction

The pairwise-capability Singular calculation previously subtracted the number
0.5 from feature-only accuracy. That is invalid when canonical pair orientation
is imbalanced: a probability-0.5 model can have realized accuracy other than
50% on the held-out rows.

The corrected definition is:

```text
Singular contribution = one-feature held-out accuracy
                      − same-row no-feature held-out accuracy
```

The no-feature model remains the exact antisymmetric null with logit 0 and
probability 0.5. Its *probability* is fixed, but its realized accuracy is
computed from the same rows rather than overwritten with 0.5. Paired delta log
loss remains available as a secondary measure. Unique effects retain their
existing full-versus-lesioned definition.

## Verification

- Fix commit: `ce41be8` on `pairwise-capability-hierarchical-refit`.
- Forty-one targeted capability tests passed.
- Corrected summary job: `3494336` on SWC; no GLM was refitted.
- Four source summary manifests were pinned by SHA-256 in the corrected
  manifest.
- The real data contain 128 feature×benchmark cells whose held-out feature
  values are identically zero. Their maximum absolute corrected Singular
  contribution is exactly **0.0 percentage points**.

The largest corrected Raw-L1 Singular effects are now modest: explains
reasoning +1.21 pp, plan position +0.88 pp, and the `composite-021` feature
+0.40 pp. This replaces the large prevalence-driven effects from the invalid
`feature-only accuracy − 0.5` calculation.

## Artifacts

```text
judge_results/capability_predictions_glm/pairwise-hierarchical-singular-corrected-20260831/
├── lesions-corrected.csv
├── feature-effects.csv
├── group-effects.csv
├── benchmark-effects.csv
├── zero-valued-feature-check.csv
└── manifest.json
```

The source GLM fits and lesion predictions are unchanged; only derived
attribution tables were regenerated.
