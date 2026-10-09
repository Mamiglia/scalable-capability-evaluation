# Model config rules

Four rules, in priority order. Check a config against the live OpenRouter endpoints API before
any run that produces a reported number.

## 1. Pin the endpoint

```yaml
model_args:
  provider:
    only: [deepinfra/bf16]   # one endpoint tag: provider AND quantization
    allow_fallbacks: false   # fail loudly rather than swap weights mid-run
```

`allow_fallbacks` defaults to **true**, so `order:` alone is a preference, not a pin — the
router will serve you a different backend the moment the preferred one is busy. A bare
provider slug (`only: [deepinfra]`) fixes the vendor but not the precision or the engine:
`deepinfra/bf16` caps output at 131 072 and `deepinfra/turbo` at 16 384, same declared
precision. Some slugs have one endpoint whose tag carries no quantization suffix
(`mistral`, `novita`, `anthropic`) — that is the provider's labelling, and pinning it is
still correct.

Keep `quantizations:` alongside `only:` as a redundant assertion: if the tag is ever
requantized under the same name, the call fails instead of silently serving something else.

## 2. Prefer the longest generation length

Output caps vary by 60× across endpoints of one slug. A short cap truncates chain-of-thought
mid-reasoning and is scored anyway, so this outranks rule 3 — take fp8 over bf16 if bf16 only
comes with a cap you'd hit. `generate_args.max_tokens` must fit the pinned endpoint's
`max_completion_tokens`.

Compare declared caps only. An endpoint that declares no `max_completion_tokens` is unknown,
not unlimited, and its `context_length` is not the same quantity.

## 3. Prefer higher precision

bf16/fp16 over fp8 over fp4/mxfp4/nvfp4, all else equal — and all else is rarely equal, see
rule 2. Quantization labels are self-reported and unaudited; 31% of endpoints declare none at
all. An undeclared label is not evidence of high precision.

## 4. Set reasoning explicitly

For models where reasoning is not enable by default, remember to enable it manually if available
e.g. gemma, qwen, etc...

```yaml
generate_args:
  extra_body:
    reasoning:
      enabled: true
```

## Deliberate exceptions

A `# deliberate:` comment in a config marks a knowing departure from the rules. Two examples,
both because rule 2 beats rule 3:

- **`moonshotai/kimi-k3.yml`** — pinned to `moonshotai/mxfp4`, a 4-bit endpoint, though
  `deepinfra/bf16` and `baseten/fp8` exist. Their caps are 16 384 and 262 144 against the
  1 032 192 this config needs. Report k3 numbers as measured at 4-bit.
- **`nvidia/nemotron-3-ultra-550b-a55b.yml`** — pinned to `together`, which declares no
  quantization. Every endpoint that does declare one is lower precision, and together is the
  only one with room for 495 904 tokens. The BF16 in the model name is unverified.

## Verifying after the fact

The provider that actually served each response is stored in the eval logs; check it there.