#!/usr/bin/env bash
# Re-collect qwen3.5-9b on AIME 24/25/26 off the broken bf16 pins.
#
# The committed 2026-08-12 aime logs are 53-73% empty completions (reasoning present, answer
# text blank), so their .267-.433 accuracies measure the empty rate, not the model. Re-running
# under the old deepinfra/bf16 pin is worse still (94-100% empty). See data/collection_runs.md.
#
# The model config now pins siliconflow/fp8. This script proves that pin on a few samples
# before it archives anything, so a bad provider cannot cost us the existing logs.
#
# Deliberately AIME-only: never pass an apex bench here, another agent owns that.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=configs/models/qwen/qwen3.5-9b.yml
BENCHES=(aime24 aime25 aime26)
EPOCHS=${EPOCHS:-2}
CONNECTIONS=${CONNECTIONS:-16}
SMOKE_LIMIT=${SMOKE_LIMIT:-10}
MAX_EMPTY=${MAX_EMPTY:-20}          # percent; healthy runs sit at 1-3%
SMOKE_DIR=$(mktemp -d /tmp/qwen35_9b_smoke.XXXXXX)
ARCHIVE=data/legacy/qwen35-9b_aime_empty_$(date +%Y%m%d)

echo "==> pin in use:"
grep -A3 '^  provider:' "$MODEL" | sed 's/^/    /'

# --- 1. smoke test -----------------------------------------------------------------------
# --limit runs never count as successful (is_successful_eval), so this cannot be mistaken
# for real data even though it lands in a throwaway dir.
echo "==> smoke: $SMOKE_LIMIT samples of aime24 -> $SMOKE_DIR"
uv run collect --fail-on-error 3 \
  -b configs/bench/aime24.yml -m "$MODEL" \
  -n 1 --limit "$SMOKE_LIMIT" --log-dir "$SMOKE_DIR" --max-connections "$CONNECTIONS"

echo "==> smoke result"
uv run python - "$SMOKE_DIR" "$MAX_EMPTY" <<'PY'
import glob, sys, statistics as st
from collections import Counter
from inspect_ai.log import read_eval_log_samples

files = glob.glob(f"{sys.argv[1]}/**/*.eval", recursive=True)
if not files:
    sys.exit("no smoke log written - collect failed before producing samples")

n = empty = 0
toks, prov = [], Counter()
for f in files:
    for s in read_eval_log_samples(f, all_samples_required=False):
        n += 1
        if not ((s.output.completion if s.output else "") or "").strip():
            empty += 1
        if s.output and s.output.usage:
            toks.append(s.output.usage.output_tokens)
        for ev in s.events or []:
            call = getattr(ev, "call", None)
            if type(ev).__name__ == "ModelEvent" and call and (call.response or {}).get("provider"):
                prov[call.response["provider"]] += 1
                break

pct = 100 * empty / n if n else 100
print(f"    samples={n} empty={empty} ({pct:.0f}%) "
      f"median_out={st.median(toks) if toks else 0:.0f} providers={dict(prov)}")
if pct > float(sys.argv[2]):
    sys.exit(f"    ABORT: {pct:.0f}% empty exceeds {sys.argv[2]}% - this pin is no better "
             f"than the old one. Nothing archived, old logs untouched. Try another provider: "
             f"siliconflow/fp8 (fp8, 262144 cap) is the next candidate; venice/fp8 caps "
             f"output at 32768, and both bf16 endpoints are the ones already ruled out.")
print("    pin looks healthy")
PY
rm -rf "$SMOKE_DIR"

# --- 2. archive the old logs -------------------------------------------------------------
# collect resumes a repeat dir whenever any .eval in it succeeded, and these do - they are
# "successful" runs that happen to be mostly empty. They have to move or the full run no-ops.
echo "==> archiving old aime logs -> $ARCHIVE"
mkdir -p "$ARCHIVE"
moved=0
for bench in aime2024 aime2025 aime2026; do
  for rep in data/$bench/*/qwen3.5-9b/*/; do
    [ -d "$rep" ] || continue
    dest="$ARCHIVE/${rep#data/}"
    mkdir -p "$dest"
    # the eval-set bookkeeping has to go too, or resume follows it back to the moved logs
    find "$rep" -maxdepth 1 -type f \
      \( -name '*.eval' -o -name 'eval-set.json' -o -name 'logs.json' -o -name 'run.log' -o -name '.eval-set-id' \) \
      -exec mv -t "$dest" {} + && moved=$((moved + 1))
  done
done
echo "    archived $moved repeat dirs"

# --- 3. full run -------------------------------------------------------------------------
benches=("${BENCHES[@]/#/configs/bench/}")
benches=("${benches[@]/%/.yml}")
echo "==> collecting ${BENCHES[*]} x $EPOCHS epochs"
exec uv run collect --fail-on-error 3 \
  -b "${benches[@]}" -m "$MODEL" \
  -n "$EPOCHS" --log-dir data --max-connections "$CONNECTIONS" "$@"
