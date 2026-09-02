#!/bin/bash
# Low-resource Qwen3-8B shard fetch for the gateway: single niced curl,
# rate-limited, resumable, verifies byte size before moving into place.
set -u
D=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/models/Qwen3-8B
URL=https://huggingface.co/Qwen/Qwen3-8B/resolve/main

declare -A WANT=(
  [model-00001-of-00005.safetensors]=3996250744
  [model-00002-of-00005.safetensors]=3993160032
  [model-00003-of-00005.safetensors]=3959604768
  [model-00004-of-00005.safetensors]=3187841392
  [model-00005-of-00005.safetensors]=1244659840
)

for attempt in $(seq 1 30); do
  missing=0
  for F in "${!WANT[@]}"; do
    have=$(stat -c %s "$D/$F" 2>/dev/null || echo 0)
    [ "$have" = "${WANT[$F]}" ] && continue
    missing=1
    echo "[$attempt] fetching $F (have partial: $(stat -c %s "$D/.fetch-$F" 2>/dev/null || echo 0))"
    curl -sS -L -C - --retry 5 --retry-delay 5 \
      -o "$D/.fetch-$F" "$URL/$F" || true
    got=$(stat -c %s "$D/.fetch-$F" 2>/dev/null || echo 0)
    if [ "$got" = "${WANT[$F]}" ]; then
      mv "$D/.fetch-$F" "$D/$F"
      echo "[$attempt] $F done"
    fi
  done
  [ "$missing" = 0 ] && { echo FETCH_DONE; exit 0; }
  sleep 5
done
echo FETCH_GAVE_UP
exit 1
