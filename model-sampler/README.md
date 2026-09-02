# model-sampler

Seeded random sampling of a data-collection model set from a
capability-tier × release-half grid, plus a selection dashboard.

Rebuild the pool (base list from interactive_viewer/model_select.html
+ OpenRouter prices + Epoch models DB + imputed ECI + GPQA transcript
lengths from external_data/epoch-gpqa), which also re-injects it into
index.html:

    uv run python model-sampler/build_model_pool.py [--refresh]

Open the dashboard:

    python3 -m http.server --directory model-sampler 8771

Checks: `node model-sampler/sampler.js` and
`cd model-sampler && uv run pytest test_build_model_pool.py`.

Hand-assigned tiers: set in the dashboard (Tier overrides panel),
`Save overrides ↓`, drop the file at model-sampler/tier_overrides.json,
rebuild.

Save set ↓ downloads model_set.json (+ csv) with the seed and settings
that produced it; Load set ↑ restores selection, settings, and overrides
from that json.
