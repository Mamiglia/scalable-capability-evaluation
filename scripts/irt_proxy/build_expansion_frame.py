"""Build the GLM verdict frame from the local data-expansion-gpqa viewer data.

Verdicts (exact picks + slot orientation) from pairs/*.json, shown transcripts
from transcripts/<model>.json for length + style features, capability from the
published gap_pairs GPQA accuracy. Keeps effort variants (deepseek-v4-flash-0731,
gpt-5.5-low, o3-mini-medium, ...) as compared models. Left minus right in shown
order. No reasoning filter (we want every candidate).
"""
import json, os, glob, sys
import numpy as np, pandas as pd

AW = "/private/tmp/claude-501/-Users-moriohamada-PycharmProjects-ai-safety-small-judge/c8dce26e-388b-4d6c-a32a-b130c6ae452e/scratchpad/analysis-wt"
sys.path.insert(0, AW + "/scripts/hibayes")
from style_features import feature_rates, load_lexicon

DE = "/Users/moriohamada/PycharmProjects/ai_safety/small-judge/interactive_dashboards/transcript_viewer/data/data-expansion-gpqa"
GP = "/Users/moriohamada/PycharmProjects/ai_safety/small-judge/judge_results/glm_correction/tables/gap_pairs.csv"
OUT = "/private/tmp/claude-501/-Users-moriohamada-PycharmProjects-ai-safety-small-judge/65e871a7-3123-4881-8df7-bceb26a9b7c8/scratchpad/expansion_gpqa_glm.csv"
STYLE = ["personal_pronouns", "hedging", "backtracking", "uncertainty", "confidence", "questions"]
lexicon = load_lexicon(AW + "/scripts/hibayes/style_lexicon.yaml")

# capability: per-model GPQA accuracy from published gap_pairs observed
gp = pd.read_csv(GP, usecols=["model_left", "model_right", "observed_left", "observed_right"])
acc = pd.concat([gp[["model_left", "observed_left"]].set_axis(["m", "a"], axis=1),
                 gp[["model_right", "observed_right"]].set_axis(["m", "a"], axis=1)]).groupby("m").a.mean().to_dict()

# authoritative major/minor per unordered pair from index runs
idx = json.load(open(f"{DE}/index.json"))
majmin = {frozenset((r["major"], r["minor"])): (r["major"], r["minor"]) for r in idx["runs"]}

_tr = {}
def transcript(model, item):
    if model not in _tr:
        _tr[model] = json.load(open(f"{DE}/transcripts/{model}.json"))
    t = _tr[model].get(item)
    return t[len("[Assistant]: "):] if t and t.startswith("[Assistant]: ") else t

_feat = {}
def features(model, item):
    """(chars, {style rates}) for a shown transcript, cached per (model, item)."""
    key = (model, item)
    if key not in _feat:
        t = transcript(model, item)
        _feat[key] = None if t is None else (len(t), feature_rates(t, lexicon))
    return _feat[key]

def family(m):
    return m.split("-")[0].lower()

rows, skipped = [], {"no_pair_meta": 0, "no_transcript": 0, "no_capability": 0, "bad_pick": 0}
for pf in sorted(glob.glob(f"{DE}/pairs/*__full.json")):
    a, b = os.path.basename(pf).split("__")[:2]
    key = frozenset((a, b))
    if key not in majmin:
        skipped["no_pair_meta"] += 1
        continue
    major, minor = majmin[key]
    for item, judges in json.load(open(pf)).items():
        for judge, jd in judges.items():
            for od in jd["orders"]:
                som, pick = od["slot_of_major"], od["pick"]
                if pick not in ("1", "2") or som not in ("1", "2"):
                    skipped["bad_pick"] += 1
                    continue
                left, right = (major, minor) if som == "1" else (minor, major)
                lf, rf = features(left, item), features(right, item)
                if lf is None or rf is None:
                    skipped["no_transcript"] += 1
                    continue
                if left not in acc or right not in acc:
                    skipped["no_capability"] += 1
                    continue
                lc, rc = acc[left], acc[right]
                lchars, lr = lf
                rchars, rr = rf
                row = {"chosen_left": int(pick == "1"), "judge": judge, "item": item,
                       "left_model": left, "right_model": right,
                       "left_capability": lc, "right_capability": rc,
                       "length_difference": np.log1p(lchars) - np.log1p(rchars),
                       "capability_difference": lc - rc,
                       "same_model_family": int(left != right and family(left) == family(right)),
                       "self_comparison": int(left == right)}
                for c in STYLE:
                    row[f"{c}_difference"] = lr[c] - rr[c]
                rows.append(row)

frame = pd.DataFrame(rows)
for col in ["length_difference", "capability_difference"] + [f"{c}_difference" for c in STYLE]:
    frame[f"{col}_sd"] = (frame[col] - frame[col].mean()) / frame[col].std(ddof=0)
frame.to_csv(OUT, index=False)
print(f"wrote {len(frame):,} verdicts to {OUT}")
print("skipped:", skipped)
print("judges:", frame.judge.nunique(), "| candidates:", len(set(frame.left_model) | set(frame.right_model)),
      "| pairs:", frame.groupby(['left_model', 'right_model']).ngroups, "| items:", frame.item.nunique())
print("chosen_left rate:", round(frame.chosen_left.mean(), 4))
cands = sorted(set(frame.left_model) | set(frame.right_model))
print("effort variants present:", [c for c in cands if c.endswith(('-low', '-high', '-medium'))])
print("deepseek-v4 present:", [c for c in cands if 'v4-flash' in c])
