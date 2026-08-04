"""Train the twin-MLP scorer and both baselines across CV folds.

Per (regime, fold): grouped inner validation carved from the training
side; one TwinScorer per config-grid option, early stopping on inner-val
NLL; the best option's held-out predictions land in
preds/{regime}_{fold}.parquet (model = mlp | position | dlogit) beside
the ml_suite models. Checkpoints per fold under checkpoints/.

Usage:
  uv run python scripts/analyses/score_prediction/train.py [--smoke]
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from cv_folds import grouped_folds
from ml_suite import grid_options, load_features, log_loss, merge_preds
from model import (TwinScorer, fit_dlogit, fit_position, predict_dlogit,
                   predict_position, symmetry_check)

ROOT = Path(__file__).resolve().parents[3]


def calculate_dlogit(rows: pd.DataFrame, models_csv: Path) -> pd.DataFrame:
    """Capability-gap column: logit(model in slot 1) - logit(slot 2)."""
    logit = pd.read_csv(models_csv).set_index("model")["logit"]
    slot1 = np.where(rows["slot_of_major"] == "1", rows["major"], rows["minor"])
    slot2 = np.where(rows["slot_of_major"] == "1", rows["minor"], rows["major"])
    dlogit = logit.reindex(slot1).to_numpy() - logit.reindex(slot2).to_numpy()
    assert not np.isnan(dlogit).any(), "model missing from models.csv"
    return rows.assign(dlogit=dlogit)


def train_scorer(e1, e2, judge_idx, y, val_mask, n_judges, params, seed):
    """One fit with early stopping on the inner validation.
    Returns (model in eval mode, best val nll)."""
    torch.manual_seed(seed)
    model = TwinScorer(e1.shape[1], n_judges, params["judge_embed_dim"],
                       params["hidden_dims"], params["dropout"])
    # No weight decay on b/tau: shrinking position bias or temperature
    # toward zero is not regularisation, it is a modelling error.
    decay = [p for name, p in model.named_parameters() if name not in ("b", "tau")]
    no_decay = [model.b, model.tau]
    opt = torch.optim.AdamW(
        [dict(params=decay, weight_decay=params["weight_decay"]),
         dict(params=no_decay, weight_decay=0.0)], lr=params["lr"])
    loss_fn = nn.BCEWithLogitsLoss()

    # Positions within this training subset: gradient steps vs the
    # early-stopping validation rows.
    train_idx = np.flatnonzero(~val_mask)
    val_idx = torch.from_numpy(np.flatnonzero(val_mask))
    e1_val, e2_val = e1[val_idx], e2[val_idx]
    judge_val, y_val = judge_idx[val_idx], y[val_idx]
    rng = np.random.default_rng(seed)
    best_nll, best_state, stale = np.inf, None, 0
    for _ in range(params["max_epochs"]):
        # One epoch: minibatch gradient steps over a fresh shuffle.
        model.train()
        rng.shuffle(train_idx)
        for start in range(0, len(train_idx), params["batch_size"]):
            batch = torch.from_numpy(train_idx[start:start + params["batch_size"]])
            opt.zero_grad()
            loss = loss_fn(model(e1[batch], e2[batch], judge_idx[batch]), y[batch])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_nll = float(loss_fn(model(e1_val, e2_val, judge_val), y_val))
        # Early stopping: remember the weights from the best validation
        # epoch; give up after `patience` epochs without improvement.
        if val_nll < best_nll - 1e-5:
            best_nll, best_state, stale = val_nll, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= params["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_nll


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="items fold 0 only, 3 epochs, first grid option")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]
    seed = cfg["seeds"][0]
    symmetry_check(seed=seed)

    # Row-aligned arrays: e1/e2 are the slot-1/slot-2 transcript embeddings
    # of each verdict row; judge_idx codes judges 0..n-1 in judge_names order.
    rows, e1_np, e2_np, judge_names, judge_idx_np, y_np = load_features(out)
    rows = calculate_dlogit(rows, ROOT / cfg["models_csv"])
    e1, e2 = torch.from_numpy(e1_np), torch.from_numpy(e2_np)
    judge_idx = torch.from_numpy(judge_idx_np.astype(np.int64))
    y = torch.from_numpy(y_np.astype(np.float32))

    # row_id -> fold number, one column per regime.
    folds = pd.read_parquet(out / "cv_folds.parquet")
    fold_of_row = folds.pivot(index="row_id", columns="regime", values="fold")
    preds_dir = out / "preds"
    preds_dir.mkdir(exist_ok=True)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Cartesian product of the *_grid entries in the mlp config block.
    grid = grid_options(cfg["mlp"])
    group_cols = {"items": "item", "pairs": "pair_key", "judges": "judge"}
    if args.smoke:
        grid = grid[:1]
        grid[0] = dict(grid[0], max_epochs=3, patience=2)
        group_cols = {"items": "item"}

    for regime, group_col in group_cols.items():
        regime_fold = rows["row_id"].map(fold_of_row[regime]).to_numpy()
        for fold in sorted(np.unique(regime_fold)):
            if args.smoke and fold > 0:
                continue
            test_mask = regime_fold == fold
            train_pos = np.flatnonzero(~test_mask)
            test_pos = torch.from_numpy(np.flatnonzero(test_mask))
            train_rows = rows[~test_mask]
            # Inner validation: regroup the training rows by the regime's
            # group column, hold inner fold 0 out for early stopping and
            # grid selection. The test fold is never touched.
            val_mask = (grouped_folds(train_rows, group_col,
                                      cfg["inner_folds"], seed) == 0).to_numpy()
            train_t = tuple(t[torch.from_numpy(train_pos)]
                            for t in (e1, e2, judge_idx, y))

            # One scorer per grid option; keep the best inner-val NLL.
            best = None
            for params in grid:
                model, val_nll = train_scorer(*train_t, val_mask=val_mask,
                                              n_judges=len(judge_names),
                                              params=params, seed=seed)
                if best is None or val_nll < best[1]:
                    best = (model, val_nll, params)
            model, _, chosen = best

            with torch.no_grad():
                # Judges regime is leave-one-judge-out: the test judge was
                # never trained on, so score without its parameters.
                if regime == "judges":
                    logits = model.score_unseen_judge(e1[test_pos], e2[test_pos])
                else:
                    logits = model(e1[test_pos], e2[test_pos], judge_idx[test_pos])
                p_mlp = torch.sigmoid(logits).numpy()

            # Baselines refit on exactly the same training rows.
            b_by_judge = fit_position(train_rows)
            dlogit_params = fit_dlogit(train_rows)
            test_rows = rows[test_mask]
            preds = {"mlp": p_mlp,
                     "position": predict_position(b_by_judge, test_rows),
                     "dlogit": predict_dlogit(dlogit_params, test_rows)}
            frames = [pd.DataFrame(dict(
                row_id=test_rows["row_id"].values, model=name, regime=regime,
                fold=fold, p_slot1=p, y=y_np[test_mask])) for name, p in preds.items()]
            merge_preds(preds_dir / f"{regime}_{fold}.parquet", frames)
            torch.save(dict(state_dict=model.state_dict(), params=chosen,
                            judge_names=judge_names),
                       ckpt_dir / f"{regime}_{fold}.pt")
            print(f"{regime} fold {fold}: " + "  ".join(
                f"{name} nll {log_loss(y_np[test_mask], p):.4f}"
                for name, p in preds.items()) + f"  chose {chosen}", flush=True)

    if args.smoke:
        print("smoke run done")
        return

    test_mask = rows["row_id"].map(fold_of_row["items"]).to_numpy() == 0
    train_rows = rows[~test_mask]
    val_mask = (grouped_folds(train_rows, "item", cfg["inner_folds"],
                              cfg["seeds"][1]) == 0).to_numpy()
    train_t = tuple(t[torch.from_numpy(np.flatnonzero(~test_mask))]
                    for t in (e1, e2, judge_idx, y))
    chosen = torch.load(ckpt_dir / "items_0.pt", weights_only=False)["params"]
    model, _ = train_scorer(*train_t, val_mask=val_mask,
                            n_judges=len(judge_names), params=chosen,
                            seed=cfg["seeds"][1])
    test_pos = torch.from_numpy(np.flatnonzero(test_mask))
    with torch.no_grad():
        p_repeat = torch.sigmoid(model(e1[test_pos], e2[test_pos],
                                       judge_idx[test_pos])).numpy()
    first = pd.read_parquet(preds_dir / "items_0.parquet").query("model == 'mlp'")
    print(f"seed stability, items fold 0: seed {cfg['seeds'][0]} nll "
          f"{log_loss(first['y'], first['p_slot1']):.4f} vs seed "
          f"{cfg['seeds'][1]} nll {log_loss(y_np[test_mask], p_repeat):.4f}")

    all_preds = pd.concat([pd.read_parquet(f)
                           for f in sorted(preds_dir.glob("*.parquet"))])
    summary = (all_preds.groupby(["regime", "model"])
               .apply(lambda g: pd.Series(dict(
                   nll=log_loss(g["y"], g["p_slot1"]),
                   acc=float((g["p_slot1"].round() == g["y"]).mean())))))
    print(f"\npooled held-out metrics (chance nll {np.log(2):.4f}):")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
