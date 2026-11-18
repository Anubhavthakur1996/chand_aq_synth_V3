"""
Evaluation: (1) fidelity of the synthetic data, (2) memorisation check, (3) downstream utility
(TSTR / TRTR / URR) and the shuffled negative control.

Conventions
  * all fidelity numbers are computed on a feature set given explicitly ("arm" columns or the
    "shared" columns that exist in every arm, so arms are comparable);
  * EMD is divided by the real std of the feature (so no variable dominates by its unit);
  * ACF is computed INSIDE each 24-h sequence (never across the seam between two synthetic days);
  * the Jensen-Shannon value is the base-2 JS *distance* (sqrt of the divergence), range 0..1.
"""
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon, cdist
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config as C
from . import windows as W


def _rows(w):
    return w.reshape(-1, w.shape[-1])


def acf_within(seqs, max_lag):
    N, L, F = seqs.shape
    K = max(1, min(max_lag, L - 2))
    out = np.zeros((F, K))
    for f in range(F):
        for k in range(1, K + 1):
            a = seqs[:, :-k, f].ravel(); b = seqs[:, k:, f].ravel()
            if a.std() > 1e-12 and b.std() > 1e-12:
                out[f, k - 1] = np.corrcoef(a, b)[0, 1]
    return out


def per_feature_fidelity(real_win, synth_seq, cols):
    R, S = _rows(real_win), _rows(synth_seq)
    acf_r, acf_s = acf_within(real_win, C.ACF_MAX_LAG), acf_within(synth_seq, C.ACF_MAX_LAG)
    rows = []
    for j, c in enumerate(cols):
        sd = R[:, j].std()
        sd = sd if sd > 1e-12 else 1.0
        emd = wasserstein_distance(R[:, j], S[:, j]) / sd
        lo, hi = min(R[:, j].min(), S[:, j].min()), max(R[:, j].max(), S[:, j].max())
        if hi - lo < 1e-12:
            js = 0.0
        else:
            bins = np.linspace(lo, hi, C.JS_BINS + 1)
            p, _ = np.histogram(R[:, j], bins); q, _ = np.histogram(S[:, j], bins)
            js = float(jensenshannon(p + 1e-12, q + 1e-12, base=2))
        rows.append({"feature": c, "emd_norm": float(emd), "js_distance": js,
                     "delta_acf": float(np.mean(np.abs(acf_r[j] - acf_s[j])))})
    return pd.DataFrame(rows)


def corr_frobenius(real_win, synth_seq, idx):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        cr = np.nan_to_num(np.corrcoef(_rows(real_win)[:, idx], rowvar=False))
        cs = np.nan_to_num(np.corrcoef(_rows(synth_seq)[:, idx], rowvar=False))
    return float(np.linalg.norm(cr - cs, "fro"))


def fidelity_block(real_win, synth_seq, cols, shared_cols):
    """Aggregates on the arm's own columns and on the shared columns, plus the per-feature table."""
    pf = per_feature_fidelity(real_win, synth_seq, cols)
    sh = pf[pf["feature"].isin(shared_cols)]
    idx_all = list(range(len(cols)))
    idx_sh = [cols.index(c) for c in shared_cols]
    agg = lambda d: {"emd_norm": float(d["emd_norm"].mean()), "js_distance": float(d["js_distance"].mean()),
                     "delta_acf": float(d["delta_acf"].mean())}
    return {
        "arm_cols": {**agg(pf), "frobenius": corr_frobenius(real_win, synth_seq, idx_all)},
        "shared_cols": {**agg(sh), "frobenius": corr_frobenius(real_win, synth_seq, idx_sh)},
        "per_feature": pf.round(5).to_dict(orient="records"),
    }


def nn_memorisation(train_win, heldout_win, synth_seq, cols, shared_cols, rng):
    """Mean distance from synthetic days to their nearest REAL TRAINING day, versus the same for
    unseen real days.  ratio < 1 => synthetic days sit closer to the training days than genuinely
    new real days do (a memorisation signal)."""
    idx = [cols.index(c) for c in shared_cols]
    base = _rows(train_win)[:, idx]
    mu, sd = base.mean(0), base.std(0)
    sd[sd < 1e-12] = 1.0
    prep = lambda w: ((w[:, :, idx] - mu) / sd).reshape(len(w), -1)
    T = prep(train_win)
    n = min(C.NN_SAMPLE, len(synth_seq))
    sel = rng.choice(len(synth_seq), n, replace=False)
    out = {"synth_to_train": float(cdist(prep(synth_seq[sel]), T).min(axis=1).mean())}
    if heldout_win is not None and len(heldout_win) > 0:
        out["heldout_to_train"] = float(cdist(prep(heldout_win), T).min(axis=1).mean())
        out["ratio"] = out["synth_to_train"] / out["heldout_to_train"]
    return out


# ------------------------------------------------------------------ downstream utility
def _models():
    return {"rf": RandomForestRegressor(n_estimators=C.RF_TREES, random_state=C.MODEL_SEED, n_jobs=C.RF_N_JOBS),
            "gbm": HistGradientBoostingRegressor(random_state=C.MODEL_SEED)}


def _fit_eval(Xtr, ytr, Xte, yte):
    out = {}
    for name, m in _models().items():
        t0 = time.time()
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        out[name] = {"mae": float(mean_absolute_error(yte, p)),
                     "rmse": float(np.sqrt(mean_squared_error(yte, p))),
                     "r2": float(r2_score(yte, p))}
        print(f"[eval]     {name}: trained on {len(ytr)} samples, tested on {len(yte)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return out


def utility_block(values, seg, n_train, test_start, cols, pred_cols, synth_seq, rng, volumes, primary_volume, shuffle_volume=None, do_shuffle=True):
    """TRTR (real train -> real test), TSTR at several synthetic volumes, URR = TSTR/TRTR MAE,
    and the shuffled-synthetic negative control, all on the SAME predictor columns."""
    pidx = [cols.index(c) for c in pred_cols]
    t = pred_cols.index(C.TARGET)
    vals = values[:, pidx]
    Xte, yte = W.lag_samples_flat(vals, seg, t, C.LOOKBACK, min_target_index=test_start)
    Xtr, ytr = W.lag_samples_flat(vals[:n_train], seg[:n_train], t, C.LOOKBACK)
    trtr = _fit_eval(Xtr, ytr, Xte, yte)
    tstr, urr = {}, {}
    for v in volumes:
        v = min(v, len(synth_seq))
        X, y = W.lag_samples_sequences(synth_seq[:v][:, :, pidx], t, C.LOOKBACK)
        tstr[str(v)] = _fit_eval(X, y, Xte, yte)
        urr[str(v)] = {m: tstr[str(v)][m]["mae"] / trtr[m]["mae"] for m in trtr}
        shuffle, sv = None, None
    if do_shuffle:
        sv = min(shuffle_volume or primary_volume, len(synth_seq))
        S = synth_seq[:sv][:, :, pidx]
        rows = S.reshape(-1, S.shape[2])
        S2 = rows[rng.permutation(len(rows))].reshape(S.shape)
        X, y = W.lag_samples_sequences(S2, t, C.LOOKBACK)
        shuffle = _fit_eval(X, y, Xte, yte)
    return {"pred_cols": pred_cols, "n_test_samples": int(len(yte)), "n_trtr_samples": int(len(ytr)),
            "trtr": trtr, "tstr": tstr, "urr": urr, "shuffle_control": shuffle,
            "shuffle_volume": sv}
