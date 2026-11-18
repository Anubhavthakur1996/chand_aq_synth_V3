"""Orchestration: one run = (dataset, arm, seed).  Also epoch tuning, reproducibility check, summary."""
import json, os, time
import numpy as np
import pandas as pd
import config as C
from . import windows as W
from . import evaluate as E
from . import train_dgan as T
from .preprocess import load_processed


# ----------------------------------------------------------------------------- data bundle
class Bundle:
    def __init__(self, dataset):
        self.dataset = dataset
        self.cfg = C.DATASETS[dataset]
        self.df = load_processed(dataset)
        self.seg = self.df["segment"].to_numpy()
        self.n_train, self.n_val = self.cfg["train_rows"], self.cfg["val_rows"]
        self.test_start = self.n_train + self.n_val

    def cols(self, features_key):
        base = list(self.cfg["base_cols"])
        return base if features_key == "base" else base + C.TIME_COLS

    def values(self, cols):
        return self.df[cols].to_numpy(dtype=float)

    def train_windows(self, cols, stride=None, return_starts=False):
        v = self.values(cols)
        return W.make_windows(v[:self.n_train], self.seg[:self.n_train], C.SEQ_LEN,
                              stride or C.TRAIN_STRIDE, return_starts=return_starts)

    def start_attributes(self, starts):
        """Encoded (start hour, start weekday) of training windows; starts = row indices in the processed table."""
        idx = self.df.index[np.asarray(starts)]
        return T.encode_start(idx.hour.to_numpy(), idx.dayofweek.to_numpy())

    def reference_windows(self, cols):
        """(train days, held-out days), non-overlapping 24-h windows of real data."""
        v = self.values(cols)
        tr = W.make_windows(v[:self.n_train], self.seg[:self.n_train], C.SEQ_LEN, C.SEQ_LEN)
        ho = None
        if self.n_val > 0:
            a, b = self.n_train, self.n_train + self.n_val
            ho = W.make_windows(v[a:b], self.seg[a:b], C.SEQ_LEN, C.SEQ_LEN)
        return tr, ho


def _out_dir(dataset, *parts):
    d = os.path.join(C.RESULTS_DIR, dataset, *parts)
    os.makedirs(d, exist_ok=True)
    return d


def chosen_short_epochs():
    p = os.path.join(C.RESULTS_DIR, "chosen_epochs.json")
    if os.path.exists(p):
        return int(json.load(open(p))["chosen_epochs"])
    print(f"[warn] no tuning result found; using default short budget {C.EPOCHS_SHORT_DEFAULT} epochs")
    return C.EPOCHS_SHORT_DEFAULT


def epochs_for(arm):
    return C.EPOCHS_LONG if C.ARMS[arm]["epochs"] == "long" else chosen_short_epochs()


# ----------------------------------------------------------------------------- one run
def run_single(dataset, arm, seed, force=False, epochs=None, n_sequences=None, tag=""):
    out_json = os.path.join(_out_dir(dataset, "runs"), f"{arm}_seed{seed}{tag}.json")
    if os.path.exists(out_json) and not force:
        print(f"[skip] {out_json} exists")
        return json.load(open(out_json))
    b = Bundle(dataset)
    spec = C.ARMS[arm]
    cols = b.cols(spec["features"])
    shared = list(b.cfg["base_cols"])
    epochs = int(epochs) if epochs is not None else epochs_for(arm)
    n_seq = int(n_sequences) if n_sequences is not None else C.N_SYNTH_SEQUENCES

    stride = spec.get("stride", C.TRAIN_STRIDE)
    attr_mode = spec.get("time_mode") == "attribute"
    train_win = b.train_windows(cols, stride)                 # evaluation view: all columns of the arm
    if attr_mode:                                             # GAN view: Base columns + start-time attributes
        gan_win, starts = b.train_windows(b.cols("base"), stride, return_starts=True)
        gan_attrs = b.start_attributes(starts)
    else:
        gan_win, gan_attrs = train_win, None
    if len(gan_win) < 2:
        raise RuntimeError("fewer than 2 training windows - check segments / TRAIN_STRIDE")
    print(f"[run] {dataset} {arm} seed={seed} epochs={epochs} train_windows={len(gan_win)} stride={stride} "
          f"time_mode={'attribute' if attr_mode else 'generated'} features={len(cols)}")

    synth, clock, secs = T.generate_synthetic(gan_win, cols, epochs, n_seq, seed, attributes=gan_attrs)
    
    print(f"[run] GAN training + generation done in {secs:.0f}s -> now evaluating", flush=True) 
    t_eval = time.time()
    
    np.savez_compressed(os.path.join(_out_dir(dataset, "synthetic"), f"{arm}_seed{seed}{tag}.npz"),
                        synth=synth.astype(np.float32), cols=np.array(cols))

    rng = np.random.default_rng(seed)
    ref_train, ref_held = b.reference_windows(cols)
    res = {"dataset": dataset, "arm": arm, "tag": tag, "seed": seed, "epochs": epochs, "features": cols,
           "stride": stride, "time_mode": "attribute" if attr_mode else "generated",
           "n_train_windows": int(len(train_win)), "n_synthetic_sequences": int(len(synth)),
           "train_generate_seconds": round(secs, 1), "clock_diagnostics_before_projection": clock}
    res["fidelity_vs_train"] = E.fidelity_block(ref_train, synth, cols, shared)
    
    print(f"[eval] fidelity metrics done ({time.time() - t_eval:.0f}s)", flush=True)

    if ref_held is not None and len(ref_held):
        res["fidelity_vs_heldout"] = E.fidelity_block(ref_held, synth, cols, shared)
    # memorisation vs the 13 non-overlapping training days: identical to paper 2 for stride-24 arms and
    # comparable across arms; for sliding-window arms the distance to ALL training windows is added.
    res["memorisation"] = E.nn_memorisation(ref_train, ref_held, synth, cols, shared, rng)
    if stride != C.SEQ_LEN:
        res["memorisation_all_windows"] = E.nn_memorisation(train_win, ref_held, synth, cols, shared, rng)

    values = b.values(cols)
    vols = [v for v in C.SYNTH_VOLUME_CURVE if v <= len(synth)] or [len(synth)]
    
    print("[eval] downstream utility (TRTR, TSTR per volume, shuffled control):", flush=True)
    
    res["utility_primary"] = E.utility_block(values, b.seg, b.n_train, b.test_start, cols, shared,
                                             synth, rng, vols, C.PRIMARY_VOLUME, shuffle_volume=C.SHUFFLE_VOLUME)
    if spec["features"] == "base_time":
        res["utility_secondary_all_columns"] = E.utility_block(values, b.seg, b.n_train, b.test_start, cols, cols,
                                                               synth, rng, [C.PRIMARY_VOLUME], C.PRIMARY_VOLUME, do_shuffle=False)
    with open(out_json, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[done] {out_json}  (evaluation took {time.time() - t_eval:.0f}s)", flush=True)
    return res


# ----------------------------------------------------------------------------- tuning
def tune(dataset="chandigarh", force=False):
    """Sweep short budgets.  Selection criterion: mean normalised EMD (shared columns) between
    synthetic data and HELD-OUT real days (the validation block), averaged over TUNE_SEEDS,
    on the Base arm only.  The test block is never used."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = Bundle(dataset)
    if b.n_val == 0:
        raise RuntimeError("tuning needs a validation block (val_rows > 0)")
    d = _out_dir(dataset, "tuning")
    path = os.path.join(d, "tuning_runs.csv")
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) and not force else []
    done = {(r["arm"], int(r["epochs"]), int(r["seed"])) for r in rows}
    shared = list(b.cfg["base_cols"])
    for arm in C.TUNE_ARMS:
        cols = b.cols(C.ARMS[arm]["features"])
        train_win = b.train_windows(cols)
        _, ref_held = b.reference_windows(cols)
        for ep in C.TUNE_EPOCHS:
            for seed in C.TUNE_SEEDS:
                if (arm, ep, seed) in done:
                    continue
                print(f"[tune] {arm} epochs={ep} seed={seed}")
                synth, _, secs = T.generate_synthetic(train_win, cols, ep, C.TUNE_N_SYNTH, seed)
                fb = E.fidelity_block(ref_held, synth, cols, shared)["shared_cols"]
                nn = E.nn_memorisation(train_win, ref_held, synth, cols, shared, np.random.default_rng(seed))
                rows.append({"arm": arm, "epochs": ep, "seed": seed, "seconds": round(secs, 1),
                             "emd_norm": fb["emd_norm"], "js_distance": fb["js_distance"],
                             "delta_acf": fb["delta_acf"], "frobenius": fb["frobenius"],
                             "nn_ratio": nn.get("ratio", np.nan)})
                pd.DataFrame(rows).to_csv(path, index=False)      # save after every run (resumable)
    df = pd.DataFrame(rows)
    summ = df.groupby(["arm", "epochs"]).agg(emd_mean=("emd_norm", "mean"), emd_std=("emd_norm", "std"),
                                             acf_mean=("delta_acf", "mean"), nn_ratio_mean=("nn_ratio", "mean")).reset_index()
    summ.to_csv(os.path.join(d, "tuning_summary.csv"), index=False)
    sel = summ[summ["arm"] == C.TUNE_SELECT_ARM].sort_values(["emd_mean", "epochs"])
    chosen = int(sel.iloc[0]["epochs"])
    json.dump({"chosen_epochs": chosen, "selected_on_arm": C.TUNE_SELECT_ARM,
               "criterion": "mean normalised EMD (shared columns) vs held-out real days",
               "tune_seeds": C.TUNE_SEEDS, "grid": C.TUNE_EPOCHS},
              open(os.path.join(C.RESULTS_DIR, "chosen_epochs.json"), "w"), indent=1)
    plt.figure(figsize=(6, 4))
    for arm in C.TUNE_ARMS:
        s = summ[summ["arm"] == arm]
        plt.errorbar(s["epochs"], s["emd_mean"], yerr=s["emd_std"].fillna(0), marker="o", capsize=3, label=arm)
    plt.axvline(chosen, ls="--", c="grey")
    plt.xscale("log"); plt.xlabel("training epochs"); plt.ylabel("normalised EMD vs held-out real days")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(d, "epochs_vs_fidelity.png"), dpi=200)
    print(f"[tune] chosen short budget = {chosen} epochs")
    return chosen


# ----------------------------------------------------------------------------- reproducibility
def repro_check(dataset="chandigarh"):
    b = Bundle(dataset)
    cols = b.cols("base")
    tw = b.train_windows(cols)
    a = T.fit_and_generate(tw, 30, 20, seed=7)
    a2 = T.fit_and_generate(tw, 30, 20, seed=7)
    c = T.fit_and_generate(tw, 30, 20, seed=8)
    same = bool(np.allclose(a, a2, atol=1e-6))
    diff = not np.allclose(a, c, atol=1e-6)
    print(f"same seed twice identical: {same} (max abs diff {np.abs(a - a2).max():.3g}); different seed differs: {diff}")
    if not same:
        print("-> NOT bit-reproducible on this machine (often GPU non-determinism). Report seeds AND this caveat, "
              "or run on CPU.")
    return same and diff


# ----------------------------------------------------------------------------- summary
def _flat(r):
    p = r["utility_primary"]
    pv = str(max(int(k) for k in p["tstr"]))
    row = {"arm": r["arm"] + r.get("tag", ""), "seed": r["seed"], "epochs": r["epochs"], "train_windows": r["n_train_windows"],
           "seconds": r["train_generate_seconds"]}
    for tag, key in (("train", "fidelity_vs_train"), ("heldout", "fidelity_vs_heldout")):
        if key in r:
            for k, v in r[key]["shared_cols"].items():
                row[f"{k}_{tag}"] = v
    row["nn_ratio"] = r["memorisation"].get("ratio", np.nan)
    for m in ("rf", "gbm"):
        row[f"tstr_{m}_mae"] = p["tstr"][pv][m]["mae"]
        row[f"trtr_{m}_mae"] = p["trtr"][m]["mae"]
        row[f"urr_{m}"] = p["urr"][pv][m]
        row[f"shuffle_{m}_mae"] = p["shuffle_control"][m]["mae"]
        sv = str(p.get("shuffle_volume"))
        row[f"shuffle_matched_tstr_{m}_mae"] = p["tstr"].get(sv, {}).get(m, {}).get("mae", np.nan)
    for v, d in p["tstr"].items():
        row[f"tstr_rf_mae_vol{v}"] = d["rf"]["mae"]
    if "utility_secondary_all_columns" in r:
        s = r["utility_secondary_all_columns"]
        sv = str(max(int(k) for k in s["tstr"]))
        row["secondary_rf_mae"] = s["tstr"][sv]["rf"]["mae"]
        row["secondary_urr_rf"] = s["urr"][sv]["rf"]
    cd = r.get("clock_diagnostics_before_projection", {}).get("hour")
    if cd:
        row["clock_radius_sq_mean"] = cd["radius_sq_mean"]
        row["clock_share_steps_1h"] = cd["share_steps_0.5_to_1.5h"]
    return row


def summarize(dataset):
    d = os.path.join(C.RESULTS_DIR, dataset, "runs")
    files = sorted(f for f in os.listdir(d) if f.endswith(".json")) if os.path.isdir(d) else []
    if not files:
        print(f"[summary] no runs for {dataset}")
        return
    df = pd.DataFrame([_flat(json.load(open(os.path.join(d, f)))) for f in files])
    out = _out_dir(dataset)
    df.to_csv(os.path.join(out, "all_runs.csv"), index=False)
    metrics = [c for c in df.columns if c not in ("arm", "seed", "epochs", "train_windows")]
    g = df.groupby("arm")[metrics]
    mean, std = g.mean(), g.std(ddof=1)
    summary = mean.join(std, lsuffix="_mean", rsuffix="_std")
    summary.insert(0, "n_seeds", df.groupby("arm").size())
    summary.to_csv(os.path.join(out, "summary.csv"))
    key = [c for c in ["emd_norm_heldout", "emd_norm_train", "delta_acf_heldout", "frobenius_heldout", "nn_ratio",
                       "tstr_rf_mae", "trtr_rf_mae", "urr_rf", "tstr_gbm_mae", "shuffle_rf_mae", "secondary_rf_mae"] if c in df]
    lines = [f"Dataset: {dataset}   runs: {len(df)}   (mean +/- std over seeds; lower is better except nn_ratio)\n"]
    for arm, grp in df.groupby("arm"):
        lines.append(f"{arm}  (n={len(grp)}, epochs={[int(e) for e in sorted(grp['epochs'].unique())]})")
        for k in key:
            lines.append(f"   {k:22s} {grp[k].mean():9.4f} +/- {grp[k].std(ddof=1) if len(grp) > 1 else float('nan'):7.4f}")
    pairs = [("a2_time_long", "a1_base_long"), ("a3_base_short", "a1_base_long"), ("a4_time_short", "a1_base_long"),
             ("a4_time_short", "a2_time_long"), ("a4_time_short", "a3_base_short"),
             ("n1_cond_short", "a3_base_short"), ("n1_cond_short", "a4_time_short"),
             ("n2_slide_short", "a3_base_short"), ("n3_condslide_short", "a3_base_short"),
             ("n3_condslide_short", "a1_base_long"), ("n3_condslide_short", "n1_cond_short")]
    lines.append("\nPaired differences over seeds (first minus second; negative = first is better for error metrics)")
    for a, bb in pairs:
        A, B = df[df["arm"] == a].set_index("seed"), df[df["arm"] == bb].set_index("seed")
        common = A.index.intersection(B.index)
        if len(common) == 0:
            continue
        lines.append(f"{a} - {bb}  (n={len(common)})")
        for k in [c for c in ("emd_norm_heldout", "tstr_rf_mae", "urr_rf") if c in df]:
            dd = (A.loc[common, k] - B.loc[common, k])
            lines.append(f"   {k:22s} {dd.mean():9.4f} +/- {dd.std(ddof=1) if len(dd) > 1 else float('nan'):7.4f}")
    txt = "\n".join(lines)
    open(os.path.join(out, "summary.txt"), "w").write(txt)
    print(txt)
