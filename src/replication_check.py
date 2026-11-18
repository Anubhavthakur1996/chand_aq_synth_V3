"""
Replication check against the base paper (Morales-Garcia et al. 2025), Bilbao data.

Question answered: "does our Base arm (a1_base_long) on the base paper's Bilbao data produce synthetic
data whose summary statistics are consistent with the synthetic data the base authors released?"

The base authors set no random seed, so exact agreement is not expected.  The fair criterion is
whether their released values fall inside (or close to) the spread produced by our seeds.

Inputs
  * our synthetic data   : results/bilbao/synthetic/a1_base_long_seed*.npz   (from `main.py run --dataset bilbao`)
  * the real Bilbao data : our processed table (same 350 hours the base paper used)
  * their synthetic data : the base repo's  3. AI models/1. Real vs Synth vs RealSynth/datasets/synth/60_Bilbao_synth.csv
                           (semicolon separated; pass its path with --base-synth)
Caveat to state in the paper: our Bilbao run uses the 17 reported variables only (no vehicle or 8-hour
columns), trains on the first 326 hours, and projects/clips as described in README.md, whereas the base
authors trained on all 23 columns.  The comparison is therefore "consistent in marginal statistics",
not "identical".
"""
import glob, os
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import config as C
from .preprocess import BILBAO_MAP, load_processed

STATS = ["mean", "std", "min", "25%", "50%", "75%", "max"]


def _describe(df):
    return df.describe().loc[STATS]


def run(base_synth_path, arm="a1_base_long", dataset="bilbao"):
    files = sorted(glob.glob(os.path.join(C.RESULTS_DIR, dataset, "synthetic", f"{arm}_seed*.npz")))
    if len(files) < 2:
        raise SystemExit(f"need >= 2 seeds of {arm} in results/{dataset}/synthetic/ (found {len(files)}). "
                         f"Run: python main.py run --dataset {dataset} --arms {arm}")
    if not os.path.exists(base_synth_path):
        raise SystemExit(f"base paper's synthetic file not found: {base_synth_path}")
    cols = list(C.DATASETS[dataset]["base_cols"])

    real = load_processed(dataset)[cols]
    theirs = pd.read_csv(base_synth_path, sep=";", index_col=0).rename(columns=BILBAO_MAP)
    missing = [c for c in cols if c not in theirs.columns]
    if missing:
        raise SystemExit(f"base synthetic file lacks columns {missing}")
    theirs = theirs[cols]

    ours = []
    for f in files:
        z = np.load(f)
        zc = [str(c) for c in z["cols"]]
        ours.append(pd.DataFrame(z["synth"].reshape(-1, len(zc)), columns=zc)[cols])
    seeds = [int(os.path.basename(f).split("seed")[1].split(".")[0]) for f in files]

    d_real, d_theirs = _describe(real), _describe(theirs)
    d_ours = np.stack([_describe(o).to_numpy() for o in ours])            # (seeds, stats, vars)
    m, s = d_ours.mean(0), d_ours.std(0, ddof=1)
    lo, hi = d_ours.min(0), d_ours.max(0)

    rows = []
    for j, v in enumerate(cols):
        for i, st in enumerate(STATS):
            b = float(d_theirs.loc[st, v])
            rows.append({"variable": v, "statistic": st, "real": float(d_real.loc[st, v]), "base_paper_synth": b,
                         "ours_mean": m[i, j], "ours_std": s[i, j], "ours_min": lo[i, j], "ours_max": hi[i, j],
                         "base_inside_our_range": bool(lo[i, j] <= b <= hi[i, j]),
                         "base_within_2sd_of_ours": bool(abs(b - m[i, j]) <= 2 * s[i, j]) if s[i, j] > 0 else bool(b == m[i, j])})
    long = pd.DataFrame(rows)
    out_dir = os.path.join(C.RESULTS_DIR, dataset, "replication_check")
    os.makedirs(out_dir, exist_ok=True)
    long.round(5).to_csv(os.path.join(out_dir, "stats_comparison.csv"), index=False)

    # distribution-level: how far is each synthetic set from the real data, and from each other
    emd = []
    for v in cols:
        sd = real[v].std() or 1.0
        e_theirs_real = wasserstein_distance(theirs[v], real[v]) / sd
        e_ours_real = np.mean([wasserstein_distance(o[v], real[v]) for o in ours]) / sd
        e_ours_theirs = np.mean([wasserstein_distance(o[v], theirs[v]) for o in ours]) / sd
        emd.append({"variable": v, "emd_theirs_vs_real": e_theirs_real, "emd_ours_vs_real": e_ours_real,
                    "emd_ours_vs_theirs": e_ours_theirs})
    emd = pd.DataFrame(emd)
    emd.round(5).to_csv(os.path.join(out_dir, "emd_comparison.csv"), index=False)

    central = long[long["statistic"].isin(["mean", "std", "25%", "50%", "75%"])]
    extreme = long[long["statistic"].isin(["min", "max"])]
    frac = lambda d, c: 100.0 * d[c].mean()
    txt = [
        f"Replication check on Bilbao  (our arm {arm}, seeds {seeds}; base-paper synthetic file: {os.path.basename(base_synth_path)})",
        f"variables compared: {len(cols)}",
        "",
        "Is the base paper's released value consistent with the spread of OUR seeds?",
        f"  central statistics (mean, std, quartiles): inside our [min,max]: {frac(central, 'base_inside_our_range'):.0f}%   "
        f"within +/-2 sd of our mean: {frac(central, 'base_within_2sd_of_ours'):.0f}%",
        f"  extremes (min, max):                       inside our [min,max]: {frac(extreme, 'base_inside_our_range'):.0f}%   "
        f"within +/-2 sd of our mean: {frac(extreme, 'base_within_2sd_of_ours'):.0f}%",
        "  (with only a few seeds the range is narrow, so low percentages are not by themselves a failure;",
        "   judge together with the distances below)",
        "",
        "Normalised EMD (distance / real std), mean over variables - lower = closer",
        f"  base paper synthetic vs real : {emd['emd_theirs_vs_real'].mean():.4f}",
        f"  our synthetic        vs real : {emd['emd_ours_vs_real'].mean():.4f}",
        f"  ours vs base paper synthetic : {emd['emd_ours_vs_theirs'].mean():.4f}",
        "",
        "Reading: if 'ours vs base paper' is of the same order as the two 'vs real' values, the two pipelines produce",
        "comparably faithful data and the Base arm can be described as a consistent replication in marginal statistics.",
        "If it is much larger, look at stats_comparison.csv to see which variables differ before claiming replication.",
    ]
    text = "\n".join(txt)
    open(os.path.join(out_dir, "replication_summary.txt"), "w").write(text)
    print(text)
    return long, emd
