"""Tests of the evaluation formulas on data where the right answer is known.   python tests/test_metrics.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from src import evaluate as E

C.RF_TREES = 30
rng = np.random.default_rng(0)
cols = ["a", "b", "c"]
real = rng.normal(5, 2, size=(40, 24, 3))                  # 40 "days" x 24 hours x 3 variables

# 1) identical data -> every distance is zero
pf = E.per_feature_fidelity(real, real.copy(), cols)
assert pf["emd_norm"].max() < 1e-9 and pf["js_distance"].max() < 1e-6 and pf["delta_acf"].max() < 1e-9
assert E.corr_frobenius(real, real.copy(), [0, 1, 2]) < 1e-9

# 2) shifting every variable by one real std -> normalised EMD is exactly 1
shifted = real + real.reshape(-1, 3).std(0)
assert np.allclose(E.per_feature_fidelity(real, shifted, cols)["emd_norm"], 1.0, atol=1e-6)

# 3) smooth daily curve vs white noise -> clearly different autocorrelation
t = np.arange(24)
smooth = np.sin(2 * np.pi * t / 24)[None, :, None] + 0.05 * rng.normal(size=(40, 24, 3))
noise = rng.normal(size=(40, 24, 3))
assert E.per_feature_fidelity(smooth, noise, cols)["delta_acf"].min() > 0.3

# 4) correlation structure: correlated pair vs independent pair
base = rng.normal(size=(200, 24, 1))
corr = np.concatenate([base, base + 0.1 * rng.normal(size=base.shape)], axis=2)
indep = rng.normal(size=(200, 24, 2))
assert E.corr_frobenius(corr, corr, [0, 1]) < 1e-9 and E.corr_frobenius(corr, indep, [0, 1]) > 1.0

# 5) memorisation check: copies of training days -> ratio ~0; fresh samples -> ratio ~1
train = rng.normal(size=(13, 24, 3)); held = rng.normal(size=(12, 24, 3))
copies = train[rng.integers(0, 13, 50)]
nn = E.nn_memorisation(train, held, copies, cols, cols, np.random.default_rng(1))
assert nn["synth_to_train"] < 1e-9 and nn["ratio"] < 1e-6
fresh = rng.normal(size=(50, 24, 3))
nn2 = E.nn_memorisation(train, held, fresh, cols, cols, np.random.default_rng(1))
assert 0.5 < nn2["ratio"] < 2.0, nn2

# 6) TSTR / TRTR / URR / shuffle control on a process where we know the truth (AR(1) series)
def ar1(n, rho, g, burn=50):
    x = np.zeros(n + burn)
    for i in range(1, n + burn):
        x[i] = rho * x[i - 1] + g.normal()
    return x[burn:]
g = np.random.default_rng(3)
real_series = np.stack([ar1(2000, 0.9, g), ar1(2000, 0.5, g)], axis=1)      # columns: pm25, x
seg = np.zeros(2000, int)
synth = np.stack([np.stack([ar1(24, 0.9, g), ar1(24, 0.5, g)], axis=1) for _ in range(400)])
res = E.utility_block(real_series, seg, 600, 1000, ["pm25", "x"], ["pm25", "x"], synth,
                      np.random.default_rng(0), [400], 400, shuffle_volume=400)
urr = res["urr"]["400"]["rf"]
tstr_mae, shuf_mae = res["tstr"]["400"]["rf"]["mae"], res["shuffle_control"]["rf"]["mae"]
assert 0.7 < urr < 1.4, urr                                  # synthetic data from the same process -> TSTR ~ TRTR
assert shuf_mae > 1.5 * tstr_mae, (shuf_mae, tstr_mae)       # scrambling the order must hurt clearly
print(f"metric tests passed  (URR on same-process data = {urr:.2f}, shuffle MAE {shuf_mae:.2f} vs TSTR MAE {tstr_mae:.2f})")