"""
Central configuration for every experiment.  Nothing in src/ hard-codes a number
that matters for results; change it here (or via main.py command-line flags).
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ----------------------------------------------------------------------------
# Experimental design (see README.md)
# ----------------------------------------------------------------------------
# Four arms: {Base, Base + cyclical time} x {long (base-paper) budget, short (tuned) budget}
ARMS = {
    "a1_base_long":  {"features": "base",      "epochs": "long"},
    "a2_time_long":  {"features": "base_time", "epochs": "long"},
    "a3_base_short": {"features": "base",      "epochs": "short"},
    "a4_time_short": {"features": "base_time", "epochs": "short"},
}
# Paper-3 arms (run with `python main.py run --paper3`); they do not change the four paper-2 arms above.
#   time_mode "attribute": the GAN is trained on the Base columns only, with the window's START hour and
#                          weekday as DoppelGANger attributes; the 4 time columns are then derived exactly.
#   stride: spacing of training windows in hours (24 = non-overlapping days as in paper 2; 1 = sliding).
ARMS.update({
    "n1_cond_short":      {"features": "base_time", "epochs": "short", "time_mode": "attribute", "stride": 24},
    "n2_slide_short":     {"features": "base",      "epochs": "short", "time_mode": "generated", "stride": 1},
    "n3_condslide_short": {"features": "base_time", "epochs": "short", "time_mode": "attribute", "stride": 1},
})
PAPER3_ARMS = ["n1_cond_short", "n2_slide_short", "n3_condslide_short"]
EPOCHS_LONG = 100_000             # budget used by the base paper (Morales-Garcia et al. 2025, Table 3)
# EPOCHS_LONG = 200                   # For test run
EPOCHS_SHORT_DEFAULT = 5_000        # used only if `main.py tune` has not been run yet
FINAL_SEEDS = [42, 123, 789]        # 3 seeds, same for every arm
TUNE_SEEDS = [11, 22, 33]                # different from FINAL_SEEDS on purpose
TUNE_EPOCHS = [1000, 2000, 3000, 5000, 7500, 10000]
TUNE_ARMS = ["a3_base_short", "a4_time_short"]   # both are swept (for the figure) ...
TUNE_SELECT_ARM = "a3_base_short"                # ... but the choice is made on Base only
TUNE_N_SYNTH = 365                  # synthetic sequences generated per tuning run

# ----------------------------------------------------------------------------
# GAN (Gretel DGAN) settings - identical to the base paper's released code
# ----------------------------------------------------------------------------
SEQ_LEN = 24                        # one-day windows (sample_len = 24 in the base paper)
TRAIN_STRIDE = 24                   # 24 = non-overlapping days (base paper); 1 = fully overlapping
LEARNING_RATE = 1e-4
N_SYNTH_SEQUENCES = 3650            # 3650 x 24 h = 10 years, as in the base paper
SYNTH_VOLUME_CURVE = [14, 365, 3650]   # TSTR is reported at each synthetic volume
PRIMARY_VOLUME = 3650
SHUFFLE_VOLUME = 365                # volume for the shuffled control; keep it in SYNTH_VOLUME_CURVE

# ----------------------------------------------------------------------------
# Evaluation settings
# ----------------------------------------------------------------------------
TARGET = "pm25"
LOOKBACK = 12                       # hours of history given to the downstream predictor.
                                    # (< SEQ_LEN so that lag windows fit INSIDE a synthetic day
                                    #  and never straddle the seam between two independent days)
ACF_MAX_LAG = 12
JS_BINS = 50
NN_SAMPLE = 500                     # synthetic windows used for the nearest-neighbour check
RF_TREES = 100
RF_N_JOBS = 3                       # CPU threads for the random forest; -1 = all
MODEL_SEED = 0                      # seed of RF/GBM themselves (kept constant: GAN seed is the variable)

# Physical limits applied to generated data (post-processing, reported in the paper)
NON_NEGATIVE = ["pm25", "pm10", "o3", "co", "so2", "no", "no2", "nox",
                "benzene", "ethylbenzene", "orthoxylene", "toluene", "ws", "precip"]
UPPER_BOUNDS = {"rh": 100.0}

# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------
INTERP_LIMIT = 3                 # fill gaps of at most this many consecutive hours; drop the rest
# How to treat the Chandigarh NOx column.  The OpenAQ file labels NO, NO2 and NOx all as ppb, but the
# NOx values (mean 0.023) are ~1500x smaller than NO + NO2 (mean 36.5), which is physically impossible
# (NOx = NO + NO2), while being almost perfectly correlated with NO + NO2 (r = 0.993).  The column is
# therefore a rescaled copy of NO + NO2 with an unreliable unit/calibration, not an independent measurement.
#   "drop"    : do not use it (default; NO and NO2 are kept, they carry the same information)
#   "derived" : replace it by NOx = NO + NO2 (exact by definition, correct units, but redundant)
#   "raw"     : keep the raw column x1000 (NOT recommended: still only ~0.6 x (NO + NO2))
NOX_HANDLING = "drop"

TIME_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

DATASETS = {
    "chandigarh": {
        "raw_path": os.path.join(BASE_DIR, "data", "raw", "Chandigarh_Sector22_AirQuality_2025_2026.csv"),
        "tz": "Asia/Kolkata",                         # time features use LOCAL time
        "base_cols": ["pm25", "pm10", "o3", "co", "so2", "temp", "rh", "ws", "wd", "no", "no2", "nox"],  # see NOX_HANDLING
        "train_rows": 336, "val_rows": 336,           # test = every row after train+val
        "arms": ["a1_base_long", "a2_time_long", "a3_base_short", "a4_time_short"],   # paper-2 arms
    },
    "bilbao": {   # base paper's data, vehicle/8h columns removed -> like-for-like with Chandigarh
        "raw_path": os.path.join(BASE_DIR, "data", "raw", "bilbao", "60_BILBAO.csv"),
        "tz": None,
        "base_cols": ["pm25", "pm10", "o3", "co", "so2", "temp", "rh", "ws", "wd", "no", "no2", "nox",
                      "benzene", "ethylbenzene", "orthoxylene", "toluene", "precip"],
        "train_rows": 326, "val_rows": 0,             # base-paper protocol: test = last 24 h
        "arms": ["a1_base_long", "a4_time_short"],    # lean plan: Tier 1 and Tier 4 only
    },
}

if NOX_HANDLING == "drop":
    DATASETS["chandigarh"]["base_cols"] = [c for c in DATASETS["chandigarh"]["base_cols"] if c != "nox"]
