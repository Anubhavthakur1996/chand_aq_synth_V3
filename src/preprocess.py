"""
Data preparation.

For every dataset this produces ONE standard table:
    index   = timestamp on a regular hourly grid (local time, tz-naive)
    columns = the dataset's base variables + 4 cyclical time features + `segment`
`segment` labels runs of consecutive hours.  Every window / lag sample built later is
taken INSIDE one segment, so nothing ever straddles a data gap.

Cleaning rules (all reported in <results>/<dataset>/data_report.json):
  1. raw timestamps -> local time, floored to the hour, duplicates averaged
  2. reindexed to a full hourly grid (missing hours become explicit NaN rows)
  3. gaps of <= INTERP_LIMIT consecutive hours are linearly interpolated
  4. any hour still missing is dropped (creating a segment break)
"""
import json, os
import numpy as np
import pandas as pd
import config as C

MOLAR_VOLUME = 24.45      # L/mol at 25 C, 1 atm (CPCB reference conditions)
MOLAR_MASS = {"co": 28.01, "no": 30.01, "no2": 46.01, "so2": 64.07}

# raw column prefix -> (canonical name, kind)
CHANDIGARH_MAP = {
    "pm25_": ("pm25", None), "pm10_": ("pm10", None), "o3_": ("o3", None),
    "co_": ("co", "ppb:co"), "so2_": ("so2", "ppb:so2"),
    "no2_": ("no2", "ppb:no2"), "no_": ("no", "ppb:no"), "nox_": ("nox", "nox"),
    "temperature_": ("temp", None), "relativehumidity_": ("rh", None),
    "wind_speed_": ("ws", None), "wind_direction_": ("wd", None),
}
BILBAO_MAP = {
    "Benceno": "benzene", "CO": "co", "Etilbenceno": "ethylbenzene", "NO": "no", "NO2": "no2",
    "NOX": "nox", "O3": "o3", "Ortoxileno": "orthoxylene", "PM10": "pm10", "PM2.5": "pm25",
    "SO2": "so2", "Tolueno": "toluene", "DirecWind": "wd", "Humid": "rh", "Precip": "precip",
    "Temp": "temp", "SpeedWind": "ws",
}   # Car/Motorcycle/Bus/Truck, 'CO 8h', 'O3 8h' are intentionally NOT used (see README)


def _ppb_to_ugm3(x, species):
    return x * MOLAR_MASS[species] / MOLAR_VOLUME


def _load_chandigarh(cfg):
    raw = pd.read_csv(cfg["raw_path"])
    ts = pd.to_datetime(raw["date.utc"], utc=True).dt.tz_convert(cfg["tz"]).dt.tz_localize(None)
    out = pd.DataFrame(index=ts)
    for col in raw.columns:
        if col == "date.utc":
            continue
        for prefix, (name, kind) in CHANDIGARH_MAP.items():
            if col.startswith(prefix):
                v = pd.to_numeric(raw[col], errors="coerce").to_numpy()
                if kind and kind.startswith("ppb:"):
                    v = _ppb_to_ugm3(v, kind.split(":")[1])
                elif kind == "nox":
                    if C.NOX_HANDLING == "raw":
                        v = _ppb_to_ugm3(v * 1000.0, "no2")     # NOT recommended, see config.py
                    else:
                        continue                                  # handled below ("drop" / "derived")
                out[name] = v
                break
        else:
            print(f"[prep] note: raw column '{col}' is not used")
    if C.NOX_HANDLING == "derived":
        # NOx = NO + NO2 (both in ppb), reported as NO2-equivalent mass
        out["nox"] = _ppb_to_ugm3(raw["no_ppb"].to_numpy(float) + raw["no2_ppb"].to_numpy(float), "no2")
    diag = _nox_raw_diagnostics(raw)
    missing = [c for c in cfg["base_cols"] if c not in out.columns]
    if missing:
        raise ValueError(f"raw file lacks expected columns: {missing}")
    return out[cfg["base_cols"]], raw.shape[0], diag


def _nox_raw_diagnostics(raw):
    """Facts about the raw NOx column, written to data_report.json so the decision is documented."""
    cols = ["no_ppb", "no2_ppb", "nox_ppb"]
    if not all(c in raw.columns for c in cols):
        return None
    d = raw[cols].apply(pd.to_numeric, errors="coerce").dropna()
    s = d["no_ppb"] + d["no2_ppb"]
    ratio = (d["nox_ppb"] / s.replace(0, np.nan)).dropna()
    return {
        "labelled_unit_in_source": "ppb (as are NO and NO2)",
        "rows_used": int(len(d)),
        "mean_nox": round(float(d["nox_ppb"].mean()), 4), "mean_no_plus_no2": round(float(s.mean()), 3),
        "share_rows_nox_ge_no_plus_no2": round(float((d["nox_ppb"] >= s).mean()), 4),
        "pearson_r_nox_vs_no_plus_no2": round(float(d["nox_ppb"].corr(s)), 4),
        "median_ratio_nox_over_no_plus_no2": round(float(ratio.median()), 6),
        "iqr_ratio": [round(float(ratio.quantile(.25)), 6), round(float(ratio.quantile(.75)), 6)],
        "conclusion": "NOx is a near-perfect rescaled copy of NO + NO2 with an inconsistent unit; "
                      f"handled as NOX_HANDLING='{C.NOX_HANDLING}'",
    }


def _load_bilbao(cfg):
    raw = pd.read_csv(cfg["raw_path"])
    ts = pd.to_datetime(raw["Fecha"].astype(str) + " " + raw["Hora"].astype(str))
    out = pd.DataFrame(index=ts)
    for src, name in BILBAO_MAP.items():
        out[name] = pd.to_numeric(raw[src], errors="coerce").to_numpy()
    return out[cfg["base_cols"]], raw.shape[0], None


def add_time_features(df):
    hour = df.index.hour.to_numpy()
    dow = df.index.dayofweek.to_numpy()
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return df


def nox_plausibility(df):
    """NOx (as NO2-equivalent mass) should be ~ NO2 + 1.53*NO. Report the ratio for the column actually used."""
    if not {"nox", "no", "no2"} <= set(df.columns):
        return None
    expected = (df["no2"] + 1.53 * df["no"]).median()
    ratio = float(df["nox"].median() / expected) if expected > 0 else float("nan")
    ok = 0.9 <= ratio <= 1.5      # NOx = NO + NO2, so the ratio should be ~1
    return {"median_nox_over_expected": round(ratio, 3), "plausible": bool(ok)}


def prepare_dataset(dataset, save=True):
    cfg = C.DATASETS[dataset]
    loader = _load_chandigarh if dataset == "chandigarh" else _load_bilbao
    df, n_raw, nox_diag = loader(cfg)

    df.index = df.index.floor("h")
    n_dup = int(df.index.duplicated().sum())
    df = df.groupby(level=0).mean().sort_index()
    grid = pd.date_range(df.index.min(), df.index.max(), freq=pd.Timedelta(hours=1))
    df = df.reindex(grid)
    n_grid = len(df)
    missing_before = df.isna().sum()
    hours_any_nan_before = int(df.isna().any(axis=1).sum())

    df = df.interpolate(method="linear", limit=C.INTERP_LIMIT, limit_area="inside")
    filled_cells = int((missing_before - df.isna().sum()).sum())
    keep = ~df.isna().any(axis=1)
    n_dropped = int((~keep).sum())
    df = df[keep].copy()

    df = add_time_features(df)
    gap = df.index.to_series().diff() > pd.Timedelta(hours=1)
    df["segment"] = gap.cumsum().to_numpy()
    seg_len = df.groupby("segment").size()
    gap_hours = (df.index.to_series().diff().dt.total_seconds() / 3600.0)[gap]

    cfg_t = cfg["train_rows"]; cfg_v = cfg["val_rows"]
    first = df.iloc[:cfg_t]
    report = {
        "dataset": dataset,
        "raw_rows": int(n_raw), "duplicate_timestamps_averaged": n_dup,
        "date_range": [str(df.index.min()), str(df.index.max())],
        "grid_hours": int(n_grid), "hours_with_any_missing_value_before_interpolation": hours_any_nan_before,
        "interpolation": f"linear, gaps <= {C.INTERP_LIMIT} consecutive hours, interior only",
        "cells_interpolated": filled_cells, "hours_dropped_after_interpolation": n_dropped,
        "final_rows": int(len(df)), "segments": int(seg_len.shape[0]),
        "longest_segment_hours": int(seg_len.max()), "longest_gap_hours": float(gap_hours.max()) if len(gap_hours) else 0.0,
        "gaps_longer_than_1h": int(gap.sum()),
        "train_rows": cfg_t, "val_rows": cfg_v, "test_rows": int(len(df) - cfg_t - cfg_v),
        "train_block_span_hours": float((first.index.max() - first.index.min()).total_seconds() / 3600 + 1),
        "train_block_gaps": int((first["segment"].nunique() - 1)),
        "raw_missing_share_percent": {k: round(float(v) / n_grid * 100, 2) for k, v in missing_before.items()},
        "max_values": {k: float(df[k].max()) for k in cfg["base_cols"]},
        "outlier_policy": "none (values kept as recorded; extreme maxima listed above)",
        "local_timezone_for_time_features": cfg["tz"] or "as recorded",
        "nox_handling": C.NOX_HANDLING if dataset == "chandigarh" else "n/a (valid NOx in the source)",
        "nox_raw_column_diagnostics": nox_diag,
        "nox_plausibility_of_used_column": nox_plausibility(df),
    }
    if save:
        out_dir = os.path.join(C.RESULTS_DIR, dataset)
        os.makedirs(out_dir, exist_ok=True)
        proc_dir = os.path.join(C.BASE_DIR, "data", "processed")
        os.makedirs(proc_dir, exist_ok=True)
        df.to_csv(os.path.join(proc_dir, f"{dataset}_processed.csv"), index_label="timestamp")
        desc = df[cfg["base_cols"]].describe().T.round(3)
        desc.to_csv(os.path.join(out_dir, "data_description.csv"))
        with open(os.path.join(out_dir, "data_report.json"), "w") as f:
            json.dump(report, f, indent=2)
    return df, report


def load_processed(dataset):
    """Load the processed table (re-creating it if needed)."""
    path = os.path.join(C.BASE_DIR, "data", "processed", f"{dataset}_processed.csv")
    if not os.path.exists(path):
        return prepare_dataset(dataset)[0]
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    return df
