"""
GAN training + generation (Gretel timeseries DGAN, PyTorch - same implementation as the base paper)
and post-processing of the generated data.
"""
import time
import numpy as np
import config as C
from .seeding import set_seed


# ----------------------------------------------------------------------------- time as attributes
def encode_start(hours, dows):
    """(start hour, start weekday) of each window -> 4 continuous DGAN attributes (sin/cos pairs)."""
    h = 2 * np.pi * np.asarray(hours, float) / 24.0
    d = 2 * np.pi * np.asarray(dows, float) / 7.0
    return np.stack([np.sin(h), np.cos(h), np.sin(d), np.cos(d)], axis=1)


def decode_start(attrs):
    """Generated attributes -> integer start hour (0-23) and weekday (0-6), plus the rounding error
    (in hours / days) so the paper can report how far generated attributes were from a valid value."""
    a = np.asarray(attrs, float)
    h = (np.arctan2(a[:, 0], a[:, 1]) % (2 * np.pi)) * 24.0 / (2 * np.pi)
    d = (np.arctan2(a[:, 2], a[:, 3]) % (2 * np.pi)) * 7.0 / (2 * np.pi)
    hi, di = np.rint(h).astype(int) % 24, np.rint(d).astype(int) % 7
    err_h = np.abs((h - np.rint(h)))
    err_d = np.abs((d - np.rint(d)))
    return hi, di, err_h, err_d


def clock_from_start(start_hour, start_dow, length):
    """Exact clock of every row of a window: hour advances by one per row, the weekday changes at
    midnight.  Returns (n, length, 4) = hour_sin, hour_cos, dow_sin, dow_cos (same order as TIME_COLS)."""
    t = np.arange(length)[None, :] + np.asarray(start_hour)[:, None]          # hours since start-of-day of the first row
    hour = t % 24
    dow = (np.asarray(start_dow)[:, None] + t // 24) % 7
    return np.stack([np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0),
                     np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0)], axis=2)


def fit_and_generate(train_windows, epochs, n_sequences, seed, attributes=None):
    """Train a DGAN on `train_windows` (n, SEQ_LEN, F) and return n_sequences synthetic sequences.
    Full-batch training, exactly like the base paper's code (batch_size = number of windows).
    If `attributes` (n, A) is given it is passed to the DGAN as per-window attributes (generated jointly
    with the sequence, as in DoppelGANger) and the call returns (synthetic_sequences, generated_attributes)."""
    from gretel_synthetics.timeseries_dgan.dgan import DGAN
    from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType

    set_seed(seed)
    n, L, F = train_windows.shape
    cfg = DGANConfig(
        max_sequence_len=L, sample_len=L,
        batch_size=n,
        apply_feature_scaling=True, apply_example_scaling=False,
        use_attribute_discriminator=attributes is not None,   # needs attributes; paper-2 arms have none
        generator_learning_rate=C.LEARNING_RATE, discriminator_learning_rate=C.LEARNING_RATE,
        epochs=int(epochs),
    )
    model = DGAN(cfg)
    if attributes is None:
        model.train_numpy(features=train_windows.astype(np.float32),
                          feature_types=[OutputType.CONTINUOUS] * F)
        _, synth = model.generate_numpy(int(n_sequences))
        return np.asarray(synth, dtype=np.float64)
    model.train_numpy(features=train_windows.astype(np.float32),
                      feature_types=[OutputType.CONTINUOUS] * F,
                      attributes=np.asarray(attributes, dtype=np.float32),
                      attribute_types=[OutputType.CONTINUOUS] * attributes.shape[1])
    attrs, synth = model.generate_numpy(int(n_sequences))
    return np.asarray(synth, dtype=np.float64), np.asarray(attrs, dtype=np.float64)


def clock_diagnostics(synth, cols):
    """How valid is the generated clock?  (hour_sin, hour_cos) should lie on the unit circle and
    advance ~1 h per row.  Reported for the paper; computed BEFORE projection."""
    out = {}
    for name, period in (("hour", 24.0), ("dow", 7.0)):
        if f"{name}_sin" not in cols:
            continue
        s = synth[:, :, cols.index(f"{name}_sin")]
        c = synth[:, :, cols.index(f"{name}_cos")]
        r2 = s ** 2 + c ** 2
        d = {"radius_sq_mean": float(r2.mean()), "radius_sq_min": float(r2.min()), "radius_sq_max": float(r2.max())}
        if name == "hour":
            ang = (np.arctan2(s, c) % (2 * np.pi)) * period / (2 * np.pi)
            step = np.diff(ang, axis=1)
            step = (step + period / 2) % period - period / 2
            d["share_steps_0.5_to_1.5h"] = float(np.mean((step > 0.5) & (step < 1.5)))
        out[name] = d
    return out


def postprocess(synth, cols):
    """Physical clipping + projection of generated (sin, cos) pairs back onto the unit circle."""
    x = synth.copy()
    for name in C.NON_NEGATIVE:
        if name in cols:
            i = cols.index(name); x[:, :, i] = np.clip(x[:, :, i], 0.0, None)
    for name, hi in C.UPPER_BOUNDS.items():
        if name in cols:
            i = cols.index(name); x[:, :, i] = np.clip(x[:, :, i], 0.0, hi)
    for name in ("hour", "dow"):
        if f"{name}_sin" in cols:
            si, ci = cols.index(f"{name}_sin"), cols.index(f"{name}_cos")
            norm = np.sqrt(x[:, :, si] ** 2 + x[:, :, ci] ** 2)
            ok = norm > 1e-6
            x[:, :, si] = np.where(ok, x[:, :, si] / np.where(ok, norm, 1.0), 0.0)
            x[:, :, ci] = np.where(ok, x[:, :, ci] / np.where(ok, norm, 1.0), 1.0)
    return x


def generate_synthetic(train_windows, cols, epochs, n_sequences, seed, attributes=None):
    """`cols` are the columns of the FINAL synthetic array.  In attribute mode `train_windows` holds the
    base columns only (cols minus TIME_COLS, which come last) and the time columns are derived from the
    generated start hour / weekday, so the clock is exact by construction."""
    t0 = time.time()
    if attributes is None:
        raw = fit_and_generate(train_windows, epochs, n_sequences, seed)
        diag = clock_diagnostics(raw, cols)
    else:
        feats, attrs = fit_and_generate(train_windows, epochs, n_sequences, seed, attributes=attributes)
        h, d, err_h, err_d = decode_start(attrs)
        raw = np.concatenate([feats, clock_from_start(h, d, feats.shape[1])], axis=2)
        diag = {"mode": "attribute", "start_hour_rounding_error_mean_h": float(err_h.mean()),
                "start_dow_rounding_error_mean_d": float(err_d.mean()),
                "start_hour_distinct": int(len(set(h.tolist()))), "start_dow_distinct": int(len(set(d.tolist()))),
                "attr_radius_sq_mean_hour": float((np.asarray(attrs)[:, :2] ** 2).sum(1).mean()),
                "attr_radius_sq_mean_dow": float((np.asarray(attrs)[:, 2:4] ** 2).sum(1).mean())}
    seconds = time.time() - t0
    return postprocess(raw, cols), diag, seconds
