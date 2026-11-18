"""Window / lag-sample builders.  Everything is taken INSIDE contiguous segments."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def segment_ranges(seg):
    """[(start, end), ...] of runs of equal segment id (end exclusive)."""
    seg = np.asarray(seg)
    if len(seg) == 0:
        return []
    cuts = np.flatnonzero(np.diff(seg) != 0) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(seg)]
    return list(zip(starts.tolist(), ends.tolist()))


def make_windows(values, seg, length, stride, return_starts=False):
    """Windows of `length` rows that lie fully inside one segment -> (n, length, F).
    With return_starts=True also returns the row index (into `values`) where each window starts."""
    out, starts = [], []
    for s, e in segment_ranges(seg):
        n = e - s
        if n < length:
            continue
        w = sliding_window_view(values[s:e], (length, values.shape[1]))[:, 0]   # (n-length+1, length, F)
        out.append(w[::stride])
        starts.append(s + np.arange(0, n - length + 1, stride))
    if not out:
        empty = np.empty((0, length, values.shape[1]))
        return (empty, np.empty((0,), dtype=int)) if return_starts else empty
    win = np.concatenate(out, axis=0).copy()
    return (win, np.concatenate(starts)) if return_starts else win


def lag_samples_flat(values, seg, target_idx, lookback, min_target_index=0):
    """(X, y) from real data: X = previous `lookback` rows (flattened), y = target at the next row.
    Only samples whose target row index >= min_target_index are kept (used to build the test set
    while allowing the history to come from the rows just before the test block)."""
    Xs, ys = [], []
    for s, e in segment_ranges(seg):
        n = e - s
        if n <= lookback:
            continue
        arr = values[s:e]
        win = sliding_window_view(arr, (lookback, arr.shape[1]))[:, 0][: n - lookback]
        tgt = arr[lookback:, target_idx]
        pos = np.arange(s + lookback, e)
        keep = pos >= min_target_index
        if keep.any():
            Xs.append(win[keep].reshape(int(keep.sum()), -1))
            ys.append(tgt[keep])
    if not Xs:
        return np.empty((0, lookback * values.shape[1])), np.empty((0,))
    return np.concatenate(Xs), np.concatenate(ys)


def lag_samples_sequences(seqs, target_idx, lookback):
    """(X, y) from synthetic sequences (N, L, F): windows never leave a sequence."""
    N, L, F = seqs.shape
    n_pos = L - lookback
    if n_pos <= 0:
        raise ValueError("LOOKBACK must be smaller than the sequence length")
    X = np.stack([seqs[:, i:i + lookback, :].reshape(N, -1) for i in range(n_pos)], axis=1)
    y = np.stack([seqs[:, i + lookback, target_idx] for i in range(n_pos)], axis=1)
    return X.reshape(N * n_pos, -1), y.reshape(-1)
