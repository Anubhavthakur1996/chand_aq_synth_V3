"""Unit tests for the window builders (no GAN needed).   python tests/test_windows.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import windows as W

# two segments: rows 0-29 (30 rows) and rows 30-39 (10 rows); value = row index, so leaks are visible
vals = np.arange(40, dtype=float)[:, None] * np.ones((1, 2))
seg = np.r_[np.zeros(30, int), np.ones(10, int)]

w = W.make_windows(vals, seg, 24, 24)
assert w.shape == (1, 24, 2), w.shape                      # only the 30-row segment fits one 24h window
w1 = W.make_windows(vals, seg, 24, 1)
assert w1.shape[0] == 7                                    # 30-24+1 windows, none from the 10-row segment
for win in w1:                                             # every window is consecutive rows of ONE segment
    assert np.all(np.diff(win[:, 0]) == 1) and win[-1, 0] < 30

X, y = W.lag_samples_flat(vals, seg, target_idx=0, lookback=12)
assert len(y) == (30 - 12) + 0                             # 10-row segment too short -> contributes nothing
for xi, yi in zip(X, y):                                   # target is the row right after the 12 history rows
    hist = xi.reshape(12, 2)[:, 0]
    assert yi == hist[-1] + 1 and hist[0] == yi - 12

X2, y2 = W.lag_samples_flat(vals, seg, 0, 12, min_target_index=25)
assert y2.min() >= 25                                      # test targets only from the test block
assert X2[0].reshape(12, 2)[0, 0] < 25                     # ... but history may come from before it

seqs = np.arange(2 * 24 * 3, dtype=float).reshape(2, 24, 3)
Xs, ys = W.lag_samples_sequences(seqs, 0, 12)
assert Xs.shape == (2 * 12, 12 * 3) and len(ys) == 24      # windows stay inside their own day
assert np.all(ys[:12] < 24 * 3) and np.all(ys[12:] >= 24 * 3)   # first 12 targets belong to day 0, rest to day 1
print("window tests passed")
