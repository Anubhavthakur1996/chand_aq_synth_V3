"""Unit tests for time-as-attributes and sliding windows (no GAN needed).   python tests/test_time_attributes.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import windows as W
from src import train_dgan as T

# 1. encode -> decode round trip for every (hour, weekday)
h = np.repeat(np.arange(24), 7); d = np.tile(np.arange(7), 24)
hh, dd, eh, ed = T.decode_start(T.encode_start(h, d))
assert np.array_equal(hh, h) and np.array_equal(dd, d) and eh.max() < 1e-9 and ed.max() < 1e-9

# 2. clock: start Sunday (6) 22:00 -> 22, 23, 0, 1 ... and the weekday rolls to Monday (0) at midnight
c = T.clock_from_start(np.array([22]), np.array([6]), 24)[0]
hour = (np.arctan2(c[:, 0], c[:, 1]) % (2 * np.pi)) * 24 / (2 * np.pi)
dow = (np.arctan2(c[:, 2], c[:, 3]) % (2 * np.pi)) * 7 / (2 * np.pi)
assert np.allclose(hour[:4], [22, 23, 0, 1]) and np.allclose(dow[:4], [6, 6, 0, 0]) and np.allclose(dow[-1], 0)
assert np.allclose(c[:, 0] ** 2 + c[:, 1] ** 2, 1) and np.allclose(c[:, 2] ** 2 + c[:, 3] ** 2, 1)   # exactly on the unit circle

# 3. window start rows are correct for stride 24 and stride 1, across two segments
vals = np.arange(60, dtype=float)[:, None] * np.ones((1, 2))
seg = np.r_[np.zeros(40, int), np.ones(20, int)]
for stride in (24, 1):
    win, st = W.make_windows(vals, seg, 24, stride, return_starts=True)
    assert len(win) == len(st) and all(win[i, 0, 0] == st[i] for i in range(len(st)))   # value = row index
    assert all((st[i] < 40 and st[i] + 23 < 40) or st[i] >= 40 for i in range(len(st)))   # no window crosses the gap
w1 = W.make_windows(vals, seg, 24, 1)
assert w1.shape[0] == (40 - 24 + 1)                         # the 20-row segment is too short; 17 windows from the first
print("time-attribute tests passed")
