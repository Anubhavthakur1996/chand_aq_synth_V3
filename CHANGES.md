# What changed, and why

Maps every code-related item of the action plan (B 9-19 and C 20-26) to what was done.
**Status legend:** TESTED = checked on your real data with a stand-in generator (`tests/`);
NOT TESTED = needs the real GAN library / your GPU, which was not available when this was written.

## Code fixes (action items 9-19)

| # | Item | What was done | Status |
|---|---|---|---|
| 9 | Golden run mixed into the "5-seed" mean | Removed the golden/robustness split. Every arm uses the same 5 seeds and the same epoch budget for all of its runs. | TESTED |
| 10 | Predictors differed by tier | Primary TSTR uses the **same predictor columns (Base columns) for every arm**; time columns are generated and evaluated but never given to the predictor. An all-columns evaluation is reported separately as "secondary" for time arms. | TESTED |
| 11 | TRTR / URR not computed in the main pipeline | TRTR and URR = TSTR/TRTR MAE are computed for every run, seed, model and synthetic volume. | TESTED |
| 12 | Generated clock not on the unit circle | The raw clock validity (mean sin^2+cos^2, share of ~1 h steps) is recorded per run, then (sin, cos) pairs are projected back onto the unit circle. Deterministic/known time (not generated) was **not** implemented; it is a possible future variant. | TESTED |
| 13 | NOx column | **Resolved differently from my first guess.** The source labels NO, NO2 and NOx all as ppb, but Chandigarh NOx averages 0.023 against NO + NO2 = 36.5 (NOx can never be smaller than NO + NO2), and it is a near-perfect rescaled copy of them (Pearson r = 0.993, ratio 0.00059, IQR 0.00056-0.00064, never >= NO + NO2). So it is not an independent or correctly-scaled measurement. Default `NOX_HANDLING = "drop"` (NO and NO2 carry the same information); `"derived"` (NOx = NO + NO2) is available. Diagnostics are written to `data_report.json`. | TESTED |
| 14 | Only 336 synthetic rows | 3,650 synthetic days (10 years, as in the base paper); TSTR also reported at 14 and 365 days. | TESTED |
| 15 | Windows crossing gaps / stride | Hourly grid + `segment` ids; training windows, lag samples and test samples never cross a gap. Training windows are non-overlapping 24-h days (stride 24, as in the base paper; stride 1 is a config switch). Chandigarh gives 13 training windows, Bilbao 13. Unit-tested for leakage. | TESTED |
| 16 | Undertrained LSTM probe | Removed from the pipeline (RF and gradient boosting remain). It was trained for 10 epochs on unscaled inputs, so it said nothing about deep learning. The old code is in `archive/`. | - |
| 17 | Fidelity metrics | EMD divided by real std; arms compared on **shared columns**; ACF computed **inside** each day (no seams); fidelity reported against training days and against a held-out validation block; nearest-neighbour memorisation ratio added. JS is now the base-2 JS *distance* (old numbers are not comparable). | TESTED |
| 18 | Day-block shuffle control | **Not added on purpose.** Because lag windows now stay inside a day, shuffling whole days changes nothing. The row-shuffle control is kept. | - |
| 19 | Housekeeping | `requirements.txt` added; old code and the old run moved to `archive/` (nothing deleted); README rewritten (it said Hour+Month and used words like "prove"); `repro` command checks that one seed gives identical output twice. | `repro`: NOT TESTED |

## Experiment design (items 20-26)

* 4 arms exactly as in the plan; short budget chosen by `main.py tune` on **validation** data, **Base arm only**,
  criterion = mean normalised EMD to held-out real days (seeds 11, 22, 33 are used for tuning, never for the final runs).
  The sweep over both Base and Base+Time arms also gives the metric-vs-epochs figure (`epochs_vs_fidelity.png`).
* Short budgets are separate runs (no checkpoint extraction). Bilbao's `a4` uses the budget chosen on Chandigarh.
* Bilbao: `a1` and `a4` only, vehicle columns and the 8-hour CO/O3 columns excluded.
* All runs are resumable (`results/<dataset>/runs/*.json`), and every run's synthetic data is saved (`.npz`).

## Replication check (action item 8) - added
`python main.py replicate --base-synth <path>` compares the summary statistics of our Bilbao `a1_base_long` synthetic data
(over all seeds) with the synthetic data released by the base authors, and reports (a) whether their values fall inside our
seed spread, (b) normalised EMD of each synthetic set to the real data and to each other. Outputs are in
`results/bilbao/replication_check/`. TESTED with the stand-in generator only (pipeline works; the numbers mean nothing until
real GAN runs exist). The base file (31 MB) is not bundled; copy it from the base repo. The optional forecasting
replication (item 9) was deliberately skipped.

## Decisions made for you (please sanity-check; all are one-line changes in `config.py`)

1. **NOx** is dropped for Chandigarh (see item 13), so Chandigarh's Base set is the base paper's pollutants + weather without NOx; Bilbao keeps its valid NOx.
2. **Local time** (Asia/Kolkata) is used for the hour/day-of-week features; raw timestamps are UTC.
3. **Lookback = 12 h** (was 24). Needed so lag windows fit inside one synthetic day (days are 24 h long).
4. **Validation block**: rows 336-671 are now validation, so the test set starts at row 672 (10,547 rows).
5. **Interpolation** is time-aware (done on the hourly grid, gaps up to 3 h), which is stricter than the old row-based version.
6. Training windows: 13 non-overlapping days, so the GAN sees far fewer samples than the old 313 overlapping windows.

## New data facts for the paper (from `main.py prep`; always re-read `data_report.json`)

Chandigarh (local time): 2025-02-19 02:00 to 2026-08-01 04:00; 11,442 raw rows; 12,676 hourly slots;
6,526 values interpolated; 1,457 hours dropped; **11,219 rows in 147 contiguous segments** (longest gap 240 h);
training block = 336 rows spanning 366 h with 1 gap.  The old text ("Jan 2024 - Dec 2025", 11,052 points,
"single gaps only") should be replaced by these.

## Not verified

* Everything involving the **real Gretel DGAN** (training, speed, memory, determinism): the library was not
  available here. The configuration is copied from your earlier code (plus `use_attribute_discriminator=False`).
  Run `python main.py repro` first, then a very short `python main.py run --arms a1_base_long --seeds 42` with
  `EPOCHS_LONG = 200` in `config.py` to confirm the GAN path works before the long runs.
* Compute: the 100,000-epoch runs dominate (Chandigarh 10 long runs + Bilbao 5). Reduce seeds to 3 in `config.py` if needed.
