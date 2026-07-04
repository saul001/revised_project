# NEPSE Manufacturing Stock Price Prediction — LSTM vs GRU

A final-year-project-ready, end-to-end pipeline for predicting the stock
prices of **5 Nepali manufacturing companies** listed on NEPSE, using
**LSTM** and **GRU** deep-learning models, plus an interactive **Streamlit**
dashboard.

Built on top of the workflow in the reference notebook (`Untitled32.ipynb`),
extended with:

- ✅ A directional **confusion matrix** (Up / Down classification) for each model
- ✅ **1-week-ahead (configurable) forecasting** on new/latest data
- ✅ A full **Streamlit web app** with performance dashboards, confusion
  matrices, actual-vs-predicted charts, and live forecasting
- ✅ A clean, reusable, executable project folder (not just a notebook)

> ⚠️ **Disclaimer**: This is an academic project. Predictions are for
> learning purposes only and must not be used as financial advice.


## What changed in this version (vs. the original)

This is the corrected pipeline, matching `nepse_lstm_gru_fixed.ipynb`:

1. **Naive baseline added.** Every model is benchmarked against "predict
   yesterday's close". Without this, `100 − MAPE` on price levels is a
   vanity metric (a naive model scores ~98-99%).
2. **Lookahead leakage removed.** `clean_series()` now uses `ffill()` only
   — never `bfill()`, which copies *future* values into the past. Leading
   rows that are still NaN after forward-filling are dropped instead.
3. **No more price winsorization.** The old pipeline computed IQR bounds on
   the full close-price series and clipped open/high/low with them — this
   leaked test-period statistics into training data, could produce
   impossible bars (`high < low`), and flattened legitimate new highs/lows
   (exactly the regimes worth predicting). Anomalies are now *reported*
   (`non_positive_prices`, `ohlc_inconsistent`, `moves_gt_20pct`), never
   silently rewritten.
4. **Lag-echo diagnostic added.** Checks whether a model is just echoing
   yesterday's price shifted by one day — the classic failure mode for
   price-level LSTMs — and flags it as `Echoes_Yesterday` in the metrics.
5. **Keras 3 compatible.** Models now use an explicit `Input(shape=...)`
   layer instead of the deprecated `input_shape=` kwarg.
6. **Proper reproducibility.** `tf.keras.utils.set_random_seed()` +
   `enable_op_determinism()`, not just `np.random.seed` / `tf.random.set_seed`.
7. **Training windows shuffled.** Shuffling *windows* is safe (each window
   is self-contained) and improves convergence; `validation_split` still
   takes the chronologically-last slice before shuffling, so the
   train/validation/test ordering is preserved where it matters.
8. **The app surfaces all of this.** The Model Performance tab now shows
   the naive baseline RMSE next to each model, flags whether a model beats
   it, and warns if a lag-echo pattern is detected.


---

## 1. Project structure

```
revised project
├── app.py                     # Streamlit dashboard (run this!)
├── generate_sample_data.py    # Creates a synthetic demo dataset
├── requirements.txt
├── README.md
├── .streamlit/config.toml     # Streamlit theme
├── data/
│   └── nepse_manufacturing.csv   # <- REPLACE with your real data
├── models/                    # Trained .keras models + scalers (created by train.py)
├── outputs/
│   ├── metrics/                  # metrics_summary.csv, confusion_reports.json
│   ├── plots/                    # loss curves, actual-vs-pred, confusion matrix PNGs
│   └── forecasts/                # CLI-generated forecast CSVs
└── src/
    ├── config.py               # ALL settings live here (companies, window size, epochs...)
    ├── data_utils.py            # cleaning, technical indicators, sequence building
    ├── model_utils.py           # LSTM/GRU architectures, metrics, confusion matrix
    ├── train.py                  # trains + evaluates + saves everything
    └── forecast.py               # recursive N-day-ahead forecasting
```

---

## 2. Setup

```bash
cd nepal_stock_lstm_gru
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.9–3.11 (TensorFlow compatibility).

---

## 3. Plug in your real data

The project ships with a **synthetic** sample dataset so everything runs out
of the box. To generate it (or regenerate it) run:

```bash
python generate_sample_data.py
```

**For your actual project**, replace `data/nepse_manufacturing.csv` with your
real historical data (e.g. exported from NEPSE / Merolagani / ShareSansar /
your own scraper) using exactly these columns:

| column | description                         |
|--------|--------------------------------------|
| symbol | company ticker, e.g. `BPCL`          |
| date   | trading date (`YYYY-MM-DD`)          |
| open   | opening price                        |
| high   | day high                             |
| low    | day low                              |
| close  | closing price (this is what we predict) |
| volume | traded volume                        |

One row per company per trading day. If your company symbols differ from the
defaults, edit `COMPANIES` and `COMPANY_FULL_NAMES` in `src/config.py`.

You need **at least ~150–200 trading days per company** for the default
60-day look-back window to produce a usable train/test split — more (a few
years) is strongly recommended for real results.

---

## 4. Train the models

```bash
python -m src.train
```

This will, for **each company × {LSTM, GRU}**:

1. Clean the data (duplicates, missing values, outlier winsorization)
2. Engineer features: SMA-10/20, EMA-10/20, RSI-14, MACD (+signal),
   Bollinger Bands, daily return, 10-day volatility
3. Build 60-day look-back sequences, MinMax-scale, chronological 80/20 split
4. Train with early stopping (`patience=10`, up to 100 epochs)
5. Evaluate on the held-out test set: **RMSE, MAE, MAPE, Accuracy (%) =
   100−MAPE, R², Directional Accuracy**
6. Build a **directional confusion matrix** (Up vs Down day-over-day)
7. Save models (`models/*.keras`), scalers (`models/*_scaler.pkl`),
   metrics (`outputs/metrics/metrics_summary.csv`), confusion-matrix data
   (`outputs/metrics/confusion_reports.json`), and plots (`outputs/plots/`)

Useful flags:

```bash
python -m src.train --csv path/to/your_data.csv --epochs 60 --companies BPCL HDL
```

Training all 5 companies × 2 models (100 epochs w/ early stopping) typically
takes 15–40 minutes on CPU, faster on GPU.

---

## 5. Forecast the next N trading days (CLI)

```bash
python -m src.forecast --symbol BPCL --model-type GRU --days 7
```

- Uses the model's own predictions recursively to step forward one NEPSE
  trading day at a time (NEPSE trades **Sunday–Thursday**; Friday/Saturday
  are automatically skipped).
- At each step, technical indicators are recomputed from the growing price
  history so the model always sees realistic feature values, not stale ones.
- Add `--csv path/to/newer_data.csv` to forecast from freshly collected data
  instead of the bundled processed history.
- Results are printed and saved to `outputs/forecasts/`.

---

## 6. Launch the Streamlit dashboard

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

**Dashboard tabs:**

| Tab | What it shows |
|-----|----------------|
|  Overview & EDA | Price/volume charts, summary stats per company |
|  Model Performance | RMSE / MAE / MAPE / Accuracy / R² / Directional Accuracy for LSTM vs GRU, per-company and overall |
|  Confusion Matrix | Up/Down directional confusion matrix + classification report (precision/recall/F1) |
|  Actual vs Predicted | Test-set actual vs predicted price chart + residual distributions |
|  1-Week Forecast | Configurable N-day-ahead forecast with an illustrative uncertainty band, table, and CSV download |

If a company/model hasn't been trained yet, the app offers a **"Quick-train
now"** button (fewer epochs, for a fast in-app demo) as a fallback to running
`src/train.py` from the command line.

You can also switch **Data source → Upload new CSV** in the sidebar to
generate a forecast on data newer than what the model was trained on,
without retraining anything.

---

## 7. Notes on methodology (for your report)

- **Directional confusion matrix**: since price prediction is a regression
  task, we derive a classification view by comparing the day-over-day
  direction (Up/Down) of the actual vs. predicted close price series. This
  answers a different, often more decision-relevant question than RMSE:
  *"would trading on this model's signal have been right or wrong?"*
- **Recursive multi-step forecasting**: only `close` is truly model-predicted
  at each future step; `open`/`high`/`low`/`volume` for synthetic future rows
  are carried forward using standard simplifying assumptions (open ≈ previous
  close, a small high/low band, volume ≈ recent rolling average) so that
  technical indicators remain computable going forward. This is a common,
  clearly-documented limitation of recursive forecasting and should be
  described as such in your report — forecast uncertainty compounds with
  horizon length, which is why the dashboard shows a widening band.
- **Chronological (not random) train/test split** is used throughout to
  avoid lookahead bias, consistent with time-series best practice.
- **MinMaxScaler is fit on the training set only**, then applied to the test
  set, avoiding data leakage.

---

## 8. Troubleshooting

- **"No trained model found"** in the app → run `python -m src.train` first,
  or use the in-app quick-train button.
- **TensorFlow install issues** → make sure you're on Python 3.9–3.11 and
  a matching `tensorflow` wheel is available for your OS/CPU.
- **"Not enough history to forecast"** → you need at least `WINDOW_SIZE`
  (60 by default) rows of usable data *after* technical-indicator warm-up
  (~30 extra rows). Provide more historical rows, or lower `WINDOW_SIZE` in
  `src/config.py` (requires retraining).
