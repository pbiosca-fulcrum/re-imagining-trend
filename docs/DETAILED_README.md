## Overview

This project:

1. **Generates synthetic data** for U.S. stocks (if no real data is found).  
2. **Transforms** that data into “chart images” or “1D signals.”  
3. **Trains Convolutional Neural Networks (CNNs)** (both 2D CNN for images and 1D CNN for time‐series).  
4. **Constructs a portfolio** based on the model’s predictions and calculates performance metrics.

You typically rebalance or “trade” at a period frequency such as monthly or quarterly – **not** every day. For example, if `pw=20` → that generally means monthly in your code, so you place trades once each month. If `pw=60` → you rebalance quarterly. The pipeline builds “decile” portfolios at each period end rather than daily trades.

Below, each major file is explained in detail:

---

# 1. **`src/main.py`**

**What It Does:**  
- This is **the main entry point** when you run:
  ```bash
  python -m src.main
  ```
- The steps in `main()`:
  1. **Sets up** your device (CPU vs. GPU) and some environment variables.
  2. **Generates** synthetic chart data for each year from 1993 to 2019:
     - Creates a `GenerateStockData` object for each year.
     - Calls `save_annual_data()` to produce 2D chart images, and `save_annual_ts_data()` (placeholder) for 1D data.
  3. **Trains** CNN models by calling `train_us_model()` with various parameters.  
     - Specifically, it runs a CNN2D model with `ws_list=[20], pw_list=[20]` multiple times.
  4. **Optionally builds** a portfolio (the `calculate_portfolio=True` parameter triggers that).  
  5. Prints “All tasks completed.”  

**Key Detail**:  
- The `window_size = 20` means you consider 20 daily bars for each chart.  
- The `freq = "month"` means you treat those 20 bars as monthly data.  
- **Trading Strategy**: For `pw=20 → "month"`, you rebalance once a month. If you used `pw=60 → "quarter"`, you’d rebalance quarterly.  

Inside `main()`, notice you do:

```python
train_us_model(
    ws_list=[20],
    pw_list=[20],
    ...
    calculate_portfolio=True,
    ...
)
```
That means: 
- **Window Size (ws)** of 20,  
- **Predict Window (pw)** of 20 → monthly frequency,  
- Build or calculate a portfolio each time the model finishes training.

**`breakpoint()`**: You added a `breakpoint()` line after the first CNN2D training call, so Python will drop into the debugger after training that model.

---

# 2. **`src/analysis/analysis_lib.py`**

**Purpose:**  
Houses some helper functions for analyzing CNN signals and computing correlations. Not crucial if you only focus on training and portfolio building, but it includes:

- `portfolio_performance_helper(ws, pw)`:  
  Creates a `PortfolioManager` for a given window size (`ws`) and horizon (`pw`), then calls `generate_portfolio()`.
- `corr_between_cnn_pred_and_stock_chars()`:  
  Demonstrates rank correlations between CNN predictions and stock characteristics.
- `glb_plot_sr_gain_vs_stocks_num(horizon)`:  
  Example function for plotting Sharpe ratio gain vs. number of stocks across countries.

These are mostly **analysis utilities** for post‐modeling checks.

---

# 3. **`src/analysis/regression_tables.py`**

**Purpose:**  
- Contains a class `SMRegression`, which uses **StatsModels** to fit logistic or OLS regressions.  
- It is used to see how the CNN signals correlate or predict returns in a more classical regression framework.

If you’re focusing on CNN training or data generation, you may not call this directly.

---

# 4. **`src/data/chart_dataset.py`**

**Purpose:**  
Implements PyTorch Dataset classes for your CNN. The two big classes are:

1. **`EquityDataset`** (2D dataset for chart images):  
   - `__init__` loads memmapped `.dat` files (the chart images) plus `.feather` label data.  
   - It transforms raw “Close/Ret” into classification labels (1 if `Ret>0`, else 0) or regression labels.  
   - `__getitem__` returns a single sample: a normalized image, plus label, date, `MarketCap`, etc.

2. **`TS1DDataset`** (1D dataset for time‐series signals):  
   - Similar structure but for the 1D “open/high/low/close/ma/vol” arrays instead of chart images.

**Key Takeaway**:  
When you do:
```python
DataLoader(EquityDataset(...))
```
You are effectively reading your bar‐chart images from disk and serving them to the CNN.

---

# 5. **`src/data/chart_library.py`**

**Purpose:**  
- Contains `DrawOHLC`, which draws an OHLC chart (bar, pixel, or centered_pixel).  
- The code deals with sizes, volumes, and optionally draws MAs.  
- The “centered_pixel” approach re-centers each bar around the current day’s close, so it’s a special transformation.

Essentially, **`DrawOHLC`** is how each day’s data becomes a grayscale image for your CNN. You don’t typically call this directly – it’s invoked inside your chart-generation code (`generate_chart.py`).

---

# 6. **`src/data/dgp_config.py`**

**Purpose:**  
- “DGP config” stands for Data‐Generating Process config.  
- Holds **global constants**: directories, `IMAGE_WIDTH`, `IMAGE_HEIGHT`, and the mapping `FREQ_DICT = {5: "week", 20: "month", 60: "quarter", ...}`.  
- This file is what ties `pw=20` → “month,” `pw=60` → “quarter,” etc.

---

# 7. **`src/data/equity_data.py`**

**Purpose:**  
Handles the reading and caching of the raw U.S. equity data.

1. **`processed_us_data()`**:  
   - Looks for a file `us_ret.feather` in `processed_data/`. If missing, tries to read your CSV from `raw_data/` or generate synthetic data.  
   - Renames columns to `[Date, StockID, Open, High, Low, Close, Vol, Shares, Ret, MarketCap, ...]`.

2. **`_generate_synthetic_us_data()`**:  
   - If no real CSV is found, it synthesizes random daily returns and prices.  
   - This is how you end up with “fake” data in the pipeline if you never add a real CSV.

3. **`get_period_end_dates(period)`**:  
   - Returns all period‐end dates for that freq (e.g., monthly or weekly).  
   - Typically used so you “rebalance” at each period end (monthly, etc.).

Hence, this module ensures your pipeline has stock data, either real or synthetic.

---

# 8. **`src/data/generate_chart.py`**

**Purpose:**  
- The main “chart generation” code.  
- `GenerateStockData` creates chart images for a single year. It loops through each stock, each date, to produce 2D images representing that stock’s 20‐day window.

Key functions:
- `save_annual_data()`:  
  1. **Checks** if pre-generated `.dat` and `.feather` files exist for this year.  
  2. If not, reads data from `equity_data.py`.  
  3. For each stock/date, calls `_generate_daily_features()` which draws the chart using `DrawOHLC`.  
  4. Saves all images in a memory‐mapped `.dat`.  
  5. Saves per-sample labels in a `.feather`.

- `_generate_daily_features()`:  
  1. Prepares a small chunk (20 days) of stock data.  
  2. Adjusts the price so the first day’s Close=1.0.  
  3. Calls `DrawOHLC(...)` to get a `PIL Image`.  
  4. Populates classification or regression labels.

**Result**: You get a large `.dat` file with thousands of images and a `.feather` describing them. This is then read by `EquityDataset`.

---

# 9. **`src/experiments/cnn_experiment.py`**

**Purpose:**  
Coordinates **model training** and **portfolio generation**. The main class is:

### **`Experiment`**
- **`__init__`** sets hyperparams (like `window_size`, `predict_window`, etc.).  
- **`get_train_validate_dataloaders_dict()`** builds a train/validate DataLoader from `EquityDataset` or `TS1DDataset`.  
- **`train_empirical_ensem_model()`** trains multiple seeds (“ensemble” models).  
- **`train_single_model()`** does the actual training loop:  
  - Goes through epochs, calculates loss/accuracy, does early stopping, saves best model to `checkpointX.pth.tar`.  
- **`generate_ensem_res()`** creates “ensemble results” CSV for each stock by averaging predictions from each model in the ensemble.  
- **`calculate_portfolio()`** calls the `PortfolioManager` to build decile portfolios from the predictions.

### **`train_us_model(...)`**  
This is a **public function** that you call in `main.py`. It:
1. Creates an `Experiment` object for each `(ws, pw)` pair.
2. Trains the CNN.
3. Optionally calls `exp_obj.calculate_portfolio()` to evaluate the predictions in a trading or “decile portfolio” setup.

---

# 10. **`src/model/cnn_model.py`**

**Purpose:**  
- Defines your **CNN architecture**.  
- **`Model`** is a container that sets up either 2D or 1D CNN layers, with parameters like dropout, batch_norm, etc.  
- **`CNNModel`** specifically handles 2D images. The code uses:
  - A sequence of `Conv2d`, optional `BatchNorm2d`, `ReLU` (or `LeakyReLU`), `MaxPool2d`, then flattens, then a final `Linear` layer with 2 output neurons for classification.  
- If `ts1d_model=True`, it would build a 1D CNN, but that’s partially omitted with `NotImplementedError`.

So when you see `train_us_model(ts1d_model=False)`, you’re training the 2D version of this network, which processes chart images.

---

# 11. **`src/portfolio/portfolio.py`**

**Purpose:**
- Implements a **decile-based** trading strategy.  
- The core class is `PortfolioManager`.

### Trading Strategy Explanation

1. The manager expects columns `[up_prob, MarketCap, next_{freq}_ret_{delay}delay, ...]` in `signal_df`.  
2. You call `generate_portfolio()`, which splits stocks each period into deciles based on `up_prob` (predicted probability of going up).  
3. It forms a “High” decile (the top 10% up_prob) and “Low” decile (lowest 10%).  
4. Weights them either equally (“ew”) or by MarketCap (“vw”).  
5. Each period’s return is computed, a “H-L” is the difference.  
6. The method repeats for each date in your in-sample or out-of-sample range, compiles final performance, etc.

**Do you trade daily or monthly?**  
- Because the code merges your predictions with “period_end_dates,” it trades or rebalances **once per period**. That period is weekly (pw=5), monthly (pw=20), or quarterly (pw=60) – not daily trades.  
- The code thus invests in a new decile portfolio at each period end, then measures returns for the next period.

So, you **trade** on each **period** boundary (for instance, once every month or once every quarter).

---

# 12. **`src/utils/config.py`**

**Purpose:**  
- Central place for environment paths:  
  - `WORK_DIR` → main workspace.  
  - `EXP_DIR` → experiments.  
  - `PORTFOLIO_DIR` → portfolio results.  
- Also defines default hyperparams (`BATCH_SIZE=128`, etc.).
- **`IS_YEARS = 1993..2000`**, **`OOS_YEARS = 2001..2019`** for training vs. out-of-sample sets.

---

# 13. **`src/utils/utilities.py`**

**Purpose:**  
- Misc. helper functions (e.g., `rank_normalization()`, `binary_one_hot()`, etc.).  
- Utility code for cross-entropy, star-significant formatting, etc.

Not a core training or data code; used by the pipeline for smaller tasks.

---

# 14. **`tests/` Folder**

**Purpose:**  
- Minimal example of how you might write a `unittest` testing a small function (`rank_normalization`).  
- You’d typically expand this to test your data loading, CNN layers, portfolio logic, etc.

---

## **How Often Does the Strategy Trade?**

As stated above:  
- The code rebalances **once per “period”**.  
- That period can be week (5 days), month (20 days), or quarter (60 days). The actual frequency is set by your `pw_list=[5, 20, 60, ...]` calls.  
- So you are **not** trading daily. You only trade or rebalance on each period’s end date (like the last day of the month or quarter).  

Hence, *if you run*:
```python
train_us_model(ws_list=[20], pw_list=[20], ...)
```
that means you do a **monthly** strategy. If you changed it to `[60]`, that implies you trade **quarterly**.

---

## Putting It All Together

1. **`main.py`** calls `GenerateStockData` to produce chart images for each year.  
2. Then calls `train_us_model()` to:
   - Build an `Experiment` → which sets up Datasets, loads those images, trains a CNN, and (optionally) generates a portfolio.  
3. The **CNN** sees each 20-day chart as a grayscale image, tries to predict up vs. down.  
4. **Portfolio** invests in the stocks with the highest predicted `up_prob` vs. the lowest. The difference becomes your “long–short” or “H-L” return, rebalanced monthly or quarterly.

**In summary**: 
- You generate data and images,  
- Train a model that classifies or does regression,  
- Use that model’s predictions to form decile portfolios each period,  
- Evaluate how those decile portfolios perform out-of-sample.