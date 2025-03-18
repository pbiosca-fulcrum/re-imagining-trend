
# Data Preparation Guide

This document explains how to structure real financial data for use with our chart-generation and CNN training pipeline, so you don’t have to rely on synthetic data.

## 1. Directory Structure

We recommend a folder layout like:
```
my_project/
├── ...
├── data/
│   └── stocks_dataset/
│       ├── raw_data/
│       └── processed_data/
└── ...
```
- **`raw_data/`** should contain the raw CSV files (e.g., daily OHLC data, volumes, etc.).  
- **`processed_data/`** is where your code places cleaned or feathered outputs after processing.

## 2. Required Columns in the Raw Data

For **U.S. equity data**, the pipeline expects a CSV or gzipped CSV with these columns (before renaming inside the code):

- **Date**  
- **PERMNO** (stock identifier, which we rename to `StockID`)  
- **BIDLO** (daily low price)  
- **ASKHI** (daily high price)  
- **PRC** (daily close price)  
- **VOL** (daily volume)  
- **SHROUT** (shares outstanding)  
- **OPENPRC** (daily open price)  
- **RET** (daily return, can be numeric or string with placeholders)

An example snippet:
```
date,PERMNO,BIDLO,ASKHI,PRC,VOL,SHROUT,OPENPRC,RET,EXCHCD
1993-01-04,10000,24.63,25.00,24.75,4500,2100,24.63,0.012,1
1993-01-05,10000,24.56,25.31,25.00,10000,2100,24.63,0.010,1
...
```
> Make sure that the `RET` column is purely numeric or set placeholders (`'C','B','A','.'`) for missing data so that the code can safely drop or convert them.

## 3. Converting and Storing Processed Data

When you run:
```bash
python -m src.main
```
the code will look for:
```
data/stocks_dataset/raw_data/us_920101-200731.csv
```
(or a similarly named file). If it **doesn’t** find this, it resorts to generating synthetic data.

### Steps:
1. **Place your real CSV** in `data/stocks_dataset/raw_data/`.  
2. Ensure it’s named something like `us_920101-200731.csv` or update your code path to match your dataset name.  
3. The code automatically runs `process_raw_data_helper()`, which:
   - Renames columns to `["Date","StockID","Low","High","Close","Vol","Shares","Open","Ret"]`.
   - Drops placeholders.
   - Computes `MarketCap = |Close| * |Shares|`.
   - Saves the result as `us_ret.feather` in `data/stocks_dataset/processed_data/`.

Once that’s done, your data is fully prepared. The pipeline will no longer create synthetic data.

## 4. Ensuring MarketCap is Present

The portfolio weighting logic uses the `MarketCap` column. If your real data doesn’t have shares outstanding, you must create a synthetic approximation or at least define a default. For instance:
```python
df["Shares"] = df["Shares"].fillna(1_000_000)  # fallback
df["MarketCap"] = df["Close"].abs() * df["Shares"].abs()
```
This ensures `MarketCap` is valid.

## 5. Next-Period Returns (Optional)

If you want **real** next‐period returns (week/month/quarter):
- Provide your own logic to compute e.g. `next_month_ret_0delay`.  
- Store it in `CACHE_DIR` or feed it directly in place of `get_period_ret()` so the code can skip the synthetic approach.

## 6. Common Pitfalls

1. **Missing columns**: If any required column is absent, the processing function will break.  
2. **Non-numeric placeholders**: `RET` must be convertible to float or safely removed.  
3. **Inconsistent date indexing**: The code merges or reindexes by date, so duplicates or invalid timestamps can cause errors.

That’s all! With these guidelines, your real dataset should integrate into the pipeline without needing the synthetic fallback.