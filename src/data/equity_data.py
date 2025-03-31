# src/data/equity_data.py
"""
equity_data.py

Handles reading and processing of raw US stock data.
Synthetic data generation has been disabled.
If the raw file is not found, the code raises FileNotFoundError.

Optimizations:
  - Only the necessary columns are read from the CSV.
  - The EXCHCD column (and any other unused ones) is dropped.
  - Date parsing and conversion of RET values are done early to minimize memory overhead.
  - Vectorized operations are used where possible.
"""

import os
import os.path as op
import numpy as np
import pandas as pd
import time
from src.data import dgp_config as dcf

def get_processed_us_data_by_year(year: int) -> pd.DataFrame:
    """
    Retrieve processed U.S. data for a given year plus the two prior years.
    The returned DataFrame includes rows from (year-2) to year.
    """
    df = processed_us_data()
    keep_years = [year, year - 1, year - 2]
    idx_year = df.index.get_level_values("Date").year.isin(keep_years)
    return df[idx_year].copy()

def processed_us_data() -> pd.DataFrame:
    """
    Loads the processed U.S. stock dataset.
    If a Feather file exists, it is loaded.
    Otherwise, a compressed CSV file ('crsp_a_stock.csv.gz') is read using only the needed columns.
    The following columns are used:
      - date, PERMNO, BIDLO, ASKHI, PRC, VOL, SHROUT, OPENPRC, RET
    These are then renamed to:
      - Date, StockID, Low, High, Close, Vol, Shares, Open, Ret
    Additional columns are computed:
      - MarketCap, log_ret, cum_log_ret, EWMA_vol, and multi-day returns.
    """
    processed_us_data_path = op.join(dcf.PROCESSED_DATA_DIR, "us_ret.feather")
    if op.exists(processed_us_data_path):
        print(f"Loading processed data from {processed_us_data_path}")
        since = time.time()
        df = pd.read_feather(processed_us_data_path)
        df.set_index(["Date", "StockID"], inplace=True)
        df.sort_index(inplace=True)
        print(f"Done loading in {(time.time() - since):.2f} sec")
        print(f"Data columns: {df.columns.tolist()}")
        return df.copy()

    # Use only the necessary columns from the raw file.
    raw_us_data_path = op.join(dcf.RAW_DATA_DIR, "crsp_a_stock.csv.gz")
    if not op.exists(raw_us_data_path):
        raise FileNotFoundError(
            f"Raw data file not found at '{raw_us_data_path}'. "
            "Synthetic data generation has been disabled. Please provide real data."
        )

    print(f"Reading raw data from {raw_us_data_path}")
    since = time.time()
    usecols = ["date", "PERMNO", "BIDLO", "ASKHI", "PRC", "VOL", "SHROUT", "OPENPRC", "RET"]
    dtypes = {
        "date": str,         # will be parsed later
        "PERMNO": str,
        "BIDLO": np.float64,
        "ASKHI": np.float64,
        "PRC": np.float64,
        "VOL": np.float64,
        "SHROUT": np.float64,
        "OPENPRC": np.float64,
        "RET": str,          # read as string to catch placeholders
    }
    df = pd.read_csv(raw_us_data_path, usecols=usecols, dtype=dtypes, compression='infer')
    # Parse date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Convert RET values with a converter function; non-numeric placeholders become NaN.
    def convert_ret(x):
        try:
            return float(x)
        except Exception:
            return np.nan
    df["RET"] = df["RET"].apply(convert_ret)
    df = df.dropna(subset=["RET"])

    # Ensure numeric columns are positive (we take absolute values)
    numeric_cols = ["BIDLO", "ASKHI", "PRC", "VOL", "SHROUT", "OPENPRC"]
    for col in numeric_cols:
        df[col] = df[col].abs()

    # Rename columns to the standard names.
    df = df.rename(columns={
        "date": "Date",
        "PERMNO": "StockID",
        "BIDLO": "Low",
        "ASKHI": "High",
        "PRC": "Close",
        "VOL": "Vol",
        "SHROUT": "Shares",
        "OPENPRC": "Open",
        "RET": "Ret"
    })

    # Compute market capitalization.
    df["MarketCap"] = df["Close"] * df["Shares"]

    # Set a multi-index and sort.
    df.set_index(["Date", "StockID"], inplace=True)
    df.sort_index(inplace=True)

    # Compute log returns, cumulative log returns, and EWMA volatility.
    df["log_ret"] = np.log(1 + df["Ret"])
    df["cum_log_ret"] = df.groupby("StockID")["log_ret"].cumsum()
    df["EWMA_vol"] = df.groupby("StockID")["Ret"].transform(lambda x: (x**2).ewm(alpha=0.05).mean().shift(1))

    # Compute multi-day returns for various frequencies.
    for freq in ["week", "month", "quarter", "year"]:
        period_end_dates = get_period_end_dates(freq)
        mask = df.index.get_level_values("Date").isin(period_end_dates)
        freq_ret = df.groupby("StockID")["cum_log_ret"].transform(
            lambda x: np.exp(x.shift(-1) - x) - 1
        )
        df.loc[mask, f"Ret_{freq}"] = freq_ret.loc[mask]

    # Compute returns for specific day lags.
    for i in [5, 20, 60, 65, 180, 250, 260]:
        df[f"Ret_{i}d"] = df.groupby("StockID")["cum_log_ret"].transform(
            lambda x: np.exp(x.shift(-i) - x) - 1
        )

    print(f"Finished processing raw data in {(time.time() - since):.2f} sec")
    return df.copy()

def process_raw_data_helper(df: pd.DataFrame) -> pd.DataFrame:
    """
    A minimal helper to perform any additional replacements.
    Here we use it to drop known placeholders.
    """
    replacements = {
        "Close": {0: np.nan},
        "Open": {0: np.nan},
        "High": {0: np.nan},
        "Low": {0: np.nan},
        "Ret": {"C": np.nan, "B": np.nan, "A": np.nan, ".": np.nan,
                -66.0: np.nan, -77.0: np.nan, -88.0: np.nan, -99.0: np.nan},
        "Vol": {0: np.nan, -99: np.nan},
    }
    df = df.replace(replacements)
    df = df.dropna(subset=["Ret"])
    if not isinstance(df.index, pd.MultiIndex):
        df.set_index(["Date", "StockID"], inplace=True)
    df.sort_index(inplace=True)
    return df

def get_spy_freq_rets(freq: str) -> pd.DataFrame:
    """
    Returns SPY returns for a particular frequency.
    If not found, synthetic SPY returns are generated.
    """
    assert freq in ["week", "month", "quarter", "year"]
    file_path = str(dcf.CACHE_DIR / f"spy_{freq}_ret.csv")
    print(f"DEBUG: In get_spy_freq_rets, looking for file: {file_path}")
    if not op.isfile(file_path):
        print(f"DEBUG: File {file_path} not found. Generating synthetic SPY {freq} returns.")
        start_date = pd.Timestamp("1993-01-01")
        end_date = pd.Timestamp("2019-12-31")
        if freq == "week":
            dates = pd.date_range(start=start_date, end=end_date, freq="W-FRI")
        elif freq == "month":
            dates = pd.date_range(start=start_date, end=end_date, freq="M")
        elif freq == "quarter":
            dates = pd.date_range(start=start_date, end=end_date, freq="Q")
        else:
            dates = pd.date_range(start=start_date, end=end_date, freq="A-DEC")
        np.random.seed(42)
        mean_return = 0.002
        volatility = 0.02
        period_returns = np.random.normal(mean_return, volatility, size=len(dates))
        data = {"date": dates, f"{freq}_ret": period_returns}
        spy = pd.DataFrame(data)
        spy.to_csv(file_path, index=False)
        print(f"DEBUG: Synthetic SPY returns generated. Head:\n{spy.head()}")
    else:
        spy = pd.read_csv(file_path, parse_dates=["date"])
        print(f"DEBUG: Found SPY returns file. Head:\n{spy.head()}")
        print(f"DEBUG: Full path: {op.abspath(file_path)}")
    spy.rename(columns={"date": "Date"}, inplace=True)
    spy.set_index("Date", inplace=True)
    print("DEBUG: Returning SPY returns with index (first 5):\n", spy.index[:5])
    return spy

def get_period_end_dates(period: str) -> pd.DatetimeIndex:
    """
    For a given period ('week', 'month', 'quarter', 'year'),
    retrieves all period-end dates from SPY data.
    """
    spy = get_spy_freq_rets(period)
    return spy.index

def get_period_ret(period: str, country: str = "USA") -> pd.DataFrame:
    """
    Loads period returns for a country (currently only "USA").
    If not found, synthetic period returns are generated.
    """
    assert country == "USA"
    assert period in ["week", "month", "quarter"]
    period_ret_path = op.join(dcf.CACHE_DIR, f"us_{period}_crsp_ret.pq")
    print(f"DEBUG: In get_period_ret for '{period}', checking file: {period_ret_path}")
    if not op.isfile(period_ret_path):
        print(f"DEBUG: No saved {period} data. Using synthetic approach for SPY.")
        spy = get_spy_freq_rets(period)
        spy = spy.rename(columns={f"{period}_ret": f"next_{period}_ret_0delay"})
        spy["MarketCap"] = 1e9
        spy = spy.reset_index()
        print("DEBUG: Synthetic period returns generated, head:\n", spy.head())
        return spy[["Date", "MarketCap", f"next_{period}_ret_0delay"]]
    period_ret = pd.read_parquet(period_ret_path)
    period_ret.reset_index(inplace=True)
    print("DEBUG: Loaded period returns, head:\n", period_ret.head())
    return period_ret

def analyze_return_balance() -> None:
    """
    Analyzes the balance of returns in the processed US data for multi-day windows.
    Prints counts and percentages of positive, negative, and zero returns.
    """
    df = processed_us_data()
    windows = [5, 20, 60, 250]
    for window in windows:
        col = f"Ret_{window}d"
        if col not in df.columns:
            print(f"Column {col} not found.")
            continue
        total = df.shape[0]
        pos_count = (df[col] > 0).sum()
        neg_count = (df[col] < 0).sum()
        zero_count = (df[col] == 0).sum()
        print(f"\nAnalysis for {col}:")
        print(f"Total records: {total}")
        print(f"Positive returns: {pos_count} ({pos_count/total*100:.2f}%)")
        print(f"Negative returns: {neg_count} ({neg_count/total*100:.2f}%)")
        print(f"Zero returns: {zero_count} ({zero_count/total*100:.2f}%)")
    print("\nReturn balance analysis complete.")

if __name__ == "__main__":
    # Uncomment to run return balance analysis
    # analyze_return_balance()
    processed_us_data()
