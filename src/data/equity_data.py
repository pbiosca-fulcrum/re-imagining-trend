# src/data/equity_data.py
# src/data/equity_data.py

"""
equity_data.py

Handles reading and processing of raw US stock data.
Previously, it would generate synthetic data if real data was missing.
**Now synthetic data generation has been disabled**. If the raw file is not found,
the code raises FileNotFoundError.
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

    The returned DataFrame includes rows from (year-2) to year, so you can do
    lookbacks that span multiple calendar years without losing the older data.
    """
    df = processed_us_data()
    # Filter data from (year-2) to year
    keep_years = [year, year - 1, year - 2]
    idx_year = df.index.get_level_values("Date").year.isin(keep_years)
    return df[idx_year].copy()


def processed_us_data() -> pd.DataFrame:
    """
    Main function to load the processed U.S. stock dataset with columns like:
        [Date, StockID, Open, High, Low, Close, Vol, Shares, Ret, MarketCap, ...]
    By default, it expects a file called 'us_ret.feather' in the PROCESSED_DATA_DIR
    containing real data. If that file is missing, the code raises an error (synthetic
    data generation is now disabled).
    """
    processed_us_data_path = op.join(dcf.PROCESSED_DATA_DIR, "us_ret.feather")
    if op.exists(processed_us_data_path):
        print(f"Loading processed data from {processed_us_data_path}")
        since = time.time()
        df = pd.read_feather(processed_us_data_path)
        df.set_index(["Date", "StockID"], inplace=True)
        df.sort_index(inplace=True)
        print(f"Done loading in {(time.time() - since):.2f} sec")
        return df.copy()

    # If we don't have a saved Feather, we rely on a raw CSV. We will NOT generate synthetic data.
    raw_us_data_path = op.join(dcf.RAW_DATA_DIR, "us_920101-200731.csv")
    if not op.exists(raw_us_data_path):
        raise FileNotFoundError(
            f"Raw data file not found at '{raw_us_data_path}'. "
            "Synthetic data generation has been disabled. Please provide real data."
        )

    print(f"Reading raw data from {raw_us_data_path}")
    since = time.time()
    df = pd.read_csv(
        raw_us_data_path,
        parse_dates=["date"],
        dtype={
            "PERMNO": str,
            "BIDLO": np.float64,
            "ASKHI": np.float64,
            "PRC": np.float64,
            "VOL": np.float64,
            "SHROUT": np.float64,
            "OPENPRC": np.float64,
            "RET": object,
            "EXCHCD": np.float64,
        },
        header=0,
    )
    print(f"Finished reading data in {(time.time() - since):.2f} sec")
    df = process_raw_data_helper(df)
    
    # --- New code: Save period returns as parquet files in CACHE_DIR ---
    for freq in ["week", "month", "quarter"]:
        col = f"Ret_{freq}"
        if col in df.columns:
            ret_df = df[df[col].notna()][[col]].reset_index()
            new_name = f"next_{freq}_ret"
            ret_df = ret_df.rename(columns={col: new_name})
            ret_pq_path = op.join(str(dcf.CACHE_DIR), f"us_{freq}_ret.pq")
            ret_df.to_parquet(ret_pq_path, index=False)
            print(f"Saved period returns for {freq} to {ret_pq_path}")
    # ---------------------------------------------------------------------

    # Save to feather for faster reload next time
    df.reset_index().to_feather(processed_us_data_path)
    return df.copy()


def process_raw_data_helper(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard cleanup for raw data: handle missing, rename columns, compute MarketCap, etc.

    Steps:
    1. Rename columns to the standard set we use.
    2. Drop or fix invalid placeholders.
    3. Compute MarketCap as absolute(Close * Shares).
    4. Add 'log_ret', 'cum_log_ret', and 'EWMA_vol' columns.
    5. Compute multi-day returns like Ret_week, Ret_month, etc. if possible.

    Returns:
        DataFrame with multi-index [Date, StockID].
    """
    df = df.rename(
        columns={
            "date": "Date",
            "PERMNO": "StockID",
            "BIDLO": "Low",
            "ASKHI": "High",
            "PRC": "Close",
            "VOL": "Vol",
            "SHROUT": "Shares",
            "OPENPRC": "Open",
            "RET": "Ret",
        }
    )
    df["StockID"] = df["StockID"].astype(str)
    df["Ret"] = df["Ret"].astype(str)

    # Remove weird placeholders
    df = df.replace({
        "Close": {0: np.nan},
        "Open": {0: np.nan},
        "High": {0: np.nan},
        "Low": {0: np.nan},
        "Ret": {"C": np.nan, "B": np.nan, "A": np.nan, ".": np.nan},
        "Vol": {0: np.nan, (-99): np.nan},
    })

    if "Shares" not in df.columns:
        df["Shares"] = 0
    df["Ret"] = df["Ret"].astype(np.float64)
    df = df.dropna(subset=["Ret"])

    numeric_cols = ["Close", "Open", "High", "Low", "Vol", "Shares"]
    df[numeric_cols] = df[numeric_cols].abs()

    # Compute market cap
    df["MarketCap"] = abs(df["Close"] * df["Shares"])

    # Set multi-index
    df.set_index(["Date", "StockID"], inplace=True)
    df.sort_index(inplace=True)

    # Compute log returns and EWMA volatility as an example
    df["log_ret"] = np.log(1 + df["Ret"])
    df["cum_log_ret"] = df.groupby("StockID")["log_ret"].cumsum(skipna=True)
    df["EWMA_vol"] = df.groupby("StockID")["Ret"].transform(lambda x: (x ** 2).ewm(alpha=0.05).mean().shift(1))

    # Compute multi-day returns for various frequencies
    for freq in ["week", "month", "quarter", "year"]:
        period_end_dates = get_period_end_dates(freq)
        mask = df.index.get_level_values("Date").isin(period_end_dates)
        freq_ret = df.groupby("StockID")["cum_log_ret"].transform(
            lambda x: np.exp(x.shift(-1) - x) - 1
        )
        df.loc[mask, f"Ret_{freq}"] = freq_ret.loc[mask]

    # Example for specific day-lags
    for i in [5, 20, 60, 65, 180, 250, 260]:
        df[f"Ret_{i}d"] = df.groupby("StockID")["cum_log_ret"].transform(
            lambda x: np.exp(x.shift(-i) - x) - 1
        )

    return df


def get_spy_freq_rets(freq: str) -> pd.DataFrame:
    """
    Returns SPY returns for a particular freq. If not found, code attempts
    to generate them randomly. (Used for some placeholders.)
    """
    assert freq in ["week", "month", "quarter", "year"]
    file_path = str(dcf.CACHE_DIR / f"spy_{freq}_ret.csv")
    print(f"DEBUG: In get_spy_freq_rets, looking for file: {file_path}")
    if not op.isfile(file_path):
        print(f"DEBUG: File {file_path} not found. Generating synthetic SPY {freq} returns (this is separate from equity data).")
        start_date = pd.Timestamp("1993-01-01")
        end_date = pd.Timestamp("2019-12-31")
        if freq == "week":
            dates = pd.date_range(start=start_date, end=end_date, freq="W-FRI")
        elif freq == "month":
            dates = pd.date_range(start=start_date, end=end_date, freq="M")
        elif freq == "quarter":
            dates = pd.date_range(start=start_date, end=end_date, freq="Q")
        else:  # year
            dates = pd.date_range(start=start_date, end=end_date, freq="A-DEC")
        np.random.seed(42)
        mean_return = 0.002
        volatility = 0.02
        period_returns = np.random.normal(mean_return, volatility, size=len(dates))
        data = {
            "date": dates,
            f"{freq}_ret": period_returns
        }
        spy = pd.DataFrame(data)
        spy.to_csv(file_path, index=False)
        print(f"DEBUG: Synthetic SPY returns generated. Head:\n{spy.head()}")
    else:
        spy = pd.read_csv(file_path, parse_dates=["date"])
        print(f"DEBUG: Found SPY returns file. Head:\n{spy.head()}")
        # Print the full path to this file
        print(f"DEBUG: Full path to SPY returns file: {op.abspath(file_path)}")
    spy.rename(columns={"date": "Date"}, inplace=True)
    spy.set_index("Date", inplace=True)
    print("DEBUG: Returning SPY returns with index (first 5):\n", spy.index[:5])
    breakpoint()
    return spy


def get_period_end_dates(period: str) -> pd.DatetimeIndex:
    """
    For a given period freq ('week', 'month', 'quarter', 'year'),
    retrieve all the period-end dates from SPY data as proxies.
    """
    spy = get_spy_freq_rets(period)
    return spy.index

def get_period_ret(period: str, country: str = "USA") -> pd.DataFrame:
    """
    Load the period returns for a country. Currently only supporting "USA".
    If not found, generate synthetic period returns using SPY data.
    """
    assert country == "USA"
    assert period in ["week", "month", "quarter"]
    period_ret_path = op.join(dcf.CACHE_DIR, f"us_{period}_ret.pq")
    print(f"DEBUG: In get_period_ret for period '{period}', checking file: {period_ret_path}")
    breakpoint()
    if not op.isfile(period_ret_path):
        print(f"DEBUG: No saved {period} data. Using synthetic approach for monthly/quarterly SPY.")
        spy = get_spy_freq_rets(period)  # returns DataFrame with index as Date, column f"{freq}_ret"
        spy = spy.rename(columns={f"{period}_ret": f"next_{period}_ret_0delay"})
        # Add a synthetic MarketCap column
        spy["MarketCap"] = 1e9
        # Reset index to have Date as a column
        spy = spy.reset_index()
        print("DEBUG: Synthetic period returns generated, head:\n", spy.head())
        return spy[["Date", "MarketCap", f"next_{period}_ret_0delay"]]
    period_ret = pd.read_parquet(period_ret_path)
    period_ret.reset_index(inplace=True)
    print("DEBUG: Loaded period returns from file, head:\n", period_ret.head())
    breakpoint()
    return period_ret


# --- Added analysis code for return balance ---
def analyze_return_balance() -> None:
    """
    Analyze the balance of returns in the processed US data for multi-day windows (R5, R20, R60).
    It prints the count of positive, negative, and zero returns, as well as percentages.
    """
    df = processed_us_data()
    windows = [5, 20, 60, 250]
    for window in windows:
        col = f"Ret_{window}d"
        if col not in df.columns:
            print(f"Column {col} not found in processed data.")
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
    
    print("\nReturn balance analysis complete. Breaking for inspection.")
    breakpoint()


if __name__ == "__main__":
    # analyze_return_balance()
    
    processed_us_data()
