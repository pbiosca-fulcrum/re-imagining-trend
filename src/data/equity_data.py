# src/data/equity_data.py

"""
equity_data.py

Handles reading and processing of raw US stock data in chunks, converting columns
to float32, optionally storing a backup in Parquet, and returning the full DataFrame.

If even this approach crashes on a ~63-million-row dataset, consider:
  - Setting SAVE_BACKUP = False (skip the final Parquet write).
  - Further splitting into multiple year-partitioned Parquet files (requires more advanced logic).
  - Using a distributed solution like Dask or Polars for truly massive datasets.
"""

import os
import os.path as op
import gc
import numpy as np
import pandas as pd
import time

from src.data import dgp_config as dcf

# Toggle whether to store a final backup. If you're running out of memory,
# set this to False to skip writing the massive DataFrame entirely.
SAVE_BACKUP = True

def get_processed_us_data_by_year(year: int) -> pd.DataFrame:
    """
    Retrieve processed U.S. data for a given year plus the two prior years.
    The returned DataFrame includes rows from (year-2) to year.
    """
    df = processed_us_data()  # This loads (and possibly writes) the entire dataset
    keep_years = [year, year - 1, year - 2]
    idx_year = df.index.get_level_values("Date").year.isin(keep_years)
    return df[idx_year].copy()

def processed_us_data() -> pd.DataFrame:
    """
    Loads the processed U.S. stock dataset in a memory-efficient manner.
    If a Parquet file already exists, load from that.
    Otherwise:
      1) Read the large CSV in chunks (reducing memory usage).
      2) Convert columns, rename, drop missing 'RET', etc.
      3) Concatenate all chunks, set multi-index, compute extra columns (log_ret, EWMA, multi-day returns).
      4) Convert numeric columns to float32.
      5) If SAVE_BACKUP is True, save the final DataFrame to Parquet.
      6) Return the final DataFrame.

    If memory usage is still too high, consider:
      - Setting SAVE_BACKUP=False to skip big disk writes.
      - Partitioning by year or by StockID in separate files.
      - Using a distributed framework like Dask/Polars.
    """

    # Updated to store Parquet (rather than Feather) for huge DataFrames
    processed_us_data_path = op.join(dcf.PROCESSED_DATA_DIR, "us_ret.parquet")

    # If the Parquet file exists, just load it
    if op.exists(processed_us_data_path):
        print(f"Loading processed data from {processed_us_data_path}")
        since = time.time()
        # Use pyarrow engine by default
        df = pd.read_parquet(processed_us_data_path, engine="pyarrow")
        # Rebuild multi-index
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index(["Date", "StockID"], inplace=True)
        df.sort_index(inplace=True)
        print(f"Done loading in {(time.time() - since):.2f} sec")
        print(f"Data columns: {df.columns.tolist()}  shape={df.shape}")

        # Recompute period returns parquet files (week, month, quarter)
        for freq in ["week", "month", "quarter"]:
            col = f"Ret_{freq}"
            if col in df.columns:
                ret_df = df[df[col].notna()][["MarketCap", col]].reset_index()
                new_name = f"next_{freq}_ret_0delay"
                ret_df = ret_df.rename(columns={col: new_name})
                ret_pq_path = op.join(str(dcf.CACHE_DIR), f"us_{freq}_crsp_ret.pq")
                ret_df.to_parquet(ret_pq_path, index=False)
                print(f"Saved period returns for {freq} to {ret_pq_path}")

        return df

    # Otherwise, read raw CSV in chunks
    raw_us_data_path = op.join(dcf.RAW_DATA_DIR, "crsp_a_stock.csv.gz")
    if not op.exists(raw_us_data_path):
        raise FileNotFoundError(
            f"Raw data file not found at '{raw_us_data_path}'. "
            "Please provide real data."
        )

    print(f"Reading raw data in chunks from {raw_us_data_path}")
    since = time.time()

    # We only load the columns we actually need
    usecols = ["date", "PERMNO", "BIDLO", "ASKHI", "PRC", "VOL", "SHROUT", "OPENPRC", "RET"]
    dtypes = {
        "date": str,         # parse later
        "PERMNO": str,       # parse or convert to category eventually
        "BIDLO": np.float64,
        "ASKHI": np.float64,
        "PRC": np.float64,
        "VOL": np.float64,
        "SHROUT": np.float64,
        "OPENPRC": np.float64,
        "RET": str,          # treat as string first to catch placeholders
    }

    chunksize = 500_000
    all_chunks = []

    def convert_ret(x):
        try:
            return float(x)
        except:
            return np.nan

    # Read in loop
    chunk_idx = 0
    for chunk in pd.read_csv(raw_us_data_path, usecols=usecols, dtype=dtypes,
                             compression='infer', chunksize=chunksize):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk.dropna(subset=["date"], inplace=True)

        # Convert RET from string to float, dropping placeholder entries
        chunk["RET"] = chunk["RET"].apply(convert_ret)
        chunk.dropna(subset=["RET"], inplace=True)

        numeric_cols = ["BIDLO", "ASKHI", "PRC", "VOL", "SHROUT", "OPENPRC"]
        for col in numeric_cols:
            chunk[col] = chunk[col].abs()

        all_chunks.append(chunk)
        chunk_idx += 1
        print(f"  Processed chunk {chunk_idx}, shape={chunk.shape}")

    if not all_chunks:
        raise ValueError("No valid data found in the raw CSV after chunk processing.")

    df = pd.concat(all_chunks, ignore_index=True)
    del all_chunks  # free memory
    print(f"Concatenated all chunks: final shape {df.shape}")

    # Rename columns
    df.rename(columns={
        "date": "Date",
        "PERMNO": "StockID",
        "BIDLO": "Low",
        "ASKHI": "High",
        "PRC": "Close",
        "VOL": "Vol",
        "SHROUT": "Shares",
        "OPENPRC": "Open",
        "RET": "Ret"
    }, inplace=True)

    # Compute MarketCap
    df["MarketCap"] = df["Close"] * df["Shares"]

    # Convert to category (reduces memory if many repeated IDs)
    df["StockID"] = df["StockID"].astype("category")

    # Create multi-index
    df.set_index(["Date", "StockID"], inplace=True)
    df.sort_index(inplace=True)

    # log returns, cumulative log returns, and EWMA
    df["log_ret"] = np.log(1.0 + df["Ret"])
    df["cum_log_ret"] = df.groupby("StockID")["log_ret"].cumsum()
    df["EWMA_vol"] = df.groupby("StockID")["Ret"].transform(lambda x: (x**2).ewm(alpha=0.05).mean().shift(1))

    from .equity_data import get_period_end_dates
    for freq in ["week", "month", "quarter", "year"]:
        period_end_dates = get_period_end_dates(freq)
        mask = df.index.get_level_values("Date").isin(period_end_dates)
        freq_ret = df.groupby("StockID")["cum_log_ret"].transform(
            lambda x: np.exp(x.shift(-1) - x) - 1
        )
        df.loc[mask, f"Ret_{freq}"] = freq_ret.loc[mask]

    for i in [5, 20, 60, 65, 180, 250, 260]:
        df[f"Ret_{i}d"] = df.groupby("StockID")["cum_log_ret"].transform(
            lambda x: np.exp(x.shift(-i) - x) - 1
        )

    print(f"Finished processing raw data in {(time.time() - since):.2f} sec")
    # Convert numeric columns to float32 to reduce memory usage ~ by half
    float_cols = [
        "Low", "High", "Close", "Vol", "Shares", "Open", "Ret",
        "MarketCap", "log_ret", "cum_log_ret", "EWMA_vol",
        "Ret_week", "Ret_month", "Ret_quarter", "Ret_year",
        "Ret_5d", "Ret_20d", "Ret_60d", "Ret_65d", "Ret_180d", "Ret_250d", "Ret_260d"
    ]
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].astype(np.float32, errors="ignore")

    if SAVE_BACKUP:
        # Because writing a massive DataFrame can spike memory usage, we do it carefully
        print("Converting index back to columns for Parquet ...")
        df_out = df.reset_index()
        # Release memory for df if needed
        df = None
        gc.collect()

        # Write Parquet using PyArrow engine (snappy or gzip compression):
        print("Storing a backup to Parquet for future fast access...")
        start_write = time.time()
        df_out.to_parquet(processed_us_data_path, index=False, engine="pyarrow", compression="snappy")
        print(f"Done writing Parquet in {(time.time() - start_write):.2f} sec. Path={processed_us_data_path}")

        # Rebuild the final df in memory if we want to return it
        # (We'll read from Parquet to ensure identical data & not blow memory.)
        df = pd.read_parquet(processed_us_data_path, engine="pyarrow")
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index(["Date", "StockID"], inplace=True)
        df.sort_index(inplace=True)
        df_out = None
        gc.collect()

    # Store period returns parquet files (week, month, quarter)
    for freq in ["week", "month", "quarter"]:
        col = f"Ret_{freq}"
        if col in df.columns:
            ret_df = df[df[col].notna()][["MarketCap", col]].reset_index()
            new_name = f"next_{freq}_ret_0delay"
            ret_df = ret_df.rename(columns={col: new_name})
            ret_pq_path = op.join(str(dcf.CACHE_DIR), f"us_{freq}_crsp_ret.pq")
            ret_df.to_parquet(ret_pq_path, index=False)
            print(f"Saved period returns for {freq} to {ret_pq_path}")

    return df.copy()


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
