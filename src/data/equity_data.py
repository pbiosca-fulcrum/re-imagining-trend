"""
equity_data.py

Handles reading and processing of raw US stock data.
Generates synthetic data if real data is missing.
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
    """
    df = processed_us_data()
    # Filter data from (year-2) to year
    keep_years = [year, year - 1, year - 2]
    idx_year = df.index.get_level_values("Date").year.isin(keep_years)
    return df[idx_year].copy()


def processed_us_data() -> pd.DataFrame:
    """
    Main function to load or create the processed U.S. stock dataset with columns:
    [Date, StockID, Open, High, Low, Close, Vol, Shares, Ret, MarketCap, ...].
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

    # Otherwise, read raw data or generate synthetic
    raw_us_data_path = op.join(dcf.RAW_DATA_DIR, "us_920101-200731.csv")
    if not op.exists(raw_us_data_path):
        print("Raw data file not found -> generating synthetic data instead.")
        df = _generate_synthetic_us_data()
    else:
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
            compression="gzip",
            header=0,
        )
        print(f"Finished reading data in {(time.time() - since):.2f} sec")
        df = process_raw_data_helper(df)

    # Save the processed result
    df.reset_index().to_feather(processed_us_data_path)
    return df.copy()


def _generate_synthetic_us_data() -> pd.DataFrame:
    """
    Generate synthetic data for demonstration if no real CSV is found.
    """
    all_data = []
    np.random.seed(42)
    for year in range(1993, 2020):
        for stock_id in range(5):  # fewer for demonstration
            dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="B")
            n = len(dates)
            returns = np.random.normal(0, 0.01, size=n)
            prices = 100 * np.cumprod(1 + returns)
            open_prices = np.concatenate([[100], prices[:-1]])
            high_prices = prices * (1 + abs(np.random.normal(0, 0.005, size=n)))
            low_prices = prices * (1 - abs(np.random.normal(0, 0.005, size=n)))
            close_prices = prices
            volume = np.random.randint(1000, 10000, size=n)
            shares = np.random.randint(10000, 100000, size=n)
            ret = np.concatenate([[0], prices[1:] / prices[:-1] - 1])
            df_stock = pd.DataFrame({
                "Date": dates,
                "StockID": str(stock_id),
                "Open": open_prices,
                "High": high_prices,
                "Low": low_prices,
                "Close": close_prices,
                "Vol": volume,
                "Shares": shares,
                "Ret": ret
            })
            all_data.append(df_stock)
    df = pd.concat(all_data, ignore_index=True)
    df = process_raw_data_helper(df)
    return df


def process_raw_data_helper(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard cleanup for raw data: handle missing, rename columns, compute MarketCap, etc.
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
    # remove weird placeholders
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
    df["MarketCap"] = abs(df["Close"] * df["Shares"])
    df.set_index(["Date", "StockID"], inplace=True)
    df.sort_index(inplace=True)
    df["log_ret"] = np.log(1 + df["Ret"])
    df["cum_log_ret"] = df.groupby("StockID")["log_ret"].cumsum(skipna=True)
    df["EWMA_vol"] = df.groupby("StockID")["Ret"].transform(lambda x: (x ** 2).ewm(alpha=0.05).mean().shift(1))

    # compute multi-day returns
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

    return df


def get_spy_freq_rets(freq: str) -> pd.DataFrame:
    """
    Returns SPY returns for a particular freq. If not found, synthetic are generated.
    """
    assert freq in ["week", "month", "quarter", "year"]
    file_path = str(dcf.CACHE_DIR / f"spy_{freq}_ret.csv")

    if not os.path.isfile(file_path):
        print(f"File {file_path} not found. Generating synthetic SPY {freq} returns.")
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
    else:
        spy = pd.read_csv(file_path, parse_dates=["date"])

    spy.rename(columns={"date": "Date"}, inplace=True)
    spy.set_index("Date", inplace=True)
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
    if not op.isfile(period_ret_path):
        print(f"No saved {period} data. Using synthetic approach.")
        spy = get_spy_freq_rets(period)  # returns DataFrame with index as Date, column f"{period}_ret"
        # Rename the column to match expected naming: next_{period}_ret_0delay
        spy = spy.rename(columns={f"{period}_ret": f"next_{period}_ret_0delay"})
        # Add a synthetic MarketCap column
        spy["MarketCap"] = 1e9
        # Reset index to have Date as a column
        spy = spy.reset_index()
        return spy[["Date", "MarketCap", f"next_{period}_ret_0delay"]]
    
    period_ret = pd.read_parquet(period_ret_path)
    period_ret.reset_index(inplace=True)
    return period_ret
