#!/usr/bin/env python3
import os
import time
import yfinance as yf
import pandas as pd
from tqdm import tqdm

# Define date range
START_DATE = "1993-01-01"
END_DATE = "2025-12-31"

# Directory paths for raw and processed data
BASE_DIR = os.path.join("data", "stocks_dataset")
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def load_ticker_marketcap(filename="ticker_marketcap.csv"):
    """
    Loads a CSV file with columns "Ticker,MarketCap" and returns a list of tickers
    along with a dictionary mapping each ticker to its market cap (as a float).
    """
    if not os.path.exists(filename):
        print(f"Ticker and market cap file '{filename}' not found. Exiting.")
        exit(1)
    df = pd.read_csv(filename)
    mapping = {}
    tickers = []
    for _, row in df.iterrows():
        ticker = str(row["Ticker"]).strip()
        try:
            market_cap = float(row["MarketCap"])
        except Exception as e:
            print(f"Error processing market cap for {ticker}: {e}")
            continue
        mapping[ticker] = market_cap
        tickers.append(ticker)
    return tickers, mapping

def download_stock_data(ticker, start_date=START_DATE, end_date=END_DATE):
    """
    Downloads historical OHLCV data for a given ticker using yfinance and saves the raw CSV.
    """
    try:
        # Download data with auto_adjust=False to retain the original columns.
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        if df.empty:
            print(f"No data found for {ticker}.")
            return None

        # Ensure the index has a name so reset_index creates a proper column.
        if df.index.name is None:
            df.index.name = "Date"
        df.reset_index(inplace=True)
        
        # Save raw data to CSV.
        raw_file = os.path.join(RAW_DATA_DIR, f"{ticker}_data.csv")
        df.to_csv(raw_file, index=False)
        print(f"Raw data for {ticker} saved to {raw_file}.")
        return df
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None

def process_stock_data(ticker, df, marketcap_mapping):
    """
    Processes the raw dataframe:
      - Flattens columns if they are tuples (from a multi-index) so that only the primary label remains.
      - Converts all column names to strings, strips, and lowers them.
      - Renames index column to 'date' if necessary.
      - Converts the Date column to datetime and numeric columns (close, open, high, low, volume)
        to numbers, dropping rows that fail conversion.
      - If "close" is missing but "adj close" exists, renames "adj close" to "close".
      - Computes daily returns (ret) as the percentage change of close.
      - Adds a stockid column (the ticker symbol).
      - Uses the provided market cap from the mapping and the most recent close to derive shares.
      - Computes daily market cap as daily close × shares.
      - Renames and reorders columns to the expected format.
    """
    # Debug: show initial columns
    print(f"[{ticker}] Initial columns: {df.columns.tolist()}")
    
    # Flatten columns if they are tuples (from a multi-index)
    if all(isinstance(col, tuple) for col in df.columns):
        df.columns = [col[0] for col in df.columns]
        print(f"[{ticker}] Flattened columns: {df.columns.tolist()}")

    # Standardize column names.
    df.columns = [str(col).strip().lower() for col in df.columns]
    print(f"[{ticker}] Columns after standardization: {df.columns.tolist()}")

    # Check for required columns before proceeding
    required_cols = ["date", "close"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        # Try to recover if 'date' is missing but 'index' is present.
        if "index" in df.columns and "date" in missing_cols:
            df.rename(columns={"index": "date"}, inplace=True)
            print(f"[{ticker}] Renamed 'index' to 'date'.")
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"[{ticker}] Missing required columns {missing_cols}.")
            return None

    # If the date column came in as "index", rename it.
    if "date" not in df.columns and "index" in df.columns:
        df.rename(columns={"index": "date"}, inplace=True)

    # Convert Date to datetime.
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        print(f"[{ticker}] Dates after conversion: {df['date'].head()}")

    # If "close" is missing but "adj close" exists, rename it before dropping rows.
    if "close" not in df.columns and "adj close" in df.columns:
        df.rename(columns={"adj close": "close"}, inplace=True)

    # Convert numeric columns.
    numeric_cols = ["adj close", "close", "high", "low", "open", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Drop rows with invalid dates or missing close prices.
    df = df.dropna(subset=["date", "close"])
    print(f"[{ticker}] Dataframe shape after dropping NA: {df.shape}")

    # If we still don't have a "close" column, skip processing.
    if "close" not in df.columns:
        print(f"[{ticker}] 'Close' column not found in data.")
        return None

    # Compute daily returns based on the close price.
    try:
        df["ret"] = df["close"].pct_change()
    except Exception as e:
        print(f"[{ticker}] Error computing returns: {e}")
        return None

    # Check that the ticker is in the market cap mapping.
    if ticker not in marketcap_mapping:
        print(f"[{ticker}] Market cap not found in mapping. Skipping.")
        return None
    provided_market_cap = marketcap_mapping[ticker]
    
    # Use the most recent close value to derive shares.
    current_close = df.iloc[-1]["close"]
    if current_close == 0:
        print(f"[{ticker}] Current close is 0; cannot derive shares. Skipping.")
        return None
    shares = provided_market_cap / current_close

    # Add stockid and shares columns.
    df["stockid"] = ticker
    df["shares"] = shares

    # Compute daily market cap as daily close × shares.
    df["marketcap"] = df["close"].abs() * abs(shares)

    # Rename 'volume' to 'vol' if present.
    if "volume" in df.columns:
        df.rename(columns={"volume": "vol"}, inplace=True)

    # Define the desired column order expected by the trend_replica_biosca repo.
    desired_columns = ["date", "stockid", "low", "high", "close", "vol", "shares", "open", "ret", "marketcap"]
    for col in desired_columns:
        if col not in df.columns:
            print(f"[{ticker}] Column {col} missing.")
            return None
    df = df[desired_columns]

    # Rename columns to the expected case.
    df.rename(columns={
        "date": "Date",
        "stockid": "StockID",
        "low": "Low",
        "high": "High",
        "close": "Close",
        "vol": "Vol",
        "shares": "Shares",
        "open": "Open",
        "ret": "Ret",
        "marketcap": "MarketCap"
    }, inplace=True)

    return df

def save_processed_data(ticker, df):
    """
    Saves the processed dataframe as both a CSV and a Feather file.
    """
    processed_csv = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_processed.csv")
    processed_feather = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_processed.feather")
    
    df.to_csv(processed_csv, index=False)
    df.to_feather(processed_feather)
    print(f"Processed data for {ticker} saved to:\n  CSV: {processed_csv}\n  Feather: {processed_feather}")

def main():
    tickers, marketcap_mapping = load_ticker_marketcap()  # Load tickers and market cap from one file.
    for ticker in tqdm(tickers, desc="Downloading stocks", unit="ticker"):
        print(f"\nProcessing ticker: {ticker}")
        raw_df = download_stock_data(ticker)
        if raw_df is not None:
            processed_df = process_stock_data(ticker, raw_df, marketcap_mapping)
            if processed_df is not None:
                save_processed_data(ticker, processed_df)
            else:
                print(f"Skipping processing for {ticker} due to data issues.")
        else:
            print(f"Skipping processing for {ticker} due to missing data.")
        # Sleep for 2 seconds between tickers.
        time.sleep(2)

if __name__ == "__main__":
    main()
