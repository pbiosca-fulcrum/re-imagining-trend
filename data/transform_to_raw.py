import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Adjust these if needed:
PROCESSED_DATA_DIR = "/root/dev/bryan-kelly/trend_cnn/trend_replica_biosca/data/stocks_dataset/processed_data_old"
RAW_DATA_DIR = "/root/dev/bryan-kelly/trend_cnn/trend_replica_biosca/data/stocks_dataset/raw_data"
OUTPUT_FILENAME = "us_920101-200731.csv"

os.makedirs(RAW_DATA_DIR, exist_ok=True)

def main():
    # List all "_processed.csv" files
    processed_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith("_processed.csv")]
    if not processed_files:
        print("No processed CSV files found. Exiting.")
        return

    all_rows = []
    for csv_file in tqdm(processed_files, desc="Processing stock files"):
        file_path = os.path.join(PROCESSED_DATA_DIR, csv_file)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Could not read {csv_file}: {e}")
            continue

        # Check for presence of essential columns
        needed = ["Date", "StockID", "Low", "High", "Close", "Vol", "Shares", "Open", "Ret"]
        missing = [col for col in needed if col not in df.columns]
        if missing:
            print(f"{csv_file} missing columns {missing}; filling with defaults.")
            for col in missing:
                if col == "Ret":
                    # fill with 0.0 if missing
                    df[col] = 0.0
                else:
                    df[col] = np.nan

        # Convert Date to datetime and filter out rows older than 1993
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df[df["Date"].notnull()]  # drop invalid dates
            df = df[df["Date"].dt.year >= 1993]
        else:
            print(f"{csv_file} has no valid Date column after reading. Skipping.")
            continue

        # Ensure the file contains data for 1993
        if 1993 not in df["Date"].dt.year.values:
            print(f"{csv_file} does not contain data for 1993. Skipping.")
            continue

        # If the file is empty after filtering, skip
        if df.empty:
            print(f"{csv_file} has no data from 1993 onward. Skipping.")
            continue

        # Rename to the columns equity_data.py expects
        # (date, PERMNO, BIDLO, ASKHI, PRC, VOL, SHROUT, OPENPRC, RET, EXCHCD)
        # We'll fill EXCHCD with 1 unless you want a random/existing value
        rename_dict = {
            "Date": "date",
            "StockID": "PERMNO",
            "Low": "BIDLO",
            "High": "ASKHI",
            "Close": "PRC",
            "Vol": "VOL",
            "Shares": "SHROUT",
            "Open": "OPENPRC",
            "Ret": "RET",
        }
        df = df.rename(columns=rename_dict)

        # We always fill EXCHCD with 1 by default
        df["EXCHCD"] = 1

        for col in ["BIDLO", "ASKHI", "PRC", "VOL", "SHROUT", "OPENPRC"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).abs()

        # For RET, keep signs
        df["RET"] = pd.to_numeric(df["RET"], errors="coerce").fillna(0.0)


        # Sort by date
        df = df.sort_values(by="date")

        # Keep just the needed columns in final order
        final_cols = ["date", "PERMNO", "BIDLO", "ASKHI", "PRC", "VOL", "SHROUT", "OPENPRC", "RET", "EXCHCD"]
        df = df[final_cols]

        # Accumulate
        all_rows.append(df)
        # # Only process the first 20 files that have data from 1993
        # if len(all_rows) == 50:
        #     print("Processed first 20 valid files with data from 1993.")
        #     print(f"all_rows contains {len(all_rows)} DataFrames and is {all_rows}.")
        #     break

    if not all_rows:
        print("No data found after processing. Nothing to write.")
        return

    # Merge into one DataFrame
    big_df = pd.concat(all_rows, ignore_index=True)
    # Sort by date, then by PERMNO
    big_df.sort_values(by=["date", "PERMNO"], inplace=True)

    # Write to single CSV
    out_path = os.path.join(RAW_DATA_DIR, OUTPUT_FILENAME)
    big_df.to_csv(out_path, index=False)
    print(f"Successfully wrote merged raw CSV to {out_path}")

if __name__ == "__main__":
    main()
