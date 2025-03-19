import glob
import os
import pandas as pd

def aggregate_processed_data(processed_data_dir: str, output_file: str) -> None:
    """
    Read all per-ticker processed CSV files from the given directory,
    concatenate them into one DataFrame, sort by Date and StockID,
    and save as a Feather file.
    
    Args:
        processed_data_dir (str): Directory containing per-ticker CSV files.
        output_file (str): Path where the aggregated Feather file will be saved.
    """
    # Look for files ending with _processed.csv
    file_pattern = os.path.join(processed_data_dir, "*_processed.csv")
    csv_files = glob.glob(file_pattern)
    
    if not csv_files:
        print(f"No processed CSV files found in {processed_data_dir}.")
        return

    # Read and concatenate all CSV files
    df_list = []
    for file in csv_files:
        print(f"Reading {file}...")
        df = pd.read_csv(file)
        df_list.append(df)
    
    aggregated_df = pd.concat(df_list, ignore_index=True)
    
    # Convert 'Date' column to datetime and sort the DataFrame
    aggregated_df['Date'] = pd.to_datetime(aggregated_df['Date'], errors='coerce')
    aggregated_df.sort_values(by=['Date', 'StockID'], inplace=True)
    aggregated_df.reset_index(drop=True, inplace=True)
    
    # Save the aggregated DataFrame as a Feather file
    aggregated_df.to_feather(output_file)
    print(f"Aggregated data saved to {output_file}.")

if __name__ == "__main__":
    # Set the directory paths (adjust if needed)
    processed_data_dir = os.path.join("stocks_dataset", "processed_data")
    output_file = os.path.join(processed_data_dir, "us_ret.feather")
    aggregate_processed_data(processed_data_dir, output_file)
