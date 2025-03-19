"""
Main entry point for the project.

This script demonstrates a workflow:
1. Generate synthetic chart data for US stocks (if no real data present).
2. Train CNN models.
3. Construct a sample portfolio using the trained model.
4. Print or log relevant results.

Run:
    python -m src.main
"""

import torch
import os

# Example: limiting GPU usage to GPU #0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from src.experiments.cnn_experiment import train_us_model
from src.data.generate_chart import GenerateStockData


def main() -> None:
    """
    Main function to run the pipeline:
    - Generates chart data for multiple years.
    - Trains CNN models using certain parameters.
    - (Optional) constructs portfolios, runs analysis, etc.
    """
    # Decide on CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)

    # Generate synthetic bar chart data for a range of years
    year_list = list(range(1993, 2025))
    chart_type = "bar"
    window_size = 20
    freq = "month"
    ma_lags = [window_size]
    volume_bar = True

    for year in year_list:
        print(f"Generating data -> window {window_size}D freq {freq} chart {chart_type} year {year}")
        dgp_obj = GenerateStockData(
            country="USA",
            year=year,
            window_size=window_size,
            freq=freq,
            chart_freq=1,  # For time-scale I20/R20 to R5/R5, set ws=20 and chart_freq=4
            ma_lags=ma_lags,
            volume_bar=volume_bar,
            need_adjust_price=True,
            allow_tqdm=True,
            chart_type=chart_type,
        )
        # Generate CNN2D Data
        dgp_obj.save_annual_data()
        # Generate CNN1D Data
        dgp_obj.save_annual_ts_data()

    # Train CNN models (U.S. example)
    # CNN2D
    train_us_model(
        ws_list=[20],
        pw_list=[20],
        total_worker=1,
        calculate_portfolio=True,
        ts1d_model=False,
        ts_scale="image_scale",
        regression_label=None,
        pf_delay_list=[0],
        lr=1e-4,
    )
    print(f"CNN2D model training completed.")
    
    train_us_model(
        ws_list=[20],
        pw_list=[20],
        total_worker=1,
        calculate_portfolio=True,
        ts1d_model=False,
        ts_scale="image_scale",
        regression_label=None,
        pf_delay_list=[0],
        lr=1e-4,
    )
    print(f"CNN2D model training completed.")
    # train_us_model(
    #     ws_list=[20],
    #     pw_list=[60],
    #     total_worker=1,
    #     calculate_portfolio=True,
    #     ts1d_model=False,
    #     ts_scale="image_scale",
    #     regression_label=None,
    #     pf_delay_list=[0],
    #     lr=1e-4,
    # )
    # CNN1D
    # train_us_model(
    #     ws_list=[20],
    #     pw_list=[20],
    #     total_worker=1,
    #     calculate_portfolio=True,
    #     ts1d_model=True,
    #     ts_scale="image_scale",
    #     regression_label=None,
    #     pf_delay_list=[0],
    #     lr=1e-4,
    # )
    # print(f"CNN1D model training completed.")
    
    # # Timescale variations
    # train_us_model(
    #     ws_list=[20],
    #     pw_list=[20],
    #     total_worker=1,
    #     calculate_portfolio=True,
    #     ts1d_model=False,
    #     ts_scale="image_scale",
    #     regression_label=None,
    #     pf_delay_list=[0],
    #     lr=1e-5,
    # )
    # print(f"Timescale variations completed.")
    
    # train_us_model(
    #     ws_list=[60],
    #     pw_list=[60],
    #     total_worker=1,
    #     calculate_portfolio=True,
    #     ts1d_model=False,
    #     ts_scale="image_scale",
    #     regression_label=None,
    #     pf_delay_list=[0],
    #     lr=1e-5,
    # )

    print("All tasks completed.")


if __name__ == "__main__":
    main()
