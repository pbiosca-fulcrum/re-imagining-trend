# src/main.py
"""
Main entry point for the project.

This script demonstrates a workflow:
1. Generate bar chart data for multiple years (2D CNN images).
2. Train CNN models.
3. Construct a sample portfolio using the trained model.
4. Print or log relevant results.
5. DEBUG PLOTS: Show highest/lowest confidence predictions on the CNN images.

Run:
    python -m src.main
"""

import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from src.experiments.cnn_experiment import train_us_model, get_bl_exp_obj
from src.data.generate_chart import GenerateStockData
from src.data.chart_dataset import EquityDataset
from src.analysis.analysis_lib import plot_confidence_debug_charts


def main() -> None:
    """
    Main function to run the pipeline:
    - Generates chart data for multiple years (2D bar charts).
    - Trains CNN models using certain parameters.
    - (Optional) constructs portfolios, runs analysis, etc.
    - Finally, produces debug plots for the highest/lowest confidence predictions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)

    # # (1) Generate bar chart data for a range of years
    # year_list = list(range(1993, 2026))
    # chart_type = "bar"
    # window_size = 5
    # freq = "week"
    # ma_lags = [window_size]
    # volume_bar = True

    # for year in year_list:
    #     print(f"Generating data -> window {window_size}D freq {freq} chart {chart_type} year {year}")
    #     dgp_obj = GenerateStockData(
    #         country="USA",
    #         year=year,
    #         window_size=window_size,
    #         freq=freq,
    #         chart_freq=1,
    #         ma_lags=ma_lags,
    #         volume_bar=volume_bar,
    #         need_adjust_price=True,
    #         allow_tqdm=True,
    #         chart_type=chart_type,
    #     )
    #     # Generate CNN2D Data
    #     dgp_obj.save_annual_data()

    #     # If you also want 1D time-series, keep the stub call:
    #     # This will not do anything except print an info message.
    #     dgp_obj.save_annual_ts_data()

    # # (2) Train CNN models
    # train_us_model(
    #     ws_list=[5],
    #     pw_list=[5],
    #     total_worker=1,
    #     calculate_portfolio=True,  # also does ensem_res
    #     ts1d_model=False,
    #     ts_scale="image_scale",
    #     regression_label=None,
    #     pf_delay_list=[0],
    #     lr=1e-4,
    # )
    # print("CNN2D model training completed.")

    # (3) Example: After training, show debug plots of top/bottom confidence
    print("\n=== Generating Debug Plots for Highest/Lowest Confidence Samples ===")

    debug_exp = get_bl_exp_obj(
        ws=5,
        pw=5,
        lr=1e-4,
        drop_prob=0.5,
        chart_type="bar",
        has_volume_bar=True,
    )

    debug_exp.generate_ensem_res(freq="week", load_saved_data=False, year_list=[2025])
    ensem_res_2025 = debug_exp.load_ensem_res(year=2025, multiindex=True, freq="week")
    if ensem_res_2025.empty:
        print("[WARN] No ensemble results for 2025. Skipping debug chart plots.")
    else:
        dataset_2025 = EquityDataset(
            window_size=5,
            predict_window=5,
            freq="week",
            year=2025,
            country="USA",
            has_volume_bar=True,
            has_ma=True,
            chart_type="bar",
            annual_stocks_num="all",
            tstat_threshold=0,
            remove_tail=False,
            delayed_ret=0,
        )

        plot_confidence_debug_charts(
            ensem_res=ensem_res_2025,
            dataset_obj=dataset_2025,
            top_k=5,
            bottom_k=5,
            ret_column="next_week_ret_0delay",
            out_dir="debug_charts_2025"
        )

    print("All tasks completed.")


if __name__ == "__main__":
    main()
