# src/analysis/analysis_lib.py
"""
analysis_lib.py

Provides tools for analyzing CNN-based signals, portfolio performance,
and correlation with stock-level fundamentals or characteristics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from numpy.polynomial.polynomial import polyfit

from src.data.dgp_config import FREQ_DICT, CACHE_DIR, PORTFOLIO, INTERNATIONAL_COUNTRIES
from src.utils.config import OOS_YEARS
from src.portfolio.portfolio import PortfolioManager
from src.utils import utilities as ut


def portfolio_performance_helper(ws: int, pw: int) -> None:
    """
    Helper to generate portfolio performance metrics for a given window size
    and prediction horizon.
    """
    assert ws in [5, 20, 60] and pw in [5, 20, 60]
    freq = FREQ_DICT[pw]
    signal_df = pd.read_csv(CACHE_DIR / f"{freq}ly_prediction_with_rets.csv")
    signal_df["Date"] = pd.to_datetime(signal_df["Date"], dayfirst=True)
    signal_df["StockID"] = signal_df["StockID"].astype(str)
    signal_df = signal_df.set_index(["Date", "StockID"])
    df = signal_df.rename({f"CNN{ws}D{pw}P": "up_prob"}, axis="columns")
    df = df[["up_prob", "MarketCap"]].copy()
    portfolio_dir = PORTFOLIO / f"cnn_{freq}ly" / f"CNN{ws}D{pw}P"
    portfolio = PortfolioManager(df, freq=freq, portfolio_dir=portfolio_dir)
    portfolio.generate_portfolio()


def load_cnn_and_monthly_stock_char(data_type: str) -> pd.DataFrame:
    """
    Load CNN predictions plus monthly stock characteristics for in-sample (is) or out-of-sample (oos).
    """
    assert data_type in ["is", "oos"]
    save_path = CACHE_DIR / f"cnn_and_monthly_stock_char_{data_type}.parquet"
    print(f"loading from {save_path}")
    df = pd.read_parquet(save_path)
    return df


def corr_between_cnn_pred_and_stock_chars() -> pd.DataFrame:
    """
    Compute average cross-sectional rank correlation
    between CNN up_prob and various stock characteristics.
    """
    stock_chars = [
        "MOM", "STR", "Lag Weekly Return", "TREND", "Beta", "Volatility",
        "52WH", "Bid-Ask", "Dollar Volume", "Zero Trade",
        "Price Delay", "Size", "Illiquidity",
    ]
    df_corr = pd.DataFrame(columns=stock_chars)
    stock_char_df = load_cnn_and_monthly_stock_char("oos")

    for ws in [5, 20, 60]:
        for pw in [5, 20, 60]:
            for c in stock_chars:
                def up_prob_char_rank_corr(sub_df: pd.DataFrame) -> float:
                    prob_rank = sub_df[f"I{ws}/R{pw}"].rank(method="average", ascending=False)
                    char_rank = sub_df[c].rank(method="average", ascending=False)
                    return char_rank.corr(prob_rank, method="spearman")

                corr_series = stock_char_df.groupby("Date").apply(up_prob_char_rank_corr)
                df_corr.loc[f"I{ws}/R{pw}", c] = f"{corr_series.mean():.2f}"

    return df_corr


def glb_plot_sr_gain_vs_stocks_num(horizon: int) -> None:
    """
    Generate a plot of Sharpe Ratio Gain vs. number of stocks for different countries.
    """
    sr_df = international_sr_table(horizon)
    sr_df = sr_df[sr_df.index.isin(INTERNATIONAL_COUNTRIES)]

    for weight_type in ["ew", "vw"]:
        fig, ax = plt.subplots()
        stock_number = sr_df[("del2", "Stock Count")]
        sr_gain = sr_df[(weight_type, "Transfer-Re-train Value")]
        ax.scatter(stock_number, sr_gain)

        texts = []
        for ctry in INTERNATIONAL_COUNTRIES:
            texts.append(ax.text(stock_number[ctry], sr_gain[ctry], ctry))

        stock_number_np = stock_number.to_numpy(dtype="float")
        sr_gain_np = sr_gain.to_numpy(dtype="float")
        b, m = polyfit(stock_number_np, sr_gain_np, 1)
        plt.plot(stock_number_np, b + m * stock_number_np, "-")
        plt.xlabel("Stock Count", fontsize=16)
        plt.ylabel("Sharpe Ratio Gain", fontsize=16)
        plt.grid()
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color="r", lw=0.5))
        plt.subplots_adjust(
            top=0.99, bottom=0.13, right=0.99, left=0.13, hspace=0, wspace=0
        )
        filename = f"./{horizon}d{horizon}p_sr_gain_Direct-Retrain_{weight_type}_ensem5.eps"
        plt.savefig(filename)
        plt.show()
        plt.clf()


def international_sr_table(horizon: int) -> pd.DataFrame:
    """
    Example table for Sharpe Ratio Gains for various countries.
    (Implementation hidden for brevity.)
    """
    return pd.DataFrame()  # placeholder


# ------------------------------------------------------------------------
# NEW DEBUGGING FUNCTION:
# ------------------------------------------------------------------------

def plot_confidence_debug_charts(
    ensem_res: pd.DataFrame,
    dataset_obj,
    top_k: int = 5,
    bottom_k: int = 5,
    ret_column: str = "next_week_ret_0delay",
    out_dir: str = "debug_charts"
) -> None:
    """
    Plot the input charts that triggered the highest and lowest confidence 
    predictions (up_prob) from the neural network, along with the actual 
    5-day (or weekly) return. Each plot includes:
      - The reconstructed chart (as the CNN sees it).
      - Ticker ID, confidence level (up_prob), and actual 5D return.
    
    This helps debug or visualize which patterns the model found most bullish/bearish.
    
    Args:
        ensem_res (pd.DataFrame): 
            DataFrame containing columns ['up_prob', 'MarketCap', ...], 
            with multi-index [Date, StockID], presumably the ensemble's predictions.
        dataset_obj: 
            An instance of EquityDataset or TS1DDataset 
            that has .images and .label_dict for reconstructing images.
        top_k (int): 
            How many top confidence samples to plot.
        bottom_k (int): 
            How many bottom confidence samples to plot.
        ret_column (str):
            The column name in ensem_res that indicates actual future returns 
            (e.g. 'next_week_ret_0delay' or 'next_month_ret_0delay').
        out_dir (str):
            Directory where the debug plots (PNG) will be saved.
    
    Returns:
        None. Saves debug charts as PNG files in out_dir.
    """
    import os
    import shutil
    from PIL import Image
    from src.data.dgp_config import get_dir

    get_dir(out_dir)  # ensure directory

    # 1) Filter to valid rows
    if ret_column not in ensem_res.columns:
        print(f"[WARN] The return column '{ret_column}' does not exist in ensem_res. Falling back to 0.0 for returns.")
        ensem_res[ret_column] = 0.0

    # We only keep rows where up_prob is not null
    sub_df = ensem_res.dropna(subset=["up_prob"]).copy()
    if sub_df.empty:
        print("[INFO] No data to plot: 'up_prob' is empty after dropna.")
        return

    # 2) Sort by up_prob and pick top_k / bottom_k
    sorted_df = sub_df.sort_values(by="up_prob", ascending=False)
    top_samples = sorted_df.head(top_k)
    bot_samples = sorted_df.tail(bottom_k)

    # 3) For each sample, find the corresponding image in dataset_obj
    # We need a quick lookup from (Date, StockID) -> index in dataset_obj
    # We'll build a dictionary that maps (str(Date), str(StockID)) 
    # to the integer index in dataset_obj.images
    date_arr = dataset_obj.label_dict["Date"]
    sid_arr = dataset_obj.label_dict["StockID"]
    idx_map = {}
    for i in range(len(date_arr)):
        key = (str(date_arr[i]), str(sid_arr[i]))
        idx_map[key] = i

    # 4) Helper to rebuild the stored chart from dataset's images
    def rebuild_cnn_image(idx: int) -> Image.Image:
        """
        Rebuild the CNN input image from the byte array in dataset_obj.images[idx].
        This is exactly the image the CNN sees (grayscale).
        """
        # shape = (1, height, width), we can directly form PIL image in 'L' mode
        # recall that dataset_obj.images is already shaped [N,1,H,W].
        raw_img_4d = dataset_obj.images[idx]  # shape => (1,H,W)
        # Squeeze out the channel dimension to get (H,W)
        raw_img_2d = np.squeeze(raw_img_4d)
        # Convert to PIL
        return Image.fromarray(raw_img_2d, mode="L")

    # 5) Utility to produce and save the final debug figure
    def plot_one_debug_chart(row: pd.Series, rank_label: str, i_counter: int) -> None:
        """
        row is a Series with index: [up_prob, next_week_ret_0delay, ...].
        rank_label is either "top" or "bot".
        i_counter is the rank in that group.
        """
        dt_str = str(row.name[0])  # because .name is (Date, StockID)
        sid_str = str(row.name[1])
        up_prob = row["up_prob"]
        act_ret = row.get(ret_column, 0.0)

        # Build a figure with:
        #   a) the CNN's grayscale image
        #   b) some text annotations
        fig, ax = plt.subplots(figsize=(6, 4))
        # Rebuild the CNN image
        key = (dt_str, sid_str)
        if key not in idx_map:
            ax.text(0.5, 0.5, "Image not found", ha="center", va="center", fontsize=12)
            ax.set_title(f"{rank_label} {i_counter}: {sid_str} {dt_str}")
            fig.tight_layout()
            fig_path = os.path.join(out_dir, f"{rank_label}_{i_counter}_{sid_str}_{dt_str}.png")
            plt.savefig(fig_path)
            plt.close(fig)
            return

        image_idx = idx_map[key]
        cnn_img = rebuild_cnn_image(image_idx)

        # Show the image in grayscale
        ax.imshow(cnn_img, cmap="gray", aspect="auto")
        ax.axis("off")

        # Add text: ticker, up_prob, actual ret
        ax.text(
            0.02, 0.92, 
            f"Ticker: {sid_str}\nConf: {up_prob:.3f}\nActual {ret_column}: {act_ret:.3f}",
            ha="left", va="top", transform=ax.transAxes,
            fontsize=10, color="yellow", 
            bbox=dict(facecolor='black', alpha=0.6)
        )
        ax.set_title(f"{rank_label.capitalize()} {i_counter} | Date: {dt_str}", fontsize=10)

        fig.tight_layout()
        fig_path = os.path.join(out_dir, f"{rank_label}_{i_counter}_{sid_str}_{dt_str}.png")
        plt.savefig(fig_path, dpi=120)
        plt.close(fig)

    # 6) Generate the top_k debug plots
    for i, (idx, row) in enumerate(top_samples.iterrows(), start=1):
        plot_one_debug_chart(row, "top", i)

    # 7) Generate the bottom_k debug plots
    for i, (idx, row) in enumerate(bot_samples.iterrows(), start=1):
        plot_one_debug_chart(row, "bot", i)

    print(f"[INFO] Debug charts saved under '{out_dir}'.")


