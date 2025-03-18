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
    """
    # Implementation hidden for brevity: references load_international_portfolio_returns, etc.
    # ...
    return pd.DataFrame()  # placeholder

