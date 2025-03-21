"""
plot_portfolio_performance.py

This module adds functionality to assess performance year by year and to
plot the cumulative returns of the strategy on a weekly frequency.
Each column in the input DataFrame should represent the portfolio returns
(e.g. for each decile) and the index must be datetime (weekly rebalancing dates).

Functions:
    plot_yearly_cumulative_returns: Plots and saves cumulative return curves for a given year.
    plot_all_years_cumulative_returns: Iterates through all years in the DataFrame to plot each year.
    assess_yearly_performance: Computes annual performance metrics for each portfolio.
"""

import os
import math
import pandas as pd
import matplotlib.pyplot as plt


def plot_yearly_cumulative_returns(portfolio_ret: pd.DataFrame, save_dir: str, weight_type: str, year: int) -> None:
    """
    Plot the cumulative returns for each portfolio (decile) for a specified year
    and save the plot.
    
    The cumulative returns for decile 0 (lowest), decile 9 (highest), and H-L (the difference 
    between the highest and lowest deciles) are plotted in distinct colours (red, green, and blue, respectively),
    while the remaining decile curves are drawn in gray.
    
    Parameters:
        portfolio_ret (pd.DataFrame): DataFrame with datetime index and columns representing portfolio returns.
        save_dir (str): Directory where the plot image will be saved.
        weight_type (str): Weighting scheme used ('ew' or 'vw').
        year (int): The year to filter data for plotting.
    """
    # Filter data for the given year
    df_year = portfolio_ret[portfolio_ret.index.year == year]
    if df_year.empty:
        print(f"No data available for year {year}.")
        return

    # Compute cumulative returns: cumulative return = (1 + return).cumprod() - 1
    cum_returns = (1 + df_year).cumprod() - 1

    # Define custom colours for specific deciles and H-L:
    # - Decile 0 (lowest) in red.
    # - Decile 9 (highest) in green.
    # - H-L in blue.
    # All other deciles are plotted in gray.
    custom_colors = {}
    for col in cum_returns.columns:
        col_str = str(col).strip()
        if col_str in ["0", "Low(L)"]:
            custom_colors[col] = 'red'
        elif col_str in ["9", "High(H)"]:
            custom_colors[col] = 'green'
        elif col_str == "H-L":
            custom_colors[col] = 'blue'
        else:
            custom_colors[col] = 'gray'

    plt.figure(figsize=(10, 6))
    for col in cum_returns.columns:
        plt.plot(cum_returns.index, cum_returns[col], label=col, color=custom_colors[col])
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title(f"Cumulative Returns for {year} ({weight_type.upper()})")
    plt.legend()
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f"cumulative_returns_{year}_{weight_type}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved cumulative returns plot for year {year} at {plot_path}")


def plot_all_years_cumulative_returns(portfolio_ret: pd.DataFrame, save_dir: str, weight_type: str) -> None:
    """
    Plot and save cumulative returns for every year found in the portfolio returns DataFrame.

    Parameters:
        portfolio_ret (pd.DataFrame): DataFrame with datetime index and portfolio return columns.
        save_dir (str): Directory where the plots will be saved.
        weight_type (str): Weighting scheme used ('ew' or 'vw').
    """
    years = sorted(portfolio_ret.index.year.unique())
    for year in years:
        plot_yearly_cumulative_returns(portfolio_ret, save_dir, weight_type, year)


def assess_yearly_performance(portfolio_ret: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.DataFrame:
    """
    Assess the performance of each portfolio (e.g. decile) on a yearly basis.
    Computes cumulative return, annualized average return, volatility, and Sharpe ratio.
    Assumes portfolio_ret contains weekly returns (as decimals).

    Parameters:
        portfolio_ret (pd.DataFrame): DataFrame with datetime index and portfolio return columns.
        risk_free_rate (float): Annual risk-free rate (default 0).

    Returns:
        pd.DataFrame: A multi-index DataFrame (Year, Portfolio) with performance metrics.
    """
    results = {}
    # Assume approximately 52 weeks per year
    annual_weeks = 52
    for year in sorted(portfolio_ret.index.year.unique()):
        df_year = portfolio_ret[portfolio_ret.index.year == year]
        if df_year.empty:
            continue
        # Cumulative return over the year (product over weeks minus 1)
        cumulative_return = (1 + df_year).prod() - 1
        # Annualized average return (mean weekly return scaled up)
        avg_return = df_year.mean() * annual_weeks
        # Annualized volatility (std scaled by sqrt(52))
        vol = df_year.std() * math.sqrt(annual_weeks)
        sharpe = (avg_return - risk_free_rate) / (vol + 1e-9)
        performance = pd.DataFrame({
            'Cumulative Return': cumulative_return,
            'Annualized Average Return': avg_return,
            'Annualized Volatility': vol,
            'Sharpe Ratio': sharpe
        })
        results[year] = performance
    if not results:
        print("No performance data available.")
        return pd.DataFrame()
    performance_df = pd.concat(results, axis=0)
    performance_df.index.names = ['Year', 'Portfolio']
    return performance_df
