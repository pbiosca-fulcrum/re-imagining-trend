# src/portfolio/portfolio.py

import os
import os.path as op
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import utilities as ut
from src.data import equity_data as eqd

class PortfolioManager:
    """
    Manages the construction of decile portfolios based on an 'up_prob' signal
    and the subsequent calculation of portfolio returns.
    """
    def __init__(
        self,
        signal_df: pd.DataFrame,
        freq: str,
        portfolio_dir: str,
        start_year: int = 2001,
        end_year: int = 2019,
        country: str = "USA",
        delay_list: list = None,
        load_signal: bool = True,
        custom_ret: str = None,
        transaction_cost: bool = False
    ) -> None:
        assert freq in ["week", "month", "quarter"], (
            f"freq must be one of 'week','month','quarter'; got {freq}"
        )
        self.freq = freq
        self.portfolio_dir = portfolio_dir
        self.start_year = start_year
        self.end_year = end_year
        self.country = country
        # For the base case, we handle 0 delay by default.
        self.delay_list = [0] if delay_list is None else delay_list
        self.custom_ret = custom_ret
        self.transaction_cost = transaction_cost
        self.no_delay_ret_name = f"next_{freq}_ret"

        # If we are loading the signal, attach period returns to that DataFrame:
        if load_signal:
            # Must contain "up_prob" and "MarketCap"
            if "up_prob" not in signal_df.columns:
                raise ValueError("signal_df must have an 'up_prob' column if load_signal=True.")
            self.signal_df = self.get_up_prob_with_period_ret(signal_df)
            print(f"[DEBUG] signal_df has {len(self.signal_df)} samples after merging with period returns.")
            breakpoint()
        else:
            self.signal_df = None
            print(f"[DEBUG] signal_df is not loaded. No data available for portfolio generation.")
            breakpoint()

    def __add_period_ret_to_us_res_df_w_delays(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        period_ret = eqd.get_period_ret(self.freq, country=self.country)
        
        # Ensure period_ret has a MultiIndex of [Date, StockID]
        if not isinstance(period_ret.index, pd.MultiIndex):
            period_ret = period_ret.set_index(["Date", "StockID"])
        
        # Filter for dates after 2000
        period_ret = period_ret[period_ret.index.get_level_values("Date").year > 2000]
        columns = ["MarketCap"] + [f"next_{self.freq}_ret_{dl}delay" for dl in self.delay_list]
        if self.custom_ret is not None:
            columns.append(self.custom_ret)

        print(f"[DEBUG] Merging signal_df with period_ret columns: {columns}")
        print(f"[DEBUG] signal_df has {len(signal_df)} samples before merging.")
        print(f"[DEBUG] signal_df index: {signal_df.index.names}")
        print(f"[DEBUG] signal_df columns: {signal_df.columns}")
        print(f"[DEBUG] signal_df.head():\n{signal_df.head()}")
        
        print(f"[DEBUG] --------------------------------------")
        
        
        print(f"[DEBUG] period_ret has {len(period_ret)} samples.")
        print(f"[DEBUG] period_ret index: {period_ret.index.names}")
        print(f"[DEBUG] period_ret columns: {period_ret.columns}")
        print(f"[DEBUG] period_ret.head():\n{period_ret.head()}")
        
        
        period_ret = period_ret.rename(columns={"MarketCap": "MC_from_ret"})

        merged_df = signal_df.join(period_ret[["MC_from_ret", "next_week_ret_0delay"]], how="inner")

        # For convenience, define a base 'no_delay_ret_name'
        merged_df[self.no_delay_ret_name] = merged_df[f"next_{self.freq}_ret_0delay"]
        
        print(f"[DEBUG] Merged DataFrame shape: {merged_df.shape}")
        print(f"[DEBUG] Merged DataFrame merged_df.head():\n{merged_df.head()}")
        
        # Finally, drop rows that are still missing any of these columns
        merged_df.dropna(subset=columns, inplace=True)
        merged_df.dropna(subset=[self.no_delay_ret_name], inplace=True)
        return merged_df

    def get_up_prob_with_period_ret(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        filtered_df = signal_df[
            signal_df.index.get_level_values("Date").year.isin(
                range(self.start_year, self.end_year + 1)
            )
        ]
        if filtered_df.empty:
            print(
                f"[DEBUG] After date filtering from year {self.start_year} to {self.end_year}, "
                f"the signal df has {filtered_df.shape[0]} rows (i.e. empty)."
            )
            return filtered_df
        final_df = self.__add_period_ret_to_us_res_df_w_delays(filtered_df)
        if self.country not in ["future", "new_future"]:
            final_df["MarketCap"] = final_df["MarketCap"].abs()
            final_df = final_df[~final_df["MarketCap"].isnull() & (final_df["MarketCap"] > 0)]
        return final_df

    def calculate_portfolio_rets(
        self,
        weight_type: str,
        cut: int = 10,
        delay: int = 0
    ) -> (pd.DataFrame, float):
        """
        For each rebalance date, this method groups stocks into deciles (by up_prob)
        and computes a decile portfolio return as the sum of the weighted returns.
        
        If self.transaction_cost is True, then after computing the decile returns for
        a given date, the method adjusts those returns by subtracting an estimate of trading
        costs. The cost is computed as the weighted average trading cost for the period times
        the turnover observed from the previous rebalance. We use 10 basis points (0.001) if a
        stock’s MarketCap is above the 80th percentile and 20 basis points (0.002) otherwise.
        
        Returns:
            portfolio_ret: DataFrame with dates as index and decile returns as columns.
            avg_turnover: the average turnover across dates.
        """
        # Determine the return column name based on delay.
        ret_name = self.no_delay_ret_name if delay == 0 else f"next_{self.freq}_ret_{delay}delay"
        df = self.signal_df.copy()
        if df is None or df.empty:
            raise ValueError("[ERROR] No signal data available.")

        dates = np.sort(np.unique(df.index.get_level_values("Date")))
        if len(dates) == 0:
            raise ValueError("[ERROR] No valid Dates in signal_df.")

        print(f"Calculating portfolio returns from {pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[-1]).date()}")

        turnover = np.zeros(len(dates) - 1)
        portfolio_ret = pd.DataFrame(index=dates, columns=list(range(cut)))
        prob_ret_corr = []
        prob_ret_pearson_corr = []
        prob_inv_ret_corr = []
        prob_inv_ret_pearson_corr = []
        prev_to_df = None  # For turnover calculation

        def get_decile_df_with_inv_ret(reb_df: pd.DataFrame, decile_idx: int) -> pd.DataFrame:
            up_prob_series = reb_df["up_prob"]
            low_quantile = np.percentile(up_prob_series, decile_idx * 100.0 / cut)
            high_quantile = np.percentile(up_prob_series, (decile_idx + 1) * 100.0 / cut)
            if decile_idx == 0:
                pf_filter = (up_prob_series >= low_quantile) & (up_prob_series <= high_quantile)
            else:
                pf_filter = (up_prob_series > low_quantile) & (up_prob_series <= high_quantile)
            decile_subset = rebalance_df[pf_filter].copy()
            if decile_subset.empty:
                decile_subset["weight"] = 0.0
                decile_subset["inv_ret"] = 0.0
                return decile_subset
            if weight_type == "ew":
                stock_num = len(decile_subset)
                decile_subset["weight"] = 1.0 / float(stock_num)
            else:
                total_value = decile_subset["MarketCap"].sum()
                decile_subset["weight"] = decile_subset["MarketCap"] / total_value
            decile_subset["inv_ret"] = decile_subset["weight"] * decile_subset[ret_name]
            return decile_subset

        # Loop over each rebalance date.
        for i, d in enumerate(dates):
            rebalance_df = df.loc[d].copy()
            corr_spearman = ut.rank_corr(rebalance_df, "up_prob", ret_name, method="spearman")
            corr_pearson = ut.rank_corr(rebalance_df, "up_prob", ret_name, method="pearson")
            prob_ret_corr.append(corr_spearman)
            prob_ret_pearson_corr.append(corr_pearson)
            if rebalance_df.empty:
                portfolio_ret.loc[d] = 0.0
                continue

            for j in range(cut):
                decile_df = get_decile_df_with_inv_ret(rebalance_df, j)
                if decile_df.empty:
                    portfolio_ret.loc[d, j] = 0.0
                    continue
                portfolio_ret.loc[d, j] = decile_df["inv_ret"].sum()

            # Compute turnover from previous period using top and bottom deciles.
            sell_decile = get_decile_df_with_inv_ret(rebalance_df, 0)
            buy_decile = get_decile_df_with_inv_ret(rebalance_df, cut - 1)
            if (not sell_decile.empty) or (not buy_decile.empty):
                buy_sell_decile = pd.concat([sell_decile, buy_decile])
                corr_inv_spearman = ut.rank_corr(buy_sell_decile, "up_prob", "inv_ret", method="spearman")
                corr_inv_pearson = ut.rank_corr(buy_sell_decile, "up_prob", "inv_ret", method="pearson")
            else:
                corr_inv_spearman = np.nan
                corr_inv_pearson = np.nan
            prob_inv_ret_corr.append(corr_inv_spearman)
            prob_inv_ret_pearson_corr.append(corr_inv_pearson)

            # For turnover calculation.
            sell_decile[["weight", "inv_ret"]] = sell_decile[["weight", "inv_ret"]] * (-1)
            to_df = pd.concat([sell_decile, buy_decile]) if not buy_decile.empty else sell_decile
            if i > 0 and prev_to_df is not None:
                all_idx = np.unique(list(to_df.index) + list(prev_to_df.index))
                tto_df = pd.DataFrame(index=all_idx)
                tto_df["cur_weight"] = to_df["weight"]
                tto_df[["prev_weight", "ret", "inv_ret"]] = prev_to_df[["weight", ret_name, "inv_ret"]]
                tto_df.fillna(0, inplace=True)
                denom = 1.0 + tto_df["inv_ret"].sum()
                turnover[i - 1] = (tto_df["cur_weight"] - tto_df["prev_weight"] * (1 + tto_df["ret"]) / denom).abs().sum()
                turnover[i - 1] *= 0.5
            prev_to_df = to_df

            # --- Trading Costs Adjustment ---
            if i > 0 and self.transaction_cost:
                # Compute cost rates for the period from rebalance_df:
                threshold = np.percentile(rebalance_df["MarketCap"], 80)
                cost_rates = np.where(rebalance_df["MarketCap"] >= threshold, 0.001, 0.002)
                if weight_type == "ew":
                    avg_cost_rate_overall = np.mean(cost_rates)
                else:
                    weights = rebalance_df["MarketCap"] / rebalance_df["MarketCap"].sum()
                    avg_cost_rate_overall = np.sum(weights * cost_rates)
                # Adjust the decile returns for date d by subtracting the cost adjustment.
                # (Turnover[i-1] is the turnover from the previous period.)
                portfolio_ret.loc[d] = portfolio_ret.loc[d] - avg_cost_rate_overall * turnover[i - 1]

        portfolio_ret = portfolio_ret.fillna(0.0)
        portfolio_ret["H-L"] = portfolio_ret[cut - 1] - portfolio_ret[0]

        print(f"[DEBUG] Spearman Corr (Prob vs. StockReturn) = {np.nanmean(prob_ret_corr):.4f}")
        print(f"[DEBUG] Pearson Corr (Prob vs. StockReturn) = {np.nanmean(prob_ret_pearson_corr):.4f}")
        print(f"[DEBUG] Spearman Corr (Prob vs. inv_ret in top/bottom decile) = {np.nanmean(prob_inv_ret_corr):.4f}")
        print(f"[DEBUG] Pearson Corr (Prob vs. inv_ret in top/bottom decile) = {np.nanmean(prob_inv_ret_pearson_corr):.4f}")

        avg_turnover = np.mean(turnover)
        return portfolio_ret, avg_turnover

    @staticmethod
    def _ret_to_cum_log_ret(rets: pd.Series) -> pd.Series:
        log_rets = np.log(rets.astype(float) + 1.0)
        return log_rets.cumsum()

    def make_portfolio_plot(
        self,
        portfolio_ret: pd.DataFrame,
        cut: int,
        weight_type: str,
        save_path: str,
        plot_title: str
    ) -> None:
        # [Plotting code remains unchanged...]
        pass

    def portfolio_res_summary(
        self,
        portfolio_ret: pd.DataFrame,
        turnover: float,
        cut: int = 10
    ) -> pd.DataFrame:
        # [Summary code remains unchanged...]
        pass

    def generate_portfolio(self, cut: int = 10, delay: int = 0) -> None:
        if self.signal_df is None or self.signal_df.empty:
            raise ValueError("signal_df is empty or None. No data available for portfolio generation.")
        assert delay in self.delay_list, f"Delay {delay} is not in {self.delay_list}."

        for weight_type in ["ew", "vw"]:
            pf_name = self.get_portfolio_name(weight_type, delay, cut)
            print(f"Calculating portfolio named '{pf_name}' ...")
            portfolio_ret, turnover = self.calculate_portfolio_rets(
                weight_type=weight_type,
                cut=cut,
                delay=delay
            )
            # [Saving files and plots... Code unchanged]
            # ...
            print(f"[INFO] Portfolio '{pf_name}' results saved.")

    def get_portfolio_name(self, weight_type: str, delay: int, cut: int) -> str:
        assert weight_type.lower() in ["ew", "vw"]
        delay_prefix = "" if delay == 0 else f"{delay}d_delay_"
        cut_suffix = "" if cut == 10 else f"_{cut}cut"
        custom_ret_suffix = f"_{self.custom_ret}" if self.custom_ret else ""
        tc_suffix = "_w_transaction_cost" if self.transaction_cost else ""
        pf_name = f"{delay_prefix}{weight_type.lower()}{cut_suffix}{custom_ret_suffix}{tc_suffix}"
        return pf_name

    def load_portfolio_ret(self, weight_type: str, cut: int = 10, delay: int = 0) -> pd.DataFrame:
        # [Loading code remains unchanged...]
        pass

    def load_portfolio_summary(self, weight_type: str, cut: int = 10, delay: int = 0) -> pd.DataFrame:
        # [Loading code remains unchanged...]
        pass

def main():
    """Example usage (not typically used this way)."""
    pass

if __name__ == "__main__":
    main()
