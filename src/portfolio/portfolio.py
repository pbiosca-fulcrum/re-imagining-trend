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
    Manages the construction of portfolios based on an 'up_prob' signal
    and the subsequent calculation of portfolio returns.

    Key modification:
      Instead of creating 10 deciles (0..9), we only create two groups per chosen tail percentage:
        - Bottom X% (call it 'Low_{X%}')
        - Top X% (call it 'High_{X%}')
      Then we compute 'H-L_{X%}' as the difference in returns.

    You can pass a list of tail_percent values (e.g. [0.01, 0.05, 0.10]) to compare multiple tail sizes
    in a single portfolio.

    Example usage:
        pm = PortfolioManager(signal_df=..., freq='week', portfolio_dir='...', tail_percent_list=[0.01, 0.05])
        pm.generate_portfolio()
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
        transaction_cost: bool = False,
        tail_percent_list: list = [0.01, 0.05, 0.10]
    ) -> None:
        assert freq in ["week", "month", "quarter"], (
            f"freq must be one of 'week','month','quarter'; got {freq}"
        )
        self.freq = freq
        self.portfolio_dir = portfolio_dir
        self.start_year = start_year
        self.end_year = end_year
        self.country = country
        self.delay_list = [0] if delay_list is None else delay_list
        self.custom_ret = custom_ret
        self.transaction_cost = transaction_cost
        self.no_delay_ret_name = f"next_{freq}_ret"

        # By default, we look at the top/bottom 10% if user hasn't specified otherwise
        if tail_percent_list is None or len(tail_percent_list) == 0:
            self.tail_percent_list = [0.10]
        else:
            # Ensure we have a list of floats (e.g. [0.01, 0.05, 0.10])
            self.tail_percent_list = sorted(tail_percent_list)

        if load_signal:
            if "up_prob" not in signal_df.columns:
                raise ValueError("signal_df must have an 'up_prob' column if load_signal=True.")
            self.signal_df = self._get_up_prob_with_period_ret(signal_df)
        else:
            self.signal_df = None

    def _add_period_ret_to_us_res_df_w_delays(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper that merges the main DataFrame with the period returns for each delay.
        """
        period_ret = eqd.get_period_ret(self.freq, country=self.country)

        # Ensure multi-index
        if not isinstance(signal_df.index, pd.MultiIndex):
            signal_df = signal_df.set_index(["Date", "StockID"])
        if not isinstance(period_ret.index, pd.MultiIndex):
            period_ret = period_ret.set_index(["Date", "StockID"])

        signal_df = signal_df.copy()
        signal_df.index = signal_df.index.set_levels([
            pd.to_datetime(signal_df.index.levels[0]).normalize(),
            signal_df.index.levels[1].astype(str)
        ], level=[0, 1])

        period_ret = period_ret.copy()
        period_ret.index = period_ret.index.set_levels([
            pd.to_datetime(period_ret.index.levels[0]).normalize(),
            period_ret.index.levels[1].astype(str)
        ], level=[0, 1])

        # Filter the date range for the user-specified start/end
        signal_df = signal_df[
            signal_df.index.get_level_values("Date").year.isin(range(self.start_year, self.end_year + 1))
        ]
        # Typically we keep period_ret from 2001 onward, but you can adjust if needed
        period_ret = period_ret[period_ret.index.get_level_values("Date").year >= (self.start_year - 1)]

        columns_needed = ["MarketCap"] + [f"next_{self.freq}_ret_{dl}delay" for dl in self.delay_list]
        if self.custom_ret is not None:
            columns_needed.append(self.custom_ret)

        # rename MarketCap in period_ret to avoid overwriting
        period_ret = period_ret.rename(columns={"MarketCap": "MC_from_ret"})

        merged_df = signal_df.join(
            period_ret[["MC_from_ret", f"next_{self.freq}_ret_0delay"]],
            how="inner"
        )

        # The "no delay" return column
        merged_df[self.no_delay_ret_name] = merged_df[f"next_{self.freq}_ret_0delay"]

        # Drop rows missing required columns
        merged_df.dropna(subset=columns_needed, inplace=True)
        merged_df.dropna(subset=[self.no_delay_ret_name], inplace=True)

        return merged_df

    def _get_up_prob_with_period_ret(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge up_prob with next_{freq}_ret_Xdelay columns.
        """
        merged_df = self._add_period_ret_to_us_res_df_w_delays(signal_df)

        if self.country not in ["future", "new_future"]:
            merged_df["MarketCap"] = merged_df["MarketCap"].abs()
            merged_df = merged_df[~merged_df["MarketCap"].isnull() & (merged_df["MarketCap"] > 0)]

        return merged_df

    def _calculate_portfolio_rets_for_tail(
        self,
        df: pd.DataFrame,
        weight_type: str,
        tail_percent: float,
        delay: int
    ) -> pd.Series:
        """
        For a single tail_percent (e.g. 0.05 for 5%) and a single delay,
        compute the daily returns for:
            - 'Low_{tp}' = bottom tail_percent
            - 'High_{tp}' = top tail_percent
        We'll store them in a single pandas Series with index = the dates,
        but each row has sum of returns for Low_{tp}, High_{tp}, and the difference.

        Actually, we'll just return a DataFrame with columns: [Low_{tp}, High_{tp}, H-L_{tp}].
        Then we can merge them across different tail percents.
        """
        # Identify the return column
        if self.custom_ret:
            ret_name = self.custom_ret
        else:
            ret_name = (
                self.no_delay_ret_name if delay == 0 else f"next_{self.freq}_ret_{delay}delay"
            )

        # Unique dates
        dates = np.sort(df.index.get_level_values("Date").unique())
        # Prepare arrays to store daily returns
        low_col = []
        high_col = []

        # We'll track turnover for these two sub-portfolios combined
        turnover = np.zeros(len(dates) - 1)
        prev_to_df = None

        # Precompute quantile levels for easier reading
        low_q = tail_percent * 100.0
        high_q = 100.0 - (tail_percent * 100.0)

        def pick_bottom_tail(data_df: pd.DataFrame, q: float) -> pd.DataFrame:
            """
            Return a subset containing those stocks at or below q-th percentile of up_prob.
            """
            up_prob_series = data_df["up_prob"]
            cutoff = np.percentile(up_prob_series, q)
            sub = data_df[up_prob_series <= cutoff].copy()
            return sub

        def pick_top_tail(data_df: pd.DataFrame, q: float) -> pd.DataFrame:
            """
            Return a subset containing those stocks at or above q-th percentile of up_prob.
            """
            up_prob_series = data_df["up_prob"]
            cutoff = np.percentile(up_prob_series, q)
            sub = data_df[up_prob_series >= cutoff].copy()
            return sub

        def compute_weighted_returns(subset_df: pd.DataFrame, ret_col: str) -> float:
            """
            Weighted returns for the subset. (EW or VW)
            """
            if subset_df.empty:
                return 0.0
            if weight_type == "ew":
                w = 1.0 / len(subset_df)
                return (subset_df[ret_col] * w).sum()
            else:
                total_value = subset_df["MarketCap"].sum()
                if total_value == 0:
                    return 0.0
                subset_df["weight"] = subset_df["MarketCap"] / total_value
                return (subset_df[ret_col] * subset_df["weight"]).sum()

        all_dates_df = []
        for i, d in enumerate(dates):
            daily_df = df.loc[d]
            if daily_df.empty:
                low_col.append(0.0)
                high_col.append(0.0)
                continue

            bottom_df = pick_bottom_tail(daily_df, low_q)
            top_df = pick_top_tail(daily_df, high_q)

            bottom_ret = compute_weighted_returns(bottom_df, ret_name)
            top_ret = compute_weighted_returns(top_df, ret_name)

            # store them for that day
            low_col.append(bottom_ret)
            high_col.append(top_ret)

            # turnover calculation
            # For day i, we create a combined "to_df" from bottom+top
            # so we can measure how the new weights differ from the old day
            # (We skip it for i==0 because there's no previous day)
            combined_now = pd.concat([bottom_df, top_df])
            combined_now = combined_now.copy()

            if weight_type == "ew":
                if not bottom_df.empty:
                    bottom_df["weight"] = 1.0 / len(bottom_df)
                if not top_df.empty:
                    top_df["weight"] = 1.0 / len(top_df)
            else:
                if not top_df.empty:
                    top_df["weight"] = top_df["MarketCap"] / top_df["MarketCap"].sum()
                if not bottom_df.empty:
                    bottom_df["weight"] = bottom_df["MarketCap"] / bottom_df["MarketCap"].sum()

            # The 'inv_ret' is just the weighted ret
            combined_now = pd.concat([bottom_df, top_df])
            # ensure no duplicates
            combined_now = combined_now.groupby(combined_now.index).sum()

            if i > 0 and prev_to_df is not None:
                # unify index
                all_idx = np.unique(list(combined_now.index) + list(prev_to_df.index))
                tto_df = pd.DataFrame(index=all_idx)
                tto_df["cur_weight"] = combined_now["weight"]
                tto_df[["prev_weight", "ret", "inv_ret"]] = prev_to_df[["weight", ret_name, "inv_ret"]]
                tto_df.fillna(0, inplace=True)

                denom = 1.0 + tto_df["inv_ret"].sum()
                this_turnover = (tto_df["cur_weight"] - tto_df["prev_weight"] * (1 + tto_df["ret"]) / denom).abs().sum()
                turnover[i - 1] = 0.5 * this_turnover

            # For next iteration
            combined_now["inv_ret"] = combined_now["weight"] * combined_now[ret_name]
            prev_to_df = combined_now

        # Build result DataFrame for these columns
        daily_ret_df = pd.DataFrame({
            f"Low_{int(tail_percent*100)}%": low_col,
            f"High_{int(tail_percent*100)}%": high_col
        }, index=dates)
        daily_ret_df[f"H-L_{int(tail_percent*100)}%"] = daily_ret_df[f"High_{int(tail_percent*100)}%"] - daily_ret_df[f"Low_{int(tail_percent*100)}%"]

        # average turnover
        avg_turn = np.mean(turnover)
        return daily_ret_df, avg_turn

    def calculate_portfolio_rets(
        self,
        weight_type: str,
        delay: int = 0
    ) -> (pd.DataFrame, dict):
        """
        For each specified tail_percent, create 3 columns in the final DataFrame:
          Low_{p}, High_{p}, H-L_{p}
        We also track turnover in a dictionary for each tail_percent.

        Returns:
          - A DataFrame with columns for each tail-percent set of Low, High, H-L
          - A dict with { tail_percent: average turnover } for each
        """
        if self.signal_df is None or self.signal_df.empty:
            raise ValueError("signal_df is empty or None. No data available for portfolio generation.")

        if self.custom_ret:
            ret_name = self.custom_ret
        else:
            ret_name = (
                self.no_delay_ret_name if delay == 0 else f"next_{self.freq}_ret_{delay}delay"
            )

        df = self.signal_df.copy()
        dates = np.sort(df.index.get_level_values("Date").unique())
        if len(dates) == 0:
            raise ValueError("No valid Dates in the final data after merges/filtering.")

        # We'll compute daily returns for each tail_percent and merge them
        big_result = pd.DataFrame(index=dates)
        turnover_dict = {}
        for p in self.tail_percent_list:
            daily_ret_df, avg_turn = self._calculate_portfolio_rets_for_tail(df, weight_type, p, delay)
            # Merge
            big_result = big_result.join(daily_ret_df, how="outer")
            turnover_dict[p] = avg_turn

        return big_result.fillna(0.0), turnover_dict

    @staticmethod
    def _ret_to_cum_log_ret(series: pd.Series) -> pd.Series:
        """ Helper to convert daily returns into cumulative log returns """
        return np.log1p(series).cumsum()

    def make_portfolio_plot(
        self,
        portfolio_ret: pd.DataFrame,
        weight_type: str,
        plot_title: str,
        save_path: str
    ) -> None:
        """
        Plot the cumulative returns for each column in `portfolio_ret`.
        Typically, these columns will be Low_{p}, High_{p}, H-L_{p} for each p in tail_percent_list.
        """
        # Build a log-return DF
        cr_df = portfolio_ret.copy()
        for col in cr_df.columns:
            cr_df[col] = self._ret_to_cum_log_ret(cr_df[col])

        # Insert zero line at the end of previous year if you want a baseline
        if len(cr_df) > 0:
            first_date = cr_df.index[0]
            prev_year = pd.to_datetime(first_date).year - 1
            prev_day = pd.to_datetime(f"{prev_year}-12-31")
            cr_df.loc[prev_day] = [0.0] * len(cr_df.columns)
            cr_df.sort_index(inplace=True)

        plt.figure(figsize=(10,6))
        for col in cr_df.columns:
            plt.plot(cr_df.index, cr_df[col], label=col)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Log Return")
        plt.title(plot_title)
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def portfolio_res_summary(self, portfolio_ret: pd.DataFrame) -> pd.DataFrame:
        """
        For each column in `portfolio_ret`, compute annualized return, annualized std, and Sharpe ratio.
        Returns a DataFrame with rows = columns in portfolio_ret, columns = [Ret, Std, SR].
        We omit turnover from this table because we have multiple tail percents at once.
        """
        if self.freq == "week":
            period = 52
        elif self.freq == "month":
            period = 12
        else:
            period = 4

        avg = portfolio_ret.mean(axis=0) * period
        std = portfolio_ret.std(axis=0) * math.sqrt(period)
        sr = avg / std.replace(0, np.nan)

        df_summary = pd.DataFrame({
            "ret": avg,
            "std": std,
            "SR": sr
        })
        return df_summary

    def generate_portfolio(self, delay: int = 0, cut = None) -> None:
        """
        Build the portfolio returns using the chosen tail_percent_list, then save them
        along with a summary and a plot.

        This yields a single CSV with all columns: e.g.,
          Low_1%, High_1%, H-L_1%, Low_5%, High_5%, H-L_5%, Low_10%, High_10%, H-L_10%
        plus an annual performance CSV and a chart.
        """
        for weight_type in ["ew", "vw"]:
            pf_name = self.get_portfolio_name(weight_type, delay)

            print(f"Calculating portfolio named '{pf_name}' ...")
            portfolio_ret, turnover_dict = self.calculate_portfolio_rets(
                weight_type=weight_type,
                delay=delay
            )
            data_dir = ut.get_dir(op.join(self.portfolio_dir, "pf_data"))
            pf_data_path = op.join(data_dir, f"pf_data_{pf_name}.csv")
            portfolio_ret.to_csv(pf_data_path)

            # Summaries
            summary_df = self.portfolio_res_summary(portfolio_ret).round(3)

            # Append turnover info for each tail_percent
            # We'll add each turnover to a row labeled "Turnover_{X%}"
            for p in self.tail_percent_list:
                row_name = f"Turnover_{int(p*100)}%"
                summary_df.loc[row_name, ["ret","std","SR"]] = [np.nan, np.nan, turnover_dict[p]]

            smry_path = os.path.join(self.portfolio_dir, f"{pf_name}.csv")
            summary_df.to_csv(smry_path)

            txt_path = os.path.join(self.portfolio_dir, f"{pf_name}.txt")
            with open(txt_path, "w+") as file:
                file.write(summary_df.to_string())

            print(f"[INFO] Portfolio '{pf_name}' results saved to:\n  - {pf_data_path}\n  - {smry_path}")

            # Plot
            from src.portfolio import plot_portfolio_performance as ppp
            plots_dir = ut.get_dir(op.join(self.portfolio_dir, "plots"))
            combined_plot_path = os.path.join(plots_dir, f"combined_cumulative_returns_{pf_name}.png")

            self.make_portfolio_plot(
                portfolio_ret=portfolio_ret,
                weight_type=weight_type,
                plot_title=f"Cumulative Returns for {pf_name} ({weight_type.upper()})",
                save_path=combined_plot_path
            )
            print(f"[INFO] Combined cumulative returns plot saved at {combined_plot_path}")

            # Optional: you could also break out a year-by-year plot or do shading, but we keep it simpler here.

    def get_portfolio_name(self, weight_type: str, delay: int) -> str:
        """
        Build a name based on the tail_percent_list, weight type, delay, etc.
        """
        tail_str = "_".join([f"{int(p*100)}p" for p in self.tail_percent_list])
        delay_prefix = "" if delay == 0 else f"{delay}d_delay_"
        custom_ret_suffix = f"_{self.custom_ret}" if self.custom_ret else ""
        tc_suffix = "_w_transaction_cost" if self.transaction_cost else ""
        pf_name = f"{delay_prefix}{weight_type.lower()}_{tail_str}{custom_ret_suffix}{tc_suffix}"
        return pf_name

    def load_portfolio_ret(self, weight_type: str, delay: int = 0) -> pd.DataFrame:
        pf_name = self.get_portfolio_name(weight_type, delay)
        data_dir = op.join(self.portfolio_dir, "pf_data")
        pf_path = op.join(data_dir, f"pf_data_{pf_name}.csv")
        if not op.isfile(pf_path):
            raise FileNotFoundError(f"Portfolio returns not found at {pf_path}")
        df = pd.read_csv(pf_path, index_col=0, parse_dates=True)
        return df

    def load_portfolio_summary(self, weight_type: str, delay: int = 0) -> pd.DataFrame:
        pf_name = self.get_portfolio_name(weight_type, delay)
        smry_path = op.join(self.portfolio_dir, f"{pf_name}.csv")
        if not op.isfile(smry_path):
            raise FileNotFoundError(f"Portfolio summary not found at {smry_path}")
        df = pd.read_csv(smry_path, index_col=0)
        return df

def main():
    pass

if __name__ == "__main__":
    main()
