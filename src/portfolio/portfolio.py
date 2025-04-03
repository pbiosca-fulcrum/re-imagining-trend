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

    We also now filter to the top 50% by MarketCap (for each Date) before doing any splits.
    This is done in the _get_up_prob_with_period_ret method.

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

        # If none provided, default to 10%
        if tail_percent_list is None or len(tail_percent_list) == 0:
            self.tail_percent_list = [0.10]
        else:
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
        # Keep period_ret from start_year-1 onward, as it might be needed for the first year
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
        Merge up_prob with next_{freq}_ret_Xdelay columns, then filter to the top 50% by MarketCap (each Date).
        """
        merged_df = self._add_period_ret_to_us_res_df_w_delays(signal_df)

        # basic checks
        merged_df["MarketCap"] = merged_df["MarketCap"].abs()
        merged_df = merged_df[~merged_df["MarketCap"].isnull() & (merged_df["MarketCap"] > 0)]

        # Now filter to top 50% by MarketCap for each Date
        # We do this cross-sectionally, meaning for each date, we only keep the top half (descending MarketCap)
        # Steps:
        #   1) reset the index
        #   2) groupby("Date"), sort by MarketCap descending
        #   3) keep the top half
        #   4) re-set index
        df_reset = merged_df.reset_index()
        def keep_top_half(group):
            group = group.sort_values("MarketCap", ascending=False)
            half_n = len(group) // 2  # integer division
            # return group.iloc[:half_n]
            return group

        filtered = (
            df_reset.groupby("Date", group_keys=False)
            .apply(keep_top_half)
            .reset_index(drop=True)
        )
        final_df = filtered.set_index(["Date", "StockID"])

        return final_df

    def _calculate_portfolio_rets_for_tail(
        self,
        df: pd.DataFrame,
        weight_type: str,
        tail_percent: float,
        delay: int
    ) -> pd.DataFrame:
        """
        For a single tail_percent and single delay, compute daily returns for:
            - Low_{tp}, High_{tp}, H-L_{tp}.
        Returns a DataFrame with those 3 columns over time, plus the average turnover.
        """
        if self.custom_ret:
            ret_name = self.custom_ret
        else:
            ret_name = (
                self.no_delay_ret_name if delay == 0 else f"next_{self.freq}_ret_{delay}delay"
            )

        dates = np.sort(df.index.get_level_values("Date").unique())
        low_col = []
        high_col = []
        turnover = np.zeros(len(dates) - 1)
        prev_to_df = None

        low_q = tail_percent * 100.0
        high_q = 100.0 - (tail_percent * 100.0)

        def pick_bottom_tail(data_df: pd.DataFrame, q: float) -> pd.DataFrame:
            up_prob_series = data_df["up_prob"]
            cutoff = np.percentile(up_prob_series, q)
            return data_df[up_prob_series <= cutoff].copy()

        def pick_top_tail(data_df: pd.DataFrame, q: float) -> pd.DataFrame:
            up_prob_series = data_df["up_prob"]
            cutoff = np.percentile(up_prob_series, q)
            return data_df[up_prob_series >= cutoff].copy()

        def compute_weighted_returns(subset_df: pd.DataFrame) -> float:
            if subset_df.empty:
                return 0.0
            if weight_type == "ew":
                w = 1.0 / len(subset_df)
                return (subset_df[ret_name] * w).sum()
            else:
                total_value = subset_df["MarketCap"].sum()
                if total_value == 0:
                    return 0.0
                subset_df["weight"] = subset_df["MarketCap"] / total_value
                return (subset_df[ret_name] * subset_df["weight"]).sum()

        for i, d in enumerate(dates):
            daily_df = df.loc[d]
            if daily_df.empty:
                low_col.append(0.0)
                high_col.append(0.0)
                continue

            bottom_df = pick_bottom_tail(daily_df, low_q)
            top_df = pick_top_tail(daily_df, high_q)

            bottom_ret = compute_weighted_returns(bottom_df)
            top_ret = compute_weighted_returns(top_df)
            low_col.append(bottom_ret)
            high_col.append(top_ret)

            # turnover logic
            combined_now = pd.concat([bottom_df, top_df])
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
            combined_now = pd.concat([bottom_df, top_df])
            combined_now = combined_now.groupby(combined_now.index).sum()  # unify duplicates
            if i > 0 and prev_to_df is not None:
                all_idx = np.unique(list(combined_now.index) + list(prev_to_df.index))
                tto_df = pd.DataFrame(index=all_idx)
                tto_df["cur_weight"] = combined_now["weight"]
                tto_df[["prev_weight", "ret", "inv_ret"]] = prev_to_df[["weight", ret_name, "inv_ret"]]
                tto_df.fillna(0, inplace=True)

                denom = 1.0 + tto_df["inv_ret"].sum()
                this_turnover = (tto_df["cur_weight"] - tto_df["prev_weight"]*(1 + tto_df["ret"])/denom).abs().sum()
                turnover[i - 1] = 0.5 * this_turnover

            combined_now["inv_ret"] = combined_now["weight"] * combined_now[ret_name]
            prev_to_df = combined_now

        daily_ret_df = pd.DataFrame({
            f"Low_{int(tail_percent*100)}%": low_col,
            f"High_{int(tail_percent*100)}%": high_col
        }, index=dates)
        daily_ret_df[f"H-L_{int(tail_percent*100)}%"] = daily_ret_df[f"High_{int(tail_percent*100)}%"] - daily_ret_df[f"Low_{int(tail_percent*100)}%"]
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
        Also track turnover in a dict { tail_percent: avg_turnover }.
        """
        if self.signal_df is None or self.signal_df.empty:
            raise ValueError("signal_df is empty or None. No data available.")

        df = self.signal_df.copy()
        dates = np.sort(df.index.get_level_values("Date").unique())
        if len(dates) == 0:
            raise ValueError("No valid Dates in the final data after merges/filtering.")

        big_result = pd.DataFrame(index=dates)
        turnover_dict = {}
        for p in self.tail_percent_list:
            daily_ret_df, avg_turn = self._calculate_portfolio_rets_for_tail(df, weight_type, p, delay)
            big_result = big_result.join(daily_ret_df, how="outer")
            turnover_dict[p] = avg_turn

        big_result.fillna(0.0, inplace=True)
        return big_result, turnover_dict

    @staticmethod
    def _ret_to_cum_log_ret(series: pd.Series) -> pd.Series:
        """ Convert daily returns into cumulative log-returns """
        return np.log1p(series).cumsum()

    def make_portfolio_plot(self, portfolio_ret: pd.DataFrame, weight_type: str, plot_title: str, save_path: str) -> None:
        """
        Plot cumulative log-returns for each column in portfolio_ret.
        """
        cr_df = portfolio_ret.copy()
        for col in cr_df.columns:
            cr_df[col] = self._ret_to_cum_log_ret(cr_df[col])

        if len(cr_df) > 0:
            first_date = cr_df.index[0]
            prev_year = pd.to_datetime(first_date).year - 1
            prev_day = pd.to_datetime(f"{prev_year}-12-31")
            cr_df.loc[prev_day] = [0.0]*len(cr_df.columns)
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
        return df_summary.round(3)

    def generate_portfolio(self, delay: int = 0, cut=None) -> None:
        """
        Builds the portfolio returns using the chosen tail_percent_list, then saves them
        along with a summary and a plot.

        One CSV with all columns:
          Low_1%, High_1%, H-L_1%, Low_5%, High_5%, H-L_5%, ...
        plus a summary CSV, plus a chart.
        """
        for weight_type in ["ew", "vw"]:
            pf_name = self._get_portfolio_name(weight_type, delay)
            print(f"Calculating portfolio named '{pf_name}' ...")

            portfolio_ret, turnover_dict = self.calculate_portfolio_rets(weight_type=weight_type, delay=delay)
            data_dir = ut.get_dir(op.join(self.portfolio_dir, "pf_data"))
            pf_data_path = op.join(data_dir, f"pf_data_{pf_name}.csv")
            portfolio_ret.to_csv(pf_data_path)

            # Summaries
            summary_df = self.portfolio_res_summary(portfolio_ret)

            # Turnover
            for p in self.tail_percent_list:
                row_name = f"Turnover_{int(p*100)}%"
                summary_df.loc[row_name, ["ret","std","SR"]] = [np.nan, np.nan, turnover_dict[p]]

            smry_path = os.path.join(self.portfolio_dir, f"{pf_name}.csv")
            summary_df.to_csv(smry_path)

            txt_path = os.path.join(self.portfolio_dir, f"{pf_name}.txt")
            with open(txt_path, "w+") as f:
                f.write(summary_df.to_string())

            print(f"[INFO] Portfolio '{pf_name}' results saved to:\n  - {pf_data_path}\n  - {smry_path}")

            # Plot
            plots_dir = ut.get_dir(op.join(self.portfolio_dir, "plots"))
            combined_plot_path = os.path.join(plots_dir, f"combined_cumulative_returns_{pf_name}.png")

            self.make_portfolio_plot(
                portfolio_ret=portfolio_ret,
                weight_type=weight_type,
                plot_title=f"Cumulative Returns for {pf_name} ({weight_type.upper()})",
                save_path=combined_plot_path
            )
            print(f"[INFO] Combined cumulative returns plot saved at {combined_plot_path}")

    def _get_portfolio_name(self, weight_type: str, delay: int) -> str:
        """
        Build a name based on tail_percent_list, weight type, delay, etc.
        """
        tail_str = "_".join([f"{int(p*100)}p" for p in self.tail_percent_list])
        delay_prefix = "" if delay == 0 else f"{delay}d_delay_"
        custom_ret_suffix = f"_{self.custom_ret}" if self.custom_ret else ""
        tc_suffix = "_w_transaction_cost" if self.transaction_cost else ""
        pf_name = f"{delay_prefix}{weight_type.lower()}_{tail_str}{custom_ret_suffix}{tc_suffix}"
        return pf_name

    def load_portfolio_ret(self, weight_type: str, delay: int = 0) -> pd.DataFrame:
        pf_name = self._get_portfolio_name(weight_type, delay)
        data_dir = op.join(self.portfolio_dir, "pf_data")
        pf_path = op.join(data_dir, f"pf_data_{pf_name}.csv")
        if not op.isfile(pf_path):
            raise FileNotFoundError(f"Portfolio returns not found at {pf_path}")
        return pd.read_csv(pf_path, index_col=0, parse_dates=True)

    def load_portfolio_summary(self, weight_type: str, delay: int = 0) -> pd.DataFrame:
        pf_name = self._get_portfolio_name(weight_type, delay)
        smry_path = op.join(self.portfolio_dir, f"{pf_name}.csv")
        if not op.isfile(smry_path):
            raise FileNotFoundError(f"Portfolio summary not found at {smry_path}")
        return pd.read_csv(smry_path, index_col=0)


def main():
    pass

if __name__ == "__main__":
    main()
