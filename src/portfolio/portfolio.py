"""
portfolio.py

Implements a portfolio manager that constructs decile portfolios
based on predicted up_prob or raw regression outputs.
"""

import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import utilities as ut
from src.data import equity_data as eqd


class PortfolioManager:
    """
    The PortfolioManager class computes decile portfolios based on a 'signal_df'
    with columns ['up_prob', 'MarketCap', 'next_{freq}_ret_{delay}delay', ...].
    It calculates H-L returns, turnover, and saves to disk.
    """

    def __init__(
        self,
        signal_df: pd.DataFrame,
        freq: str,
        portfolio_dir: str,
        start_year: int = 2001,
        end_year: int = 2019,
        country: str = "USA",
        delay_list=None,
        load_signal: bool = True,
        custom_ret: str = None,
        transaction_cost: bool = False,
    ):
        assert freq in ["week", "month", "quarter"]
        self.freq = freq
        self.portfolio_dir = portfolio_dir
        self.start_year = start_year
        self.end_year = end_year
        self.country = country
        self.delay_list = delay_list or [0]
        self.transaction_cost = transaction_cost
        self.custom_ret = custom_ret

        if load_signal:
            # ensure columns exist
            if "up_prob" not in signal_df.columns:
                raise ValueError("signal_df must have 'up_prob' column.")
            self.signal_df = self.__get_up_prob_with_period_ret(signal_df)

    def __get_up_prob_with_period_ret(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        keep_years = range(self.start_year, self.end_year + 1)
        in_years = signal_df.index.get_level_values("Date").year.isin(keep_years)
        df = signal_df[in_years].copy()
        df = self.__add_period_ret_to_us_res_df_w_delays(df)
        if self.country not in ["future", "new_future"]:
            df["MarketCap"] = df["MarketCap"].abs()
            df = df[~df["MarketCap"].isnull()].copy()
        return df

    def __add_period_ret_to_us_res_df_w_delays(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges next_{freq}_ret_{delay} columns onto the existing signal_df
        from period_ret, but DOES NOT merge 'MarketCap' from period_ret
        to avoid column conflicts.
        """
        period_ret = eqd.get_period_ret(self.freq, country=self.country)
        # Only bring in next_{freq}_ret_{delay} columns (plus custom_ret, if any)
        columns = [f"next_{self.freq}_ret_{dl}delay" for dl in self.delay_list]
        if self.custom_ret:
            columns.append(self.custom_ret)
        df_reset = df.reset_index()
        merged = df_reset.merge(
            period_ret[["Date"] + columns],  # No "MarketCap" here
            on="Date",
            how="left"
        )
        merged.dropna(inplace=True)
        return merged.set_index(["Date", "StockID"])

    def generate_portfolio(self, delay: int = 0, cut: int = 10) -> None:
        """
        Build a decile-based portfolio for either EW or VW weighting, for a chosen delay.
        """
        if delay not in self.delay_list:
            return

        for weight_type in ["ew", "vw"]:
            pf_name = self._get_portfolio_name(weight_type, delay, cut)
            print(f"Calculating {pf_name}")
            pf_ret, turnover = self.__calculate_portfolio_rets(weight_type, cut, delay)
            pf_data_dir = ut.get_dir(op.join(self.portfolio_dir, "pf_data"))
            pf_ret.to_csv(op.join(pf_data_dir, f"pf_data_{pf_name}.csv"))

            summary_df = self.__portfolio_res_summary(pf_ret, turnover, cut)
            smry_path = os.path.join(self.portfolio_dir, f"{pf_name}.csv")
            summary_df.to_csv(smry_path)
            with open(os.path.join(self.portfolio_dir, f"{pf_name}.txt"), "w+") as f:
                summary_df = summary_df.astype(float).round(2)
                f.write(ut.to_latex_w_turnover(summary_df, cut))

    def __calculate_portfolio_rets(self, weight_type: str, cut: int, delay: int):
        assert weight_type in ["ew", "vw"]
        df = self.signal_df.copy()
        ret_name = self.custom_ret if self.custom_ret else f"next_{self.freq}_ret_{delay}delay"

        dates = np.sort(np.unique(df.index.get_level_values("Date")))
        turnover = np.zeros(len(dates) - 1)
        portfolio_ret = pd.DataFrame(index=dates, columns=list(range(cut)))

        prev_to_df = None
        for i, date in enumerate(dates):
            reb_df = df.loc[date].copy()
            low = np.percentile(reb_df["up_prob"], 10)
            high = np.percentile(reb_df["up_prob"], 90)
            if low == high:
                continue
            for j in range(cut):
                decile_df = self.__get_decile_df_with_inv_ret(reb_df, j, cut, weight_type, ret_name)
                portfolio_ret.loc[date, j] = np.sum(decile_df["inv_ret"])

            sell_decile = self.__get_decile_df_with_inv_ret(reb_df, 0, cut, weight_type, ret_name)
            buy_decile = self.__get_decile_df_with_inv_ret(reb_df, cut - 1, cut, weight_type, ret_name)

            sell_decile[["weight", "inv_ret"]] *= -1
            to_df = pd.concat([sell_decile, buy_decile])

            if i > 0:
                merged_idx = np.unique(list(to_df.index) + list(prev_to_df.index))
                tto_df = pd.DataFrame(index=merged_idx)
                tto_df["cur_weight"] = to_df["weight"]
                tto_df[["prev_weight", "ret", "inv_ret"]] = prev_to_df[["weight", ret_name, "inv_ret"]]
                tto_df.fillna(0, inplace=True)
                denom = 1 + np.sum(tto_df["inv_ret"])
                turnover[i - 1] = np.sum(
                    (tto_df["cur_weight"] - tto_df["prev_weight"] * (1 + tto_df["ret"]) / denom).abs()
                ) * 0.5

            prev_to_df = to_df

        portfolio_ret = portfolio_ret.fillna(0)
        portfolio_ret["H-L"] = portfolio_ret[cut - 1] - portfolio_ret[0]
        return portfolio_ret, turnover.mean()

    def __get_decile_df_with_inv_ret(
        self, reb_df: pd.DataFrame, decile_idx: int, cut: int, weight_type: str, ret_name: str
    ) -> pd.DataFrame:
        rebalance_up_prob = reb_df["up_prob"]
        low = np.percentile(rebalance_up_prob, decile_idx * 100.0 / cut)
        high = np.percentile(rebalance_up_prob, (decile_idx + 1) * 100.0 / cut)
        if decile_idx == 0:
            pf_filter = (rebalance_up_prob >= low) & (rebalance_up_prob <= high)
        else:
            pf_filter = (rebalance_up_prob > low) & (rebalance_up_prob <= high)
        decile_df = reb_df[pf_filter].copy()

        if weight_type == "ew":
            decile_df["weight"] = 1.0 / len(decile_df) if len(decile_df) > 0 else 0
        else:
            decile_df["weight"] = decile_df["MarketCap"] / decile_df["MarketCap"].sum()

        decile_df["inv_ret"] = decile_df["weight"] * decile_df[ret_name]
        return decile_df

    def __portfolio_res_summary(self, portfolio_ret: pd.DataFrame, turnover: float, cut: int) -> pd.DataFrame:
        avg = portfolio_ret.mean().to_numpy()
        std = portfolio_ret.std().to_numpy()
        freq_to_period = {"week": 52, "month": 12, "quarter": 4}
        period = freq_to_period[self.freq]
        res = np.zeros((cut + 1, 3))
        res[:, 0] = avg * period
        res[:, 1] = std * np.sqrt(period)
        res[:, 2] = res[:, 0] / res[:, 1]

        summary_df = pd.DataFrame(res, columns=["ret", "std", "SR"])
        summary_df = summary_df.set_index(
            pd.Index(["Low"] + list(range(2, int(cut))) + ["High", "H-L"])
        )
        summary_df.loc["Turnover", "SR"] = turnover / (
            1 if self.freq == "month" else (4 if self.freq == "quarter" else 52 / 12)
        )
        return summary_df

    def _get_portfolio_name(self, weight_type: str, delay: int, cut: int) -> str:
        delay_prefix = "" if delay == 0 else f"{delay}d_delay_"
        cut_suffix = "" if cut == 10 else f"_{cut}cut"
        custom_surfix = f"_{self.custom_ret}" if self.custom_ret else ""
        tc_suffix = "_w_transaction_cost" if self.transaction_cost else ""
        return f"{delay_prefix}{weight_type}{cut_suffix}{custom_surfix}{tc_suffix}"
