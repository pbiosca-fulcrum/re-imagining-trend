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

    Key modification for transaction costs:
      We replicate the Bryan Kelly approach where:
         (1) The short side is given negative weights (so its returns are negative).
         (2) For each side (top or bottom), we subtract transaction_fee * 2 * weight 
             from 'inv_ret' if it is the long/ top basket, 
             and also subtract transaction_fee * 2 * weight if it is the short/ bottom basket. 
             However, because the bottom basket has negative weights, it effectively adds a cost.
    
    We also keep the possibility to filter by daily volume. See the `_get_up_prob_with_period_ret`
    method, which uses 'DollarVolume' or a top-60% filter for liquidity constraints.

    Instead of creating multiple deciles, we only create two groups per chosen tail percentage:
        - Bottom X% (short side)
        - Top X% (long side)
    Then we compute H-L_{X%} as the difference in returns from top minus bottom.

    By default, the code uses self.transaction_fee = 0.0005 if you set `self.transaction_cost=True`,
    but you can adjust it as needed.
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
        transaction_cost: bool = True,
        tail_percent_list: list = [0.0075, 0.05, 0.10],
        transaction_fee: float = 0.001  # <--- New default transaction fee if transaction_cost=True
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
        self.transaction_fee = transaction_fee  # <--- store the fee
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

    def plot_weekly_confidence_tails(self, signal_df: pd.DataFrame, save_path: str) -> None:
        """
        For each week in signal_df (indexed by ['Date','StockID'] or a Date column),
        compute the average 'up_prob' for the top 10% and the bottom 10% of stocks.
        Plots both time series in a single chart.
        """

        # Make sure the DataFrame has a Date in the index (or as a column)
        if not isinstance(signal_df.index, pd.MultiIndex) or 'Date' not in signal_df.index.names:
            if 'Date' in signal_df.columns:
                signal_df = signal_df.set_index('Date')
            else:
                raise ValueError("signal_df must have 'Date' in the index or a 'Date' column.")

        results = []
        for date, group in signal_df.groupby(level='Date'):
            up_probs = group['up_prob'].dropna()
            if up_probs.empty:
                continue

            # Sort ascending and pick bottom 10% (lowest up_prob) and top 10% (highest up_prob)
            up_probs_sorted = up_probs.sort_values()
            n = len(up_probs_sorted)
            if n < 10:
                # If there aren't enough stocks, skip this date
                continue

            cutoff = int(0.1 * n)
            bottom_mean = up_probs_sorted.iloc[:cutoff].mean()
            top_mean = up_probs_sorted.iloc[-cutoff:].mean()

            results.append({
                'Date': date,
                'Bottom10_mean': bottom_mean,
                'Top10_mean': top_mean
            })

        # Create a DataFrame of the results
        tail_df = pd.DataFrame(results).set_index('Date').sort_index()

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(tail_df.index, tail_df['Bottom10_mean'], label='Bottom 10% Avg Confidence')
        plt.plot(tail_df.index, tail_df['Top10_mean'], label='Top 10% Avg Confidence')
        plt.xlabel('Date')
        plt.ylabel('Average "up_prob"')
        plt.title('Weekly Bottom 10% vs Top 10% Average Confidence')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/weekly_confidence_tails.png")
        plt.close()

    def _add_period_ret_to_us_res_df_w_delays(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper that merges the main DataFrame with the period returns for each delay.
        Also merges MarketCap, which we rename in period_ret to avoid overwriting.
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

        # Filter date range
        signal_df = signal_df[
            signal_df.index.get_level_values("Date").year.isin(range(self.start_year, self.end_year + 1))
        ]
        # Keep period_ret from start_year-1 onward
        period_ret = period_ret[period_ret.index.get_level_values("Date").year >= (self.start_year - 1)]

        if self.delay_list is None:
            self.delay_list = [0]
        columns_needed = ["MarketCap"] + [f"next_{self.freq}_ret_{dl}delay" for dl in self.delay_list]
        if self.custom_ret is not None:
            columns_needed.append(self.custom_ret)

        # Rename MarketCap in period_ret
        period_ret = period_ret.rename(columns={"MarketCap": "MC_from_ret"})

        # Merge
        merged_df = signal_df.join(period_ret, how="inner")

        # The "no delay" return column
        merged_df[self.no_delay_ret_name] = merged_df[f"next_{self.freq}_ret_0delay"]

        # Drop rows missing required columns
        merged_df.dropna(subset=columns_needed, inplace=True)
        merged_df.dropna(subset=[self.no_delay_ret_name], inplace=True)

        return merged_df

    def _get_up_prob_with_period_ret(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge up_prob with next_{freq}_ret_Xdelay columns,
        then optionally filter by daily volume or top slice by that volume, etc.
        """
        merged_df = self._add_period_ret_to_us_res_df_w_delays(signal_df)

        # Example of computing DollarVolume and filtering or slicing top fraction
        merged_df["DollarVolume"] = merged_df["Close"] * merged_df["Vol"]

        def keep_top_60_percent_dailydollarvolume(group: pd.DataFrame) -> pd.DataFrame:
            group = group.sort_values("DollarVolume", ascending=False)
            keep_n = int(len(group) * 0.53)  # keep top 60% by DollarVolume
            
            # Print what is the cutoff, the dollarvolume cutoff.
            print(f"[INFO] DollarVolume cutoff: {group.iloc[keep_n]['DollarVolume']:.2f}")
            
            return group.iloc[:keep_n]

        df_reset = merged_df.reset_index()
        prior_volume_filter = (
            df_reset.groupby("Date", group_keys=False)
            .apply(keep_top_60_percent_dailydollarvolume)
            .reset_index(drop=True)
        )
        final_df = prior_volume_filter.set_index(["Date", "StockID"])
        return final_df

    def _calculate_portfolio_rets_for_tail(
        self,
        df: pd.DataFrame,
        weight_type: str,
        tail_percent: float,
        delay: int
    ) -> pd.DataFrame:
        """
        For a single tail_percent and single delay, compute returns for:
            - Low_{tp}  (short side)
            - High_{tp} (long side)
            - H-L_{tp}

        We replicate the Bryan Kelly transaction cost approach:
        - The short leg is assigned negative weights, so its returns are negative.
        - If self.transaction_cost is True, we also subtract (weight * transaction_fee * 2)
            from inv_ret, which effectively pays a cost on both sides.

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
        ls_col = []
        turnover = np.zeros(len(dates) - 1)
        prev_to_df = None

        # Determine the actual up_prob cutoffs
        low_q = tail_percent * 100.0
        high_q = 100.0 - (tail_percent * 100.0)

        def pick_bottom_tail(data_df: pd.DataFrame, q: float) -> pd.DataFrame:
            up_prob_series = data_df["up_prob"]
            cutoff = np.percentile(up_prob_series, q)
            return data_df[up_prob_series < cutoff].copy()   # Use strict < instead of <= if desired

        def pick_top_tail(data_df: pd.DataFrame, q: float) -> pd.DataFrame:
            up_prob_series = data_df["up_prob"]
            cutoff = np.percentile(up_prob_series, q)
            return data_df[up_prob_series >= cutoff].copy()

        # Weighted returns logic
        def assign_weights(subset_df: pd.DataFrame, wtype: str) -> pd.DataFrame:
            if subset_df.empty:
                subset_df["weight"] = 0.0
            else:
                if wtype == "ew":
                    subset_df["weight"] = 1.0 / len(subset_df)
                else:
                    total_value = subset_df["MarketCap"].sum()
                    if total_value == 0:
                        subset_df["weight"] = 0.0
                    else:
                        subset_df["weight"] = subset_df["MarketCap"] / total_value
            return subset_df

        for i, d in enumerate(dates):
            daily_df = df.loc[d]
            if daily_df.empty:
                low_col.append(0.0)
                high_col.append(0.0)
                ls_col.append(0.0)
                continue

            # Pick bottom and top tails
            bottom_df = pick_bottom_tail(daily_df, low_q)
            top_df    = pick_top_tail(daily_df, high_q)

            # Assign weights
            bottom_df = assign_weights(bottom_df, weight_type)
            top_df    = assign_weights(top_df, weight_type)

            # Negative weights for short positions
            bottom_df["weight"] = bottom_df["weight"] * (-1)

            # Now compute "inv_ret" = weight * stock_return
            bottom_df["inv_ret"] = bottom_df["weight"] * bottom_df[ret_name]
            top_df["inv_ret"]    = top_df["weight"] * top_df[ret_name]

            # If transaction_cost is True, subtract cost
            if self.transaction_cost:
                bottom_df["inv_ret"] = bottom_df["inv_ret"] - (
                    bottom_df["weight"] * self.transaction_fee * 2
                )
                top_df["inv_ret"] = top_df["inv_ret"] - (
                    top_df["weight"] * self.transaction_fee * 2
                )

            # Concatenate and GROUP BY the index to remove duplicates
            to_df = pd.concat([bottom_df, top_df])
            to_df = to_df.groupby(to_df.index).sum()  # <--- This ensures no duplicates

            # Turnover logic
            if i > 0 and prev_to_df is not None:
                # Build a DataFrame with the union of indices
                all_idx = np.unique(list(to_df.index) + list(prev_to_df.index))
                tto_df = pd.DataFrame(index=all_idx)
                tto_df["cur_weight"] = to_df["weight"]

                # Bring over previous info
                tto_df[["prev_weight", "ret", "inv_ret"]] = prev_to_df[
                    ["weight", ret_name, "inv_ret"]
                ]
                tto_df.fillna(0, inplace=True)

                # Adjust previous weights for returns
                denom = 1.0 + tto_df["inv_ret"].sum()
                adjusted_prev = tto_df["prev_weight"] * (1 + tto_df["ret"]) / denom

                # sum of abs(cur_weight - adjusted_prev) is the gross turnover
                # multiply by 0.5 to avoid double-counting
                cur_turnover = (tto_df["cur_weight"] - adjusted_prev).abs().sum() * 0.5
                turnover[i - 1] = cur_turnover

            # Save for next iteration
            prev_to_df = to_df.copy()
            prev_to_df[ret_name] = prev_to_df[ret_name].fillna(0.0)

            # L-S return = top minus bottom (both sides accounted for sign).
            # bottom side is negative weighting, so just sum them.
            bottom_ret = to_df.loc[to_df["weight"] < 0.0, "inv_ret"].sum()
            top_ret    = to_df.loc[to_df["weight"] > 0.0, "inv_ret"].sum()
            ls_ret     = top_ret + bottom_ret

            low_col.append(bottom_ret)
            high_col.append(top_ret)
            ls_col.append(ls_ret)

        # Build the daily return DataFrame
        low_name  = f"Low_{int(tail_percent*100)}%"
        high_name = f"High_{int(tail_percent*100)}%"
        ls_name   = f"H-L_{int(tail_percent*100)}%"

        daily_ret_df = pd.DataFrame({
            low_name: low_col,
            high_name: high_col,
            ls_name: ls_col
        }, index=dates)

        avg_turn = np.mean(turnover)
        print(f"[INFO] Average turnover for {low_name}, {high_name}, {ls_name}: {avg_turn:.4f}")
        return daily_ret_df, avg_turn


    def calculate_portfolio_rets(
        self,
        weight_type: str,
        delay: int = 0
    ) -> (pd.DataFrame, dict):
        """
        For each specified tail_percent, create 3 columns in the final DataFrame:
          Low_{p}, High_{p}, H-L_{p}.
        Also track turnover in a dict { tail_percent: avg_turnover }.

        We optionally apply the daily volume filter (in _get_up_prob_with_period_ret)
        and the Bryan Kelly transaction cost approach (negative weights for shorts,
        subtracting fee from inv_ret).
        """
        if self.signal_df is None or self.signal_df.empty:
            raise ValueError("signal_df is empty or None. No data available.")

        # Optionally plot the confidence tails
        plots_dir = ut.get_dir(op.join(self.portfolio_dir, "plots"))
        self.plot_weekly_confidence_tails(self.signal_df, save_path=plots_dir)

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
        """ Convert daily (or weekly, monthly, etc.) returns into cumulative log-returns """
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
        elif self.freq == "quarter":
            period = 4
        else:
            period = 252  # fallback for daily

        avg = portfolio_ret.mean(axis=0) * period
        std = portfolio_ret.std(axis=0) * math.sqrt(period)
        sr = avg / std.replace(0, np.nan)

        df_summary = pd.DataFrame({
            "ret": avg,
            "std": std,
            "SR": sr
        })
        return df_summary.round(3)

    def annual_sharpe_ratio(self, pf_ret: pd.DataFrame, weight_type: str) -> pd.DataFrame:
        """
        Compute Sharpe ratio by calendar year for each strategy column.
        Uses the self.freq attribute to determine annualization.
        """
        if self.freq == "week":
            periods = 52
        elif self.freq == "month":
            periods = 12
        elif self.freq == "quarter":
            periods = 4
        else:
            periods = 252  # fallback for daily

        df_cp = pf_ret.copy()
        df_cp["Year"] = df_cp.index.year

        def per_year_sharpe(g: pd.DataFrame) -> pd.Series:
            g = g.drop(columns=["Year"], errors="ignore")
            avg_annual = g.mean() * periods
            std_annual = g.std() * np.sqrt(periods)
            return (avg_annual / std_annual.replace(0, np.nan))

        sr_by_year = df_cp.groupby("Year").apply(per_year_sharpe)
        sr_by_year = sr_by_year.reset_index()
        sr_by_year.columns = ["Year"] + list(sr_by_year.columns[1:])
        sr_by_year = sr_by_year.set_index("Year").round(3)

        sr_by_year_path = os.path.join(self.portfolio_dir, f"annual_sharpe_ratios_{weight_type}.csv")
        sr_by_year.to_csv(sr_by_year_path)
        return sr_by_year

    def generate_portfolio(self, delay: int = 0, cut: int = 0) -> None:
        """
        Builds the portfolio returns using the chosen tail_percent_list, then saves them
        along with a summary and a plot.

        One CSV with all columns for each tail_percent:
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

            # Add turnover info
            for p in self.tail_percent_list:
                row_name = f"Turnover_{int(p*100)}%"
                summary_df.loc[row_name, ["ret","std","SR"]] = [np.nan, np.nan, turnover_dict[p]]

            smry_path = os.path.join(self.portfolio_dir, f"{pf_name}.csv")
            summary_df.to_csv(smry_path)

            txt_path = os.path.join(self.portfolio_dir, f"{pf_name}.txt")
            with open(txt_path, "w+") as f:
                f.write(summary_df.to_string())

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

            # Annual Sharpe
            self.annual_sharpe_ratio(
                pf_ret=portfolio_ret,
                weight_type=weight_type
            )

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
        smry_path = os.path.join(self.portfolio_dir, f"{pf_name}.csv")
        if not op.isfile(smry_path):
            raise FileNotFoundError(f"Portfolio summary not found at {smry_path}")
        return pd.read_csv(smry_path, index_col=0)


def main():
    pass

if __name__ == "__main__":
    main()
