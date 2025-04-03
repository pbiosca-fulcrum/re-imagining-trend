# src/portfolio/portfolio.py
import os
import os.path as op
import pdb
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        self.delay_list = [0] if delay_list is None else delay_list
        self.custom_ret = custom_ret
        self.transaction_cost = transaction_cost
        self.no_delay_ret_name = f"next_{freq}_ret"
        
        if load_signal:
            if "up_prob" not in signal_df.columns:
                raise ValueError("signal_df must have an 'up_prob' column if load_signal=True.")
            self.signal_df = self.get_up_prob_with_period_ret(signal_df)
        else:
            self.signal_df = None

    def __add_period_ret_to_us_res_df_w_delays(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        period_ret = eqd.get_period_ret(self.freq, country=self.country)

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

        signal_df = signal_df[
            signal_df.index.get_level_values("Date").year.isin(range(self.start_year, self.end_year + 1))
        ]
        period_ret = period_ret[period_ret.index.get_level_values("Date").year > 2000]

        columns = ["MarketCap"] + [f"next_{self.freq}_ret_{dl}delay" for dl in self.delay_list]
        if self.custom_ret is not None:
            columns.append(self.custom_ret)

        period_ret = period_ret.rename(columns={"MarketCap": "MC_from_ret"})

        # Inner join on the MultiIndex
        merged_df = signal_df.join(
            period_ret[["MC_from_ret", f"next_{self.freq}_ret_0delay"]],
            how="inner"
        )

        # Our "no delay" return column
        merged_df[self.no_delay_ret_name] = merged_df[f"next_{self.freq}_ret_0delay"]

        print(f"[DEBUG] Merged DataFrame columns: {merged_df.columns}")
        print(f"[DEBUG] Merged DataFrame head:\n{merged_df.head()}")
        print(f"[DEBUG] Merged DataFrame shape: {merged_df.shape}")
        print(f"[DEBUG] --------------------------------------")

        # Drop rows missing required columns (MarketCap, etc.)
        merged_df.dropna(subset=columns, inplace=True)
        merged_df.dropna(subset=[self.no_delay_ret_name], inplace=True)

        # Debug prints for each delay
        for dl in self.delay_list:
            dl_ret_name = f"next_{self.freq}_ret_{dl}delay"
            if dl_ret_name not in merged_df.columns:
                pass
            else:
                pass

        return merged_df

    def get_up_prob_with_period_ret(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        filtered_df = signal_df[
            signal_df.index.get_level_values("Date").year.isin(range(self.start_year, self.end_year + 1))
        ]
        if filtered_df.empty:
            return filtered_df

        final_df = self.__add_period_ret_to_us_res_df_w_delays(filtered_df)

        if self.country not in ["future", "new_future"]:
            final_df["MarketCap"] = final_df["MarketCap"].abs()
            final_df = final_df[~final_df["MarketCap"].isnull() & (final_df["MarketCap"] > 0)]

        # Do some statistics on what the distribution of market cap looks like and save it by year
        if not final_df.empty:
            # Reset index to get Date as a column
            df_reset = final_df.reset_index()
            df_reset['Year'] = pd.DatetimeIndex(df_reset['Date']).year
            
            # For each year, get the last market cap for each stock
            yearly_market_caps = {}
            for year in range(self.start_year, self.end_year + 1):
                year_df = df_reset[df_reset['Year'] == year]
            if not year_df.empty:
                # Get the last observation for each StockID in this year
                last_obs = year_df.sort_values('Date').groupby('StockID').last()
                yearly_market_caps[year] = last_obs['MarketCap']
            
            # Create overall plot with subplots for each year
            num_years = self.end_year - self.start_year + 1
            cols = min(3, num_years)
            rows = (num_years + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1 or cols == 1:
                axes = axes.reshape(-1, 1) if cols == 1 else axes.reshape(1, -1)
            
            year_idx = 0
            for year, market_caps in sorted(yearly_market_caps.items()):
                row, col = divmod(year_idx, cols)
                ax = axes[row, col]
            
            # Use log scale for MarketCap
            log_market_cap = np.log10(market_caps)
            ax.hist(log_market_cap, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_xlabel('Market Cap (log10 scale)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Market Cap Distribution - {year}')
            
            # Add vertical lines for common market cap thresholds
            thresholds = [1e6, 1e7, 1e8, 1e9, 1e10]
            labels = ['$1M', '$10M', '$100M', '$1B', '$10B']
            for threshold, label in zip(thresholds, labels):
                ax.axvline(np.log10(threshold), color='red', linestyle='--', alpha=0.7)
                ax.text(np.log10(threshold), ax.get_ylim()[1]*0.9, label, 
                   rotation=90, verticalalignment='top', fontsize=8)
            
            # Add summary statistics as text
            stats = (f"N: {len(market_caps)}\n"
                   f"Mean: ${market_caps.mean()/1e9:.2f}B\n"
                   f"Median: ${market_caps.median()/1e9:.2f}B\n"
                   f"Min: ${market_caps.min()/1e6:.2f}M\n"
                   f"Max: ${market_caps.max()/1e9:.2f}B")
            ax.text(0.02, 0.95, stats, transform=ax.transAxes, 
                  bbox=dict(facecolor='white', alpha=0.8), fontsize=8)
            
            year_idx += 1
            
            # Hide any unused subplots
            for i in range(year_idx, rows * cols):
                row, col = divmod(i, cols)
                axes[row, col].axis('off')
            
            plt.tight_layout()
            os.makedirs(op.join(self.portfolio_dir, 'analysis'), exist_ok=True)
            plt.savefig(op.join(self.portfolio_dir, 'analysis', f'market_cap_by_year_{self.country}.png'))
            plt.close()
            
            # Also create a single plot with all years for comparison
            plt.figure(figsize=(12, 8))
            for year, market_caps in sorted(yearly_market_caps.items()):
            # Plot each year as a line showing the distribution
                sns_data = pd.DataFrame({'MarketCap': np.log10(market_caps), 'Year': str(year)})
                sns.kdeplot(data=sns_data, x='MarketCap', label=f'{year} (n={len(market_caps)})')
            
            plt.xlabel('Market Cap (log10 scale)')
            plt.ylabel('Density')
            plt.title(f'Market Cap Distribution by Year ({self.country})')
            
            # Add vertical lines for common thresholds
            for threshold, label in zip(thresholds, labels):
                plt.axvline(np.log10(threshold), color='gray', linestyle=':', alpha=0.5)
                plt.text(np.log10(threshold), plt.ylim()[1]*0.95, label, 
                    rotation=90, verticalalignment='top', fontsize=8)
            
            plt.legend()
            plt.tight_layout()
            plt.savefig(op.join(self.portfolio_dir, 'analysis', f'market_cap_yearly_comparison_{self.country}.png'))
            plt.close()

        # Added filters:
        # Log final_df
        print(f"[DEBUG] Final DataFrame columns: {final_df.head(20)}")
        
        # # Filter for top 20% largest companies by MarketCap for each year
        # final_df = final_df.reset_index()
        # final_df['Year'] = pd.DatetimeIndex(final_df['Date']).year
        
        # # Group by year and keep only top 20% by market cap
        # def top_quintile(group):
        #     threshold = group['MarketCap'].quantile(0.5)  # 80th percentile (top 20%)
        #     return group[group['MarketCap'] < threshold]
        
        # final_df = final_df.groupby('Year').apply(top_quintile).reset_index(drop=True)
        # final_df = final_df.set_index(['Date', 'StockID'])
        
        # print(f"[DEBUG] Filtered DataFrame columns: {final_df.head(20)}")
        # breakpoint()
        # if "Vol" in final_df.columns:
        #     final_df = final_df[final_df["Vol"] >= 1e6]     # >= 1 million daily volume

        return final_df

    def calculate_portfolio_rets(
        self,
        weight_type: str,
        cut: int = 10,
        delay: int = 0
    ) -> (pd.DataFrame, float):
        assert weight_type in ["ew", "vw"], "weight_type must be either 'ew' or 'vw'."
        assert delay in self.delay_list, f"delay={delay} not in the allowed list: {self.delay_list}"

        if self.custom_ret:
            ret_name = self.custom_ret
        else:
            ret_name = (
                self.no_delay_ret_name
                if delay == 0
                else f"next_{self.freq}_ret_{delay}delay"
            )

        df = self.signal_df.copy() if self.signal_df is not None else None
        if df is None or df.empty:
            raise ValueError(
                "[ERROR] No signal data (signal_df is empty or None) after merges and filtering. "
                "Check date ranges, merges, or if your up_prob CSV has valid data."
            )

        dates = np.sort(np.unique(df.index.get_level_values("Date")))
        if len(dates) == 0:
            raise ValueError(
                "[ERROR] The final data has no valid Dates. Possibly the date filtering or merges left it empty. "
                "Cannot compute decile portfolios with zero rows."
            )

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

                if self.transaction_cost:
                    pass

                portfolio_ret.loc[d, j] = decile_df["inv_ret"].sum()

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

            sell_decile[["weight", "inv_ret"]] = sell_decile[["weight", "inv_ret"]] * (-1)
            to_df = pd.concat([sell_decile, buy_decile]) if not buy_decile.empty else sell_decile

            if not to_df.empty:
                to_df = to_df.groupby(to_df.index).sum()
            if i > 0 and prev_to_df is not None and not prev_to_df.empty:
                prev_to_df = prev_to_df.groupby(prev_to_df.index).sum()

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

            # if i > 0 and self.transaction_cost:
            #     threshold = np.percentile(rebalance_df["MarketCap"], 80)
            #     cost_rates = np.where(rebalance_df["MarketCap"] >= threshold, 0.001, 0.002)
            #     if weight_type == "ew":
            #         avg_cost_rate_overall = np.mean(cost_rates)
            #     else:
            #         weights = rebalance_df["MarketCap"] / rebalance_df["MarketCap"].sum()
            #         avg_cost_rate_overall = np.sum(weights * cost_rates)
            #     portfolio_ret.loc[d] = portfolio_ret.loc[d] - avg_cost_rate_overall * turnover[i - 1]

        portfolio_ret = portfolio_ret.fillna(0.0)
        portfolio_ret["H-L"] = portfolio_ret[cut - 1] - portfolio_ret[0]

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
        ret_name = "nxt_freq_ewret" if weight_type == "ew" else "nxt_freq_vwret"
        df = portfolio_ret.copy()
        df.columns = ["Low(L)"] + [str(i) for i in range(2, cut)] + ["High(H)", "H-L"]

        spy = eqd.get_spy_freq_rets(self.freq)
        if ret_name not in spy.columns:
            df["SPY"] = spy[spy.columns[-1]]
        else:
            df["SPY"] = spy[ret_name]

        df.dropna(inplace=True)

        log_ret_df = pd.DataFrame(index=df.index)
        for column in df.columns:
            log_ret_df[column] = self._ret_to_cum_log_ret(df[column])

        # Insert a starting point for previous year's end if needed
        if len(log_ret_df) > 0:
            prev_year = pd.to_datetime(log_ret_df.index[0]).year - 1
            prev_day = pd.to_datetime(f"{prev_year}-12-31")
            log_ret_df.loc[prev_day] = [0] * len(log_ret_df.columns)
            log_ret_df.sort_index(inplace=True)

        plot_cols = ["Low(L)", "High(H)", "H-L", "SPY"]
        log_ret_df = log_ret_df[plot_cols]

        custom_colors = {
            "Low(L)": "darkred",
            "High(H)": "darkgreen",
            "H-L": "darkblue",
            "SPY": "gray"
        }
        
        plt.figure(figsize=(10, 6))
        for col in plot_cols:
            plt.plot(log_ret_df.index, log_ret_df[col], label=col, color=custom_colors.get(col, "gray"), lw=1)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.title(plot_title)
        plt.legend(loc=2)
        plt.grid()
        plt.savefig(save_path)
        plt.close()

    def portfolio_res_summary(
        self,
        portfolio_ret: pd.DataFrame,
        turnover: float,
        cut: int = 10
    ) -> pd.DataFrame:
        avg = portfolio_ret.mean().to_numpy()
        std = portfolio_ret.std().to_numpy()
        res = np.zeros((cut + 1, 3))

        if self.freq == "week":
            period = 52
        elif self.freq == "month":
            period = 12
        else:
            period = 4

        res[:, 0] = avg * period
        res[:, 1] = std * math.sqrt(period)
        res[:, 2] = res[:, 0] / (res[:, 1])

        summary_df = pd.DataFrame(res, columns=["ret", "std", "SR"])
        index_names = ["Low"] + list(map(str, range(2, int(cut)))) + ["High", "H-L"]
        summary_df = summary_df.set_index(pd.Index(index_names))

        freq_factor = 0.25 if self.freq == "week" else (1 if self.freq == "month" else 3)
        turnover_annualized = turnover / freq_factor
        summary_df.loc["Turnover", :] = [np.nan, np.nan, turnover_annualized]

        return summary_df

    def generate_portfolio(self, cut: int = 10, delay: int = 0) -> None:
        if self.signal_df is None or self.signal_df.empty:
            raise ValueError(
                "signal_df is empty or None. There's no data to generate portfolios. "
                "Check if your CSV had rows in the date range."
            )
        assert delay in self.delay_list, f"Delay {delay} is not in {self.delay_list}."

        for weight_type in ["ew", "vw"]:
            pf_name = self.get_portfolio_name(weight_type, delay, cut)
            print(f"Calculating portfolio named '{pf_name}' ...")

            portfolio_ret, turnover = self.calculate_portfolio_rets(
                weight_type=weight_type,
                cut=cut,
                delay=delay
            )
            
            data_dir = ut.get_dir(op.join(self.portfolio_dir, "pf_data"))
            pf_data_path = op.join(data_dir, f"pf_data_{pf_name}.csv")
            portfolio_ret.to_csv(pf_data_path)
            
            summary_df = self.portfolio_res_summary(portfolio_ret, turnover, cut)
            smry_path = os.path.join(self.portfolio_dir, f"{pf_name}.csv")
            summary_df.to_csv(smry_path)
            
            txt_path = os.path.join(self.portfolio_dir, f"{pf_name}.txt")
            with open(txt_path, "w+") as file:
                summary_df = summary_df.astype(float).round(2)
                file.write(ut.to_latex_w_turnover(summary_df, cut=cut))
            
            print(f"[INFO] Portfolio '{pf_name}' results saved to:\n  - {pf_data_path}\n  - {smry_path}")
            
            # --- Plotting ---
            from src.portfolio import plot_portfolio_performance as ppp
            plots_dir = ut.get_dir(op.join(self.portfolio_dir, "plots"))
            print(f"[INFO] Generating cumulative returns plots for portfolio '{pf_name}'...")
            ppp.plot_all_years_cumulative_returns(portfolio_ret, plots_dir, weight_type)

            combined_plot_path = os.path.join(plots_dir, f"combined_cumulative_returns_{pf_name}.png")
            collapse_periods = [
                (pd.Timestamp("2007-10-01"), pd.Timestamp("2009-03-01"), "GFC"),
                (pd.Timestamp("2020-02-20"), pd.Timestamp("2020-03-23"), "COVID")
            ]
            ppp.plot_all_returns_with_shading(
                portfolio_ret,
                combined_plot_path,
                collapse_periods,
                title=f"Combined Cumulative Returns ({weight_type.upper()})"
            )
            
            annual_perf = ppp.assess_yearly_performance(portfolio_ret)
            annual_perf_path = os.path.join(self.portfolio_dir, f"{pf_name}_annual_performance.csv")
            annual_perf.to_csv(annual_perf_path)

    def get_portfolio_name(self, weight_type: str, delay: int, cut: int) -> str:
        assert weight_type.lower() in ["ew", "vw"]
        delay_prefix = "" if delay == 0 else f"{delay}d_delay_"
        cut_suffix = "" if cut == 10 else f"_{cut}cut"
        custom_ret_suffix = f"_{self.custom_ret}" if self.custom_ret else ""
        tc_suffix = "_w_transaction_cost" if self.transaction_cost else ""
        pf_name = f"{delay_prefix}{weight_type.lower()}{cut_suffix}{custom_ret_suffix}{tc_suffix}"
        return pf_name

    def load_portfolio_ret(self, weight_type: str, cut: int = 10, delay: int = 0) -> pd.DataFrame:
        pf_name = self.get_portfolio_name(weight_type, delay, cut)
        data_dir = op.join(self.portfolio_dir, "pf_data")
        pf_path = op.join(data_dir, f"pf_data_{pf_name}.csv")
        if not op.isfile(pf_path):
            pf_path = op.join(data_dir, f"pf_data_{pf_name}_100.csv")
        df = pd.read_csv(pf_path, index_col=0)
        df.index = pd.to_datetime(df.index)
        return df

    def load_portfolio_summary(self, weight_type: str, cut: int = 10, delay: int = 0) -> pd.DataFrame:
        pf_name = self.get_portfolio_name(weight_type, delay, cut)
        smry_path = op.join(self.portfolio_dir, f"{pf_name}.csv")
        if not op.isfile(smry_path):
            smry_path = op.join(self.portfolio_dir, f"{pf_name}_100.csv")
        df = pd.read_csv(smry_path, index_col=0)
        return df

def main():
    pass

if __name__ == "__main__":
    main()
