# src/data/generate_chart.py

"""
generate_chart.py

Generates chart images (OHLC) from daily data for each stock, saving them
in a memory-mapped file. These images can then feed into CNN training.

Key modifications to speed up:
  1) We add a flag 'verbose_missing_date' to disable spamming logs for each missing date.
  2) (Optionally) we show an example of parallel processing with joblib.
"""

from typing import Optional, List, Tuple, Union
import os
import os.path as op
import sys
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from src.data import dgp_config as dcf
from src.data import equity_data as eqd
from src.data.chart_library import DrawOHLC, DrawChartError
from src.utils import utilities as ut

class ChartGenerationError(Exception):
    """Custom exception for chart generation."""
    pass

class GenerateStockData:
    """
    Class to create and save bar/pixel OHLC chart images for CNN.
    By default, step_size=1 ensures daily chart generation.

    Args:
        country: e.g. "USA"
        year: e.g. 1993
        window_size: Number of daily bars in each chart.
        freq: "week", "month", etc. We use eqd.get_period_end_dates(freq) by default.
        chart_freq: Aggregates daily data (e.g., 4 means each "bar" is 4 days).
        ma_lags: List of MA lags (e.g., [20])
        volume_bar: If True, add a volume sub-chart at the bottom.
        need_adjust_price: If True, normalize so the first day's close=1.0
        allow_tqdm: If True, display progress bars
        chart_type: "bar", "pixel", or "centered_pixel"
        step_size: # days to skip between samples. Defaults to 1 (daily).
        verbose_missing_date: If True, print a line for each missing date. Otherwise skip.
    """

    def __init__(
        self,
        country: str,
        year: int,
        window_size: int,
        freq: str,
        chart_freq: int = 1,
        ma_lags: Optional[List[int]] = None,
        volume_bar: bool = False,
        need_adjust_price: bool = True,
        allow_tqdm: bool = True,
        chart_type: str = "bar",
        step_size: Optional[int] = 1,
        verbose_missing_date: bool = True
    ) -> None:
        self.country = country
        self.year = year
        self.window_size = window_size
        self.freq = freq
        self.chart_freq = chart_freq
        self.chart_len = int(window_size / chart_freq)
        self.ma_lags = ma_lags
        self.volume_bar = volume_bar
        self.need_adjust_price = need_adjust_price
        self.allow_tqdm = allow_tqdm
        self.chart_type = chart_type
        self.step_size = step_size if step_size else 1
        self.verbose_missing_date = verbose_missing_date

        # For storing label columns:
        self.ret_len_list = [5, 20, 60, 65, 180, 250, 260]

        # Directory structure
        self.save_dir = ut.get_dir(op.join(dcf.STOCKS_SAVEPATH, f"stocks_{country}", "dataset_all"))
        vb_str = "has_vb" if self.volume_bar else "no_vb"
        ohlc_len_str = "" if self.chart_freq == 1 else f"_{self.chart_len}ohlc"
        chart_type_str = "" if self.chart_type == "bar" else f"{self.chart_type}_"
        self.file_name = (
            f"{chart_type_str}{self.window_size}d_{self.freq}_{vb_str}_{str(self.ma_lags)}_ma_{self.year}{ohlc_len_str}"
        )

        self.log_file_name = op.join(self.save_dir, f"{self.file_name}.txt")
        self.labels_filename = op.join(self.save_dir, f"{self.file_name}_labels.feather")
        self.images_filename = op.join(self.save_dir, f"{self.file_name}_images.dat")

        self.df: Union[pd.DataFrame, None] = None
        self.stock_id_list: Union[np.ndarray, None] = None

    def save_annual_data(self) -> None:
        """
        Create chart images for the specified year, memory-map them, and store label data
        in a Feather file. If existing pre-generated files appear valid, skip regeneration.
        """
        if self._pre_generated_file_exists_and_valid():
            print(f"Found valid pre-generated file {self.file_name}, skipping.")
            return

        # Remove old partial files
        self._remove_old_files()

        print(f"Generating {self.file_name}")
        # load the raw processed data for (year-2..year)
        self.df = eqd.get_processed_us_data_by_year(self.year)

        self.stock_id_list = np.unique(self.df.index.get_level_values("StockID"))

        # Provisional capacity ~60 samples per stock
        capacity = len(self.stock_id_list) * 60

        # Prepare arrays
        dtype_dict, feature_list = self._get_feature_and_dtype_list()
        data_dict = {feature: np.empty(capacity, dtype=dtype_dict[feature]) for feature in feature_list}
        data_dict["image"] = np.empty((capacity, self._img_width * self._img_height), dtype=dtype_dict["image"])
        data_dict["image"].fill(0)

        sample_num = 0
        data_miss = np.zeros(6)  # track error codes

        # We'll build an iterator for stock IDs
        iterator = self.stock_id_list
        if self.allow_tqdm and ("tqdm" in sys.modules):
            iterator = tqdm(iterator, desc="GenerateChart", unit="stock")

        for stock_id in iterator:
            stock_df = self.df.xs(stock_id, level=1).copy().reset_index()
            # Keep only the target year’s valid daily rows
            valid_dates = stock_df[~pd.isna(stock_df["Ret"])].Date
            valid_dates = valid_dates[valid_dates.dt.year == self.year].sort_values()

            # We sample every self.step_size days (default 1 => daily).
            # But here we override with eqd.get_period_end_dates(self.freq) => weekly/monthly last day:
            # If you truly want daily, comment out the 2 lines below, and use range(0,len(valid_dates),self.step_size)
            date_candidates = eqd.get_period_end_dates(self.freq)
            date_candidates = date_candidates[date_candidates.year == self.year]
            # If you want daily:
            # date_candidates = valid_dates[::self.step_size]

            for dt in date_candidates:
                if dt not in valid_dates.values:
                    # Silently skip if not found, unless verbose_missing_date is True
                    if self.verbose_missing_date:
                        print(f"[DEBUG] Missing date {dt} for {stock_id}")
                    continue

                date = dt
                # Check capacity
                if sample_num >= capacity:
                    old_cap = capacity
                    capacity += 100
                    print(f"[DEBUG] Expanding arrays from {old_cap} to {capacity} for {stock_id} {date}")
                    for feature in feature_list:
                        old_arr = data_dict[feature]
                        new_arr = np.empty(capacity, dtype=dtype_dict[feature])
                        new_arr[:old_cap] = old_arr
                        data_dict[feature] = new_arr
                    old_img = data_dict["image"]
                    new_img = np.empty((capacity, self._img_width * self._img_height), dtype=dtype_dict["image"])
                    new_img[:old_cap, :] = old_img
                    data_dict["image"] = new_img

                # Generate features & chart
                image_label_data = self._generate_daily_features(stock_df, date)
                if isinstance(image_label_data, dict):
                    image_label_data["StockID"] = stock_id
                    im_arr = np.frombuffer(image_label_data["image"].tobytes(), dtype=np.uint8)
                    if im_arr.size != self._img_width * self._img_height:
                        print(f"[DEBUG] Mismatch in image size for {stock_id} {date}, got {im_arr.size}.")
                    data_dict["image"][sample_num, :] = im_arr[:]
                    for feature in [f for f in feature_list if f != "image"]:
                        data_dict[feature][sample_num] = image_label_data[feature]
                    sample_num += 1
                elif isinstance(image_label_data, int):
                    data_miss[image_label_data] += 1

        # Truncate
        for feature in feature_list:
            data_dict[feature] = data_dict[feature][:sample_num]
        data_dict["image"] = data_dict["image"][:sample_num, :]

        # Write to memmap
        fp_x = np.memmap(
            self.images_filename,
            dtype=np.uint8,
            mode="w+",
            shape=data_dict["image"].shape,
        )
        fp_x[:] = data_dict["image"][:]
        fp_x.flush()
        fp_x = None  # close memmap

        # Save label data
        df_out = pd.DataFrame({k: data_dict[k] for k in data_dict.keys() if k != "image"})
        df_out.to_feather(self.labels_filename)

        with open(self.log_file_name, "w+") as log_file:
            log_file.write(f"total_dates:{sample_num} total_missing:{int(np.sum(data_miss))}\n")

        print(f"[INFO] Wrote memmap to {self.images_filename}")
        print(f"[INFO] Wrote label data to {self.labels_filename}, shape={df_out.shape}")

    def save_annual_ts_data(self) -> None:
        """Placeholder if you also generate TS1D data."""
        pass

    def _remove_old_files(self) -> None:
        for fpath in [self.log_file_name, self.labels_filename, self.images_filename]:
            if op.isfile(fpath):
                print(f"[DEBUG] Removing old file {fpath}")
                os.remove(fpath)

    def _pre_generated_file_exists_and_valid(self) -> bool:
        if not (op.isfile(self.log_file_name) and op.isfile(self.labels_filename) and op.isfile(self.images_filename)):
            return False
        try:
            images_mem = np.memmap(self.images_filename, dtype=np.uint8, mode="r")
            total_size = images_mem.shape[0]
            exp_pixels = self._img_width * self._img_height
            if total_size % exp_pixels != 0:
                return False
        except Exception:
            return False
        return True

    @property
    def _img_width(self) -> int:
        return dcf.IMAGE_WIDTH[self.chart_len]

    @property
    def _img_height(self) -> int:
        base_h = dcf.IMAGE_HEIGHT[self.chart_len]
        if self.volume_bar:
            base_h += int(base_h / 5) + dcf.VOLUME_CHART_GAP
        return base_h

    def _generate_daily_features(
        self, stock_df: pd.DataFrame, date: pd.Timestamp
    ) -> Union[dict, int]:
        """
        Build a daily chart for one stock & date. Returns a dict if success, int code on error.
        """
        res = self.load_adjusted_daily_prices(stock_df, date)
        if isinstance(res, int):
            return res

        df, local_ma_lags = res
        try:
            ohlc_obj = DrawOHLC(
                df,
                has_volume_bar=self.volume_bar,
                ma_lags=local_ma_lags,
                chart_type=self.chart_type
            )
            image_data = ohlc_obj.draw_image()
            if image_data is None:
                return 5
        except DrawChartError:
            return 5

        # last row
        last_day = df[df.Date == date].iloc[0]
        feature_dict = {col: last_day[col] for col in stock_df.columns if col in last_day}

        # add classification/regression labels
        ret_list = ["Ret"] + [f"Ret_{i}d" for i in self.ret_len_list]
        for ret in ret_list:
            feature_dict[f"{ret}_label"] = 1 if feature_dict.get(ret, 0) > 0 else 0
            vol = feature_dict.get("EWMA_vol", 0.0)
            if (vol is None) or (vol == 0.0) or pd.isna(vol):
                feature_dict[f"{ret}_tstat"] = 0.0
            else:
                feature_dict[f"{ret}_tstat"] = feature_dict.get(ret, 0.0) / vol

        feature_dict["image"] = image_data
        feature_dict["window_size"] = self.window_size
        feature_dict["Date"] = date
        return feature_dict

    def load_adjusted_daily_prices(
        self, stock_df: pd.DataFrame, date: pd.Timestamp
    ) -> Union[int, Tuple[pd.DataFrame, List[int]]]:
        if date not in set(stock_df.Date):
            return 0
        date_index = stock_df[stock_df.Date == date].index[0]
        ma_offset = 0 if self.ma_lags is None else max(self.ma_lags)
        data = stock_df.loc[(date_index - (self.window_size - 1) - ma_offset): date_index]
        if len(data) < self.window_size:
            return 1
        if len(data) < (self.window_size + ma_offset):
            local_ma_lags = []
            data = stock_df.loc[(date_index - (self.window_size - 1)): date_index]
        else:
            local_ma_lags = self.ma_lags if self.ma_lags else []

        if self.chart_freq != 1:
            try:
                data = self.convert_daily_df_to_chart_freq_df(data)
            except ChartGenerationError:
                return 2

        if self.need_adjust_price and (data["Close"].iloc[0] == 0.0 or pd.isna(data["Close"].iloc[0])):
            return 2

        if self.need_adjust_price:
            try:
                data = self.adjust_price(data)
            except ChartGenerationError:
                return 2

        start_ix = data.index[-1] - self.chart_len + 1
        if data["Close"].loc[start_ix] == 0 or np.isnan(data["Close"].loc[start_ix]):
            return 3
        factor = 1.0 / data["Close"].loc[start_ix]
        data[["Open", "High", "Low", "Close"]] *= factor

        if local_ma_lags:
            for ml in local_ma_lags:
                ma_name = f"ma{ml}"
                data[ma_name] = data["Close"].rolling(int(ml / self.chart_freq)).mean()

        data["Prev_Close"] = data["Close"].shift(1)
        df = data.loc[start_ix:].reset_index(drop=True)

        if (len(df) != self.chart_len) or (round(df.iloc[0]["Close"], 3) != 1.000):
            return 4

        df["Date"] = pd.to_datetime(df["Date"])
        return df, local_ma_lags

    def convert_daily_df_to_chart_freq_df(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        length = len(daily_df)
        if length % self.chart_freq != 0:
            raise ChartGenerationError("df not divisible by chart_freq")

        ohlc_len = length // self.chart_freq
        out = pd.DataFrame(index=range(ohlc_len), columns=daily_df.columns)
        for i in range(ohlc_len):
            chunk = daily_df.iloc[i*self.chart_freq:(i+1)*self.chart_freq]
            out.loc[i] = chunk.iloc[-1]
            out.loc[i, "Open"] = chunk.iloc[0]["Open"]
            out.loc[i, "High"] = chunk["High"].max()
            out.loc[i, "Low"] = chunk["Low"].min()
            out.loc[i, "Vol"] = chunk["Vol"].sum()
            out.loc[i, "Ret"] = np.prod(1 + np.array(chunk["Ret"])) - 1
        return out

    @staticmethod
    def adjust_price(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            raise ChartGenerationError("Empty DataFrame in adjust_price.")
        if len(df.Date.unique()) != len(df):
            raise ChartGenerationError("Dates not unique in chunk for adjust_price.")

        df = df.reset_index(drop=True)
        fd_close = abs(df.at[0, "Close"])
        if fd_close == 0.0 or pd.isna(fd_close):
            raise ChartGenerationError("First day close is zero/nan.")

        res_df = df.copy()
        res_df.at[0, "Close"] = 1.0
        res_df.at[0, "Open"] = abs(res_df.at[0, "Open"]) / fd_close
        res_df.at[0, "High"] = abs(res_df.at[0, "High"]) / fd_close
        res_df.at[0, "Low"] = abs(res_df.at[0, "Low"]) / fd_close
        pre_close = 1.0

        for i in range(1, len(res_df)):
            ret = float(res_df.at[i, "Ret"])
            this_close = (1 + ret)*pre_close
            orig_close = abs(res_df.at[i, "Close"])
            if orig_close == 0.0 or pd.isna(orig_close):
                continue
            res_df.at[i, "Close"] = this_close
            scale = this_close/orig_close
            res_df.at[i, "Open"] *= scale
            res_df.at[i, "High"] *= scale
            res_df.at[i, "Low"] *= scale
            res_df.at[i, "Ret"] = ret
            pre_close = this_close
        return res_df

    def _get_feature_and_dtype_list(self):
        float32_features = [
            "EWMA_vol", "Ret", "Ret_tstat", "Ret_week", "Ret_month", "Ret_quarter",
            "MarketCap",
        ] + [f"Ret_{i}d" for i in self.ret_len_list] + [f"Ret_{i}d_tstat" for i in self.ret_len_list]

        int8_features = ["Ret_label"] + [f"Ret_{i}d_label" for i in self.ret_len_list]
        uint8_features = ["window_size"]
        object_features = ["StockID"]
        datetime_features = ["Date"]

        feature_list = float32_features + int8_features + uint8_features + object_features + datetime_features
        float32_dict = {f: np.float32 for f in float32_features}
        int8_dict = {f: np.int8 for f in int8_features}
        uint8_dict = {f: np.uint8 for f in uint8_features}
        object_dict = {f: object for f in object_features}
        datetime_dict = {f: "datetime64[ns]" for f in datetime_features}
        dtype_dict = {
            **float32_dict,
            **int8_dict,
            **uint8_dict,
            **object_dict,
            **datetime_dict,
        }
        dtype_dict["image"] = np.uint8
        return dtype_dict, feature_list
