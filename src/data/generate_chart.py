"""
generate_chart.py

Generates chart images (OHLC) from daily data for each stock, saving them
in a memory-mapped file. These images can then feed into CNN training.

**Changes**:
- Force removal of existing output files before writing new ones to avoid partial/corrupt data.
- Added more debug logging for shape checks, sample counts, and final memmap sizes.
- Flush and close memmap to ensure data is truly written.
- Minor docstring improvements for clarity.
"""

from typing import Optional, List, Tuple, Union
import os
import os.path as op
import sys
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


class GenerateStockData:
    """
    Class to create and save bar/pixel OHLC chart images for CNN.

    Workflow:
      - For a given year, retrieve stock data from equity_data (real or synthetic).
      - For each trading date, build an OHLC chunk (self.window_size days).
      - Optionally compute moving averages if self.ma_lags is not empty.
      - Adjust price so the first day of the chunk is 1.0, then produce an image.
      - Save the images in a memory-mapped .dat file and the labels (features) in a .feather file.
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
        chart_type: str = "bar"
    ) -> None:
        """
        Args:
            country (str): Country name, e.g. "USA".
            year (int): Year for which to generate data.
            window_size (int): Number of daily bars in a chart before prediction.
            freq (str): e.g. "week", "month", "quarter", "year".
            chart_freq (int): Aggregates daily data into blocks (e.g., 4).
            ma_lags (Optional[List[int]]): List of lags for moving averages (e.g., [20]).
            volume_bar (bool): Whether to include a volume sub-chart in the generated image.
            need_adjust_price (bool): If True, normalizes chart so the first day has close=1.0.
            allow_tqdm (bool): If True, shows progress bars.
            chart_type (str): One of ["bar", "pixel", "centered_pixel"].
        """
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

        # Ret length for storing classification/regression labels
        self.ret_len_list = [5, 20, 60, 65, 180, 250, 260]

        # Directory for saving dataset
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

        # Force removing old files to avoid leftover partial data
        self._remove_old_files()

        print(f"Generating {self.file_name}")
        self.df = eqd.get_processed_us_data_by_year(self.year)
        self.stock_id_list = np.unique(self.df.index.get_level_values("StockID"))

        # We provisionally allocate ~60 samples per stock in a year.
        capacity = len(self.stock_id_list) * 60

        # Setup dtypes and containers
        dtype_dict, feature_list = self._get_feature_and_dtype_list()
        data_dict = {feature: np.empty(capacity, dtype=dtype_dict[feature]) for feature in feature_list}
        data_dict["image"] = np.empty(
            (capacity, self._img_width * self._img_height), dtype=dtype_dict["image"]
        )
        data_dict["image"].fill(0)

        sample_num = 0
        data_miss = np.zeros(6)  # track error codes

        iterator = (
            tqdm(self.stock_id_list) if self.allow_tqdm and ("tqdm" in sys.modules) else self.stock_id_list
        )
        for stock_id in iterator:
            stock_df = self.df.xs(stock_id, level=1).copy().reset_index()
            dates = stock_df[~pd.isna(stock_df["Ret"])].Date
            dates = dates[dates.dt.year == self.year]

            for date in dates:
                # Expand capacity if needed
                if sample_num >= capacity:
                    old_cap = capacity
                    new_capacity = capacity + 100
                    print(f"[DEBUG] Expanding arrays from {old_cap} to {new_capacity} for {stock_id} {date}")
                    for feature in feature_list:
                        old_arr = data_dict[feature]
                        new_arr = np.empty(new_capacity, dtype=dtype_dict[feature])
                        new_arr[:old_cap] = old_arr
                        data_dict[feature] = new_arr

                    old_img = data_dict["image"]
                    new_img = np.empty(
                        (new_capacity, self._img_width * self._img_height),
                        dtype=dtype_dict["image"],
                    )
                    new_img[:old_cap, :] = old_img
                    data_dict["image"] = new_img
                    capacity = new_capacity

                image_label_data = self._generate_daily_features(stock_df, date)
                if isinstance(image_label_data, dict):
                    if sample_num < 10:
                        # Save example image for debugging
                        dbg_dir = ut.get_dir(op.join(self.save_dir, "sample_images"))
                        image_label_data["image"].save(
                            op.join(dbg_dir, f"{self.file_name}_{stock_id}_{date.strftime('%Y%m%d')}.png")
                        )

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

        # Truncate final arrays to used length
        for feature in feature_list:
            data_dict[feature] = data_dict[feature][:sample_num]
        data_dict["image"] = data_dict["image"][:sample_num, :]

        # Debug info: shape checks
        total_pixels_written = sample_num * self._img_width * self._img_height
        print(f"[DEBUG] Final sample_num = {sample_num}")
        print(f"[DEBUG] _img_width={self._img_width}, _img_height={self._img_height}")
        print(f"[DEBUG] Expecting total pixels {total_pixels_written}")
        print(f"[DEBUG] data_dict['image'] shape => {data_dict['image'].shape}")

        # Write images to memmap
        fp_x = np.memmap(
            self.images_filename,
            dtype=np.uint8,
            mode="w+",
            shape=data_dict["image"].shape,
        )
        fp_x[:] = data_dict["image"][:]
        fp_x.flush()  # Ensure data is fully written
        fp_x_base_shape = fp_x.shape
        fp_x = None  # close memmap
        print(f"[DEBUG] Wrote memmap of shape {fp_x_base_shape} to {self.images_filename}")

        # Save label data
        df_out = pd.DataFrame({k: data_dict[k] for k in data_dict.keys() if k != "image"})
        df_out.head(10).to_string(sys.stdout)  # print a sample
        df_out.to_feather(self.labels_filename)
        print(f"[DEBUG] Saved label data to {self.labels_filename} with shape {df_out.shape}")
        print(f"[DEBUG] label_data first few rows =>\n{df_out.head(5)}")

        with open(self.log_file_name, "w+") as log_file:
            log_file.write(f"total_dates:{sample_num} total_missing:{int(np.sum(data_miss))}\n")

        print(f"Saved image data to {self.images_filename}")
        print(f"Saved label data to {self.labels_filename}")

    def save_annual_ts_data(self) -> None:
        """
        Placeholder to generate 1D time-series data if needed. Not fully implemented in this example.
        """
        pass

    def _remove_old_files(self) -> None:
        """
        Remove old data files (log, label, images) to ensure a fresh start.
        """
        for fpath in [self.log_file_name, self.labels_filename, self.images_filename]:
            if op.isfile(fpath):
                print(f"[DEBUG] Removing old file {fpath}")
                os.remove(fpath)

    def _pre_generated_file_exists_and_valid(self) -> bool:
        """
        Checks if the log_file, label_file, and image_file exist, and if
        the image_file size is compatible with the expected shape.
        Returns False if any check fails, otherwise True.
        """
        if not (
            op.isfile(self.log_file_name)
            and op.isfile(self.labels_filename)
            and op.isfile(self.images_filename)
        ):
            return False

        # Check .dat file size as a multiple of (img_height * img_width)
        try:
            images_mem = np.memmap(self.images_filename, dtype=np.uint8, mode="r")
            total_size = images_mem.shape[0]
            expected_pixels_per_image = self._img_width * self._img_height
            if total_size % expected_pixels_per_image != 0:
                return False
        except Exception:
            # If any error reading the file, treat it as invalid
            return False

        return True

    @property
    def _img_width(self) -> int:
        """Return the image width based on chart_len."""
        return dcf.IMAGE_WIDTH[self.chart_len]

    @property
    def _img_height(self) -> int:
        """Return the image height based on chart_len (plus extra if volume_bar)."""
        base_h = dcf.IMAGE_HEIGHT[self.chart_len]
        if self.volume_bar:
            base_h += int(base_h / 5) + dcf.VOLUME_CHART_GAP
        return base_h

    def _generate_daily_features(
        self, stock_df: pd.DataFrame, date: pd.Timestamp
    ) -> Union[dict, int]:
        """
        Build the daily chart for one stock & date, returning a dict of features if successful.
        Otherwise return an int code for specific errors.

        Args:
            stock_df (pd.DataFrame): The single-stock data.
            date (pd.Timestamp): Date in that year for which we want the chart.

        Returns:
            dict: If successful, with "image" (PIL image) plus feature columns.
            int: If there's an error code from load_adjusted_daily_prices().
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

        last_day = df[df.Date == date].iloc[0]
        feature_dict = {col: last_day[col] for col in stock_df.columns if col in last_day}

        # Add classification/regression labels
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
        """
        For a given date, load the chunk of daily data to form an OHLC chart, adjusting price if needed.
        Return an int error code if something fails.

        Args:
            stock_df (pd.DataFrame): The DataFrame for a single stock.
            date (pd.Timestamp): Target date.

        Returns:
            int: An error code if not enough data or other problem.
            (pd.DataFrame, list): The final chunk plus local ma_lags used.
        """
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
        """
        Aggregate daily_df rows into blocks of chart_freq days (e.g., for chart_freq=4).
        Raises ChartGenerationError if length is not divisible by chart_freq.
        """
        length = len(daily_df)
        if length % self.chart_freq != 0:
            raise ChartGenerationError("df not divisible by chart_freq")

        ohlc_len = length // self.chart_freq
        out = pd.DataFrame(index=range(ohlc_len), columns=daily_df.columns)

        for i in range(ohlc_len):
            chunk = daily_df.iloc[i * self.chart_freq: (i + 1) * self.chart_freq]
            out.loc[i] = chunk.iloc[-1]
            out.loc[i, "Open"] = chunk.iloc[0]["Open"]
            out.loc[i, "High"] = chunk["High"].max()
            out.loc[i, "Low"] = chunk["Low"].min()
            out.loc[i, "Vol"] = chunk["Vol"].sum()
            out.loc[i, "Ret"] = np.prod(1 + np.array(chunk["Ret"])) - 1
        return out

    @staticmethod
    def adjust_price(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust daily OHLC so the first row has Close=1.0, subsequent days
        follow from daily returns. Raises ChartGenerationError if invalid.
        """
        if len(df) == 0:
            raise ChartGenerationError("Empty DataFrame in adjust_price.")
        if len(df.Date.unique()) != len(df):
            raise ChartGenerationError("Dates not unique in chunk for adjust_price.")

        df = df.reset_index(drop=True)
        fd_close = abs(df.at[0, "Close"])
        if (fd_close == 0.0) or pd.isna(fd_close):
            raise ChartGenerationError("First day close is zero/nan.")

        res_df = df.copy()
        res_df.at[0, "Close"] = 1.0
        res_df.at[0, "Open"] = abs(res_df.at[0, "Open"]) / fd_close
        res_df.at[0, "High"] = abs(res_df.at[0, "High"]) / fd_close
        res_df.at[0, "Low"] = abs(res_df.at[0, "Low"]) / fd_close
        pre_close = 1.0

        for i in range(1, len(res_df)):
            ret = float(res_df.at[i, "Ret"])
            this_close = (1 + ret) * pre_close
            orig_close = abs(res_df.at[i, "Close"])
            if orig_close == 0.0 or pd.isna(orig_close):
                continue
            res_df.at[i, "Close"] = this_close
            scale = this_close / orig_close
            res_df.at[i, "Open"] *= scale
            res_df.at[i, "High"] *= scale
            res_df.at[i, "Low"] *= scale
            res_df.at[i, "Ret"] = ret
            pre_close = this_close

        return res_df

    def _get_feature_and_dtype_list(self) -> Tuple[dict, list]:
        """
        Return (dtype_dict, feature_list) for memory-mapping arrays.

        Returns:
            (dict, list): {feature: dtype}, and the ordered list of feature names.
        """
        float32_features = [
            "EWMA_vol",
            "Ret",
            "Ret_tstat",
            "Ret_week",
            "Ret_month",
            "Ret_quarter",
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
