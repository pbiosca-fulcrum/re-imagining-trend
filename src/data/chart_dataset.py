"""
chart_dataset.py

Defines Torch Dataset classes for handling 2D CNN images (OHLC charts) or 1D CNN signals.

**Changes**:
- In load_image_np_data, added a clear check for file-size mismatch and a
  more helpful ValueError if we can't reshape the memmap to the expected dimensions.
- Added docstrings and small formatting improvements for clarity.
"""

from typing import Optional, List, Dict, Any
import os.path as op
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data import dgp_config as dcf
from src.data import equity_data as eqd
from src.utils import utilities as ut
from src.data.chart_library import DrawChartError


class EquityDataset(Dataset):
    """
    Torch Dataset for 2D CNN usage: loads pre-saved chart images and labels for a given year.
    """

    def __init__(
        self,
        window_size: int,
        predict_window: int,
        freq: str,
        year: int,
        country: str = "USA",
        has_volume_bar: bool = True,
        has_ma: bool = True,
        chart_type: str = "bar",
        annual_stocks_num: Any = "all",
        tstat_threshold: float = 0,
        stockid_filter: Optional[List[str]] = None,
        remove_tail: bool = False,
        ohlc_len: Optional[int] = None,
        regression_label: Optional[str] = None,
        delayed_ret: int = 0
    ) -> None:
        """
        Args:
            window_size (int): The number of days in each chart.
            predict_window (int): The horizon for next returns (e.g. 20 = monthly).
            freq (str): Data frequency ("week", "month", "quarter", "year").
            year (int): Target year of the data subset.
            country (str): "USA" or other region code.
            has_volume_bar (bool): Whether volume is included in the bottom sub-chart.
            has_ma (bool): Whether moving averages are drawn on the chart.
            chart_type (str): One of ["bar", "pixel", "centered_pixel"].
            annual_stocks_num (Any): "all" or an integer specifying top stocks by market cap.
            tstat_threshold (float): For optional filtering by t-stat percentile.
            stockid_filter (Optional[List[str]]): An explicit stock list filter.
            remove_tail (bool): Remove tail data if needed to avoid label leakage near year-end.
            ohlc_len (Optional[int]): Actual number of bars in each chart row. Usually matches window_size.
            regression_label (Optional[str]): "raw_ret" or "vol_adjust_ret" if regression-based, else None for classification.
            delayed_ret (int): Whether to use delayed returns (0-5).
        """
        self.ws = window_size
        self.pw = predict_window
        self.freq = freq
        self.year = year
        self.ohlc_len = ohlc_len if ohlc_len is not None else window_size
        self.country = country
        self.has_vb = has_volume_bar
        self.has_ma = has_ma
        self.chart_type = chart_type
        self.regression_label = regression_label
        self.delayed_ret = delayed_ret

        # Load images/labels
        self.images, self.label_dict = self.load_annual_data_by_country(self.country)

        # Decide on label name for next returns
        if self.country == "USA":
            self.ret_val_name = (
                f"Ret_{dcf.FREQ_DICT[self.pw]}"
                + ("" if delayed_ret == 0 else f"_{delayed_ret}delay")
            )
        else:
            self.ret_val_name = f"next_{dcf.FREQ_DICT[self.pw]}_ret_{delayed_ret}delay"

        # Convert raw returns to classification or regression
        self.label = self.get_label_value()

        # Filter data as needed
        self.filter_data(annual_stocks_num, stockid_filter, tstat_threshold, remove_tail)

        # Compute and store dataset mean/std for normalization
        self.demean = self._get_insample_mean_std()

    def get_image_label_save_path(self, country: str) -> tuple:
        """
        Return where images and labels are stored on disk for a given country/year.
        """
        save_dir = op.join(dcf.STOCKS_SAVEPATH, f"stocks_{country}", "dataset_all")
        dataset_name = self.__get_stock_dataset_name()
        img_save_path = op.join(save_dir, f"{dataset_name}_images.dat")
        label_path = op.join(save_dir, f"{dataset_name}_labels.feather")
        return img_save_path, label_path

    def load_annual_data_by_country(self, country: str) -> tuple:
        """
        Load pre-saved chart images and label data for a given country and year.
        """
        img_save_path, label_path = self.get_image_label_save_path(country)
        print(f"loading images from {img_save_path}")
        images = self.images = self.load_image_np_data(img_save_path,
                                      self.ohlc_len,
                                      self.has_vb)


        # Optionally rebuild the first image for demonstration
        # (Usually disabled, but kept here for completeness.)
        self.rebuild_image(
            images[0][0],
            image_name=self.__get_stock_dataset_name(),
            par_save_dir=op.dirname(img_save_path)
        )

        label_df = pd.read_feather(label_path)
        label_df["StockID"] = label_df["StockID"].astype(str)
        label_dict = {c: np.array(label_df[c]) for c in label_df.columns}
        return images, label_dict

    @staticmethod
    def rebuild_image(
        image: np.ndarray,
        image_name: str,
        par_save_dir: str,
        image_mode: str = "L"
    ) -> None:
        """
        Rebuild an image from raw bytes and save for verification.
        """
        from PIL import Image
        save_dir = ut.get_dir(op.join(par_save_dir, "images_rebuilt_from_dataset"))
        img = Image.fromarray(image, image_mode)
        img.save(op.join(save_dir, f"{image_name}.png"))

    @staticmethod
    def load_image_np_data(img_save_path: str,
                        ohlc_len: int,
                        volume_bar: bool) -> np.memmap:
        # compute final height the same way as generation
        base_h = dcf.IMAGE_HEIGHT[ohlc_len]
        if volume_bar:
            base_h += int(base_h / 5) + dcf.VOLUME_CHART_GAP

        width = dcf.IMAGE_WIDTH[ohlc_len]
        required_pixels = base_h * width

        images = np.memmap(img_save_path, dtype=np.uint8, mode="r")
        if images.size % required_pixels != 0:
            raise ValueError(
                f"Cannot reshape array of size {images.size} into shape "
                f"(-1, 1, {base_h}, {width}). File might be corrupted."
            )
        new_len = images.size // required_pixels
        images = images.reshape((new_len, 1, base_h, width))
        return images

    def __get_stock_dataset_name(self) -> str:
        """
        Return a consistent naming scheme for the dataset files based on chart configs.
        """
        chart_type_str = "" if self.chart_type == "bar" else f"{self.chart_type}_"
        vb_str = "has_vb" if self.has_vb else "no_vb"
        ma_str = f"[{self.ws}]_ma" if self.has_ma else "None_ma"
        data_freq = self.freq if self.ohlc_len == self.ws else "month"
        str_list = [
            f"{chart_type_str}{self.ws}d",
            data_freq,
            vb_str,
            ma_str,
            str(self.year)
        ]
        if self.ohlc_len != self.ws:
            str_list.append(f"{self.ohlc_len}ohlc")
        dataset_name = "_".join(str_list)
        return dataset_name

    def get_label_value(self) -> np.ndarray:
        """
        Convert raw returns to classification or regression label.
        """
        ret = self.label_dict[self.ret_val_name]
        if self.regression_label == "raw_ret":
            label = np.nan_to_num(ret, nan=-99)
        elif self.regression_label == "vol_adjust_ret":
            label = np.nan_to_num(ret / np.sqrt(self.label_dict["EWMA_vol"]), nan=-99)
        else:
            # Classification (up vs down)
            label = np.where(ret > 0, 1, 0)
            label = np.nan_to_num(label, nan=-99)
        return label

    def filter_data(
        self,
        annual_stocks_num: Any,
        stockid_filter: Optional[List[str]],
        tstat_threshold: float,
        remove_tail: bool
    ) -> None:
        """
        Filter the dataset based on user-specified thresholds, stock picks, or time tail removal.
        """
        df = pd.DataFrame({
            "StockID": self.label_dict["StockID"],
            "MarketCap": abs(self.label_dict["MarketCap"]),
            "Date": pd.to_datetime([str(t) for t in self.label_dict["Date"]])
        })

        if annual_stocks_num != "all":
            period_end_dates = eqd.get_period_end_dates(self.freq)
            stockids = self._pick_top_stocks_by_marketcap(df, period_end_dates, annual_stocks_num, stockid_filter)
        else:
            stockids = stockid_filter if stockid_filter else np.unique(df.StockID)

        stockid_idx = pd.Series(df.StockID).isin(stockids)
        idx = stockid_idx & pd.Series(self.label != -99) & pd.Series(self.label_dict["EWMA_vol"] != 0.0)

        if tstat_threshold != 0:
            tstats = np.abs(self.label_dict[self.ret_val_name] / self.label_dict["EWMA_vol"])
            t_th = np.nanpercentile(tstats[idx], tstat_threshold)
            tstat_idx = tstats > t_th
            idx = idx & tstat_idx

        if remove_tail:
            last_day = {
                5: "12/24", 20: "12/1", 60: "10/1"
            }.get(self.pw, "12/1")
            tail_date = pd.Timestamp(f"{last_day}/{self.year}")
            idx = idx & (pd.to_datetime([str(t) for t in self.label_dict["Date"]]) < tail_date)

        self.label = self.label[idx]
        for k in self.label_dict.keys():
            self.label_dict[k] = self.label_dict[k][idx]
        self.images = self.images[idx]
        self.label_dict["StockID"] = self.label_dict["StockID"].astype(str)
        self.label_dict["Date"] = self.label_dict["Date"].astype(str)

        assert len(self.label) == len(self.images)
        for k in self.label_dict.keys():
            assert len(self.images) == len(self.label_dict[k])

    def _pick_top_stocks_by_marketcap(
        self,
        df: pd.DataFrame,
        period_end_dates: pd.DatetimeIndex,
        annual_stocks_num: int,
        stockid_filter: Optional[List[str]]
    ) -> np.ndarray:
        """
        Select the top N stocks by market cap near mid-year, or fallback if not enough data.
        """
        num_stockid = len(np.unique(df.StockID))
        new_df = df
        # find date in June or so
        for _ in range(15):
            date_candidates = period_end_dates[
                (period_end_dates.year == self.year) & (period_end_dates.month == 6)
            ]
            if not len(date_candidates):
                break
            date = date_candidates[-1]
            new_df = df[df.Date == date]
            if len(np.unique(new_df.StockID)) > num_stockid / 2:
                break

        if stockid_filter:
            new_df = new_df[new_df.StockID.isin(stockid_filter)]

        new_df = new_df.sort_values(by=["MarketCap"], ascending=False)
        if len(new_df) > annual_stocks_num:
            stockids = new_df.iloc[:annual_stocks_num]["StockID"]
        else:
            stockids = new_df.StockID
        return stockids

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, idx: int) -> dict:
        image = (self.images[idx] / 255.0 - self.demean[0]) / self.demean[1]
        return {
            "image": image,
            "label": self.label[idx],
            "ret_val": self.label_dict[self.ret_val_name][idx],
            "ending_date": self.label_dict["Date"][idx],
            "StockID": self.label_dict["StockID"][idx],
            "MarketCap": self.label_dict["MarketCap"][idx]
        }

    def _get_insample_mean_std(self) -> list:
        """
        Compute or load the dataset's per-pixel mean/std (first 50k images for approximation).
        """
        ohlc_len_str = f"_{self.ohlc_len}ohlc" if self.ohlc_len != self.ws else ""
        chart_str = f"_{self.chart_type}" if self.chart_type != "bar" else ""
        fname = (
            f"mean_std_{self.ws}d{self.freq}_vb{self.has_vb}_ma{self.has_ma}_{self.year}"
            f"{ohlc_len_str}{chart_str}.npz"
        )
        mean_std_path = op.join(
            dcf.STOCKS_SAVEPATH,
            f"stocks_{self.country}",
            "dataset_all",
            fname
        )

        if op.exists(mean_std_path):
            x = np.load(mean_std_path, allow_pickle=True)
            return [x["mean"], x["std"]]

        mean = self.images[:50000].mean() / 255.0
        std = self.images[:50000].std() / 255.0
        np.savez(mean_std_path, mean=mean, std=std)
        return [mean, std]


class TS1DDataset(Dataset):
    """
    Torch Dataset for 1D CNN usage: loads pre-saved time-series data (open/high/low/close/ma/vol).
    """

    def __init__(
        self,
        window_size: int,
        predict_window: int,
        freq: str,
        year: int,
        country: str = "USA",
        remove_tail: bool = False,
        ohlc_len: Optional[int] = None,
        ts_scale: str = "image_scale",
        regression_label: Optional[str] = None
    ) -> None:
        self.ws = window_size
        self.pw = predict_window
        self.freq = freq
        self.year = year
        self.ohlc_len = ohlc_len if ohlc_len else window_size
        self.country = country
        self.remove_tail = remove_tail
        self.ts_scale = ts_scale
        self.regression_label = regression_label

        assert self.ts_scale in ["image_scale", "ret_scale", "vol_scale"]

        self.images, self.label_dict = self.load_ts1d_data()

        # Decide on label name
        self.ret_val_name = f"Retx_{dcf.FREQ_DICT[self.pw]}"
        self.label = self.get_label_value()

        # filter
        self.filter_data(self.remove_tail)

        # finalize normalization
        self.demean = self._get_1d_mean_std()

    def load_ts1d_data(self) -> tuple:
        """
        Load 1D time-series data for a given year/country from .npz files.
        """
        dataset_name = self.__get_stock_dataset_name()
        filename = op.join(
            dcf.STOCKS_SAVEPATH,
            "stocks_USA_ts/dataset_all/",
            f"{dataset_name}_data_new.npz"
        )
        data = np.load(filename, mmap_mode="r", encoding="latin1", allow_pickle=True)
        label_dict = data["data_dict"].item()
        images = label_dict["predictor"].copy()
        del label_dict["predictor"]
        label_dict["StockID"] = label_dict["StockID"].astype(str)
        return images, label_dict

    def __get_stock_dataset_name(self) -> str:
        base = f"{self.ws}d"
        data_freq = self.freq if self.ohlc_len == self.ws else "month"
        suffix = "ts"
        str_list = [base, data_freq, "has_vb", f"[{self.ws}]_ma", str(self.year)]
        if self.ohlc_len != self.ws:
            str_list.append(f"{self.ohlc_len}ohlc")
        str_list.append(suffix)
        return "_".join(str_list)

    def get_label_value(self) -> np.ndarray:
        """
        Convert raw returns to classification/regression label
        for 1D time-series model.
        """
        ret = self.label_dict[self.ret_val_name]
        if self.regression_label == "raw_ret":
            label = np.nan_to_num(ret, nan=-99)
        elif self.regression_label == "vol_adjust_ret":
            label = np.nan_to_num(ret / np.sqrt(self.label_dict["EWMA_vol"]), nan=-99)
        else:
            # binary classification: up vs down
            label = np.where(ret > 0, 1, 0)
            label = np.nan_to_num(label, nan=-99)
        return label

    def filter_data(self, remove_tail: bool) -> None:
        """
        Filter out invalid or tail data from the 1D dataset.
        """
        idx = (self.label != -99) & (self.label_dict["EWMA_vol"] != 0.0)
        if remove_tail:
            last_day_map = {5: "12/24", 20: "12/1", 60: "10/1"}
            last_day = last_day_map.get(self.pw, "12/1")
            tail_date = pd.Timestamp(f"{last_day}/{self.year}")
            date_arr = pd.to_datetime([str(t) for t in self.label_dict["Date"]])
            idx = idx & (date_arr < tail_date)

        self.label = self.label[idx]
        for k in self.label_dict.keys():
            self.label_dict[k] = self.label_dict[k][idx]
        self.images = self.images[idx]
        self.label_dict["StockID"] = self.label_dict["StockID"].astype(str)
        self.label_dict["Date"] = self.label_dict["Date"].astype(str)

        assert len(self.label) == len(self.images)
        for k in self.label_dict.keys():
            assert len(self.images) == len(self.label_dict[k])

    def _get_1d_mean_std(self) -> list:
        """
        Determine the channel-wise mean/std for the 1D time-series.
        """
        ohlc_len_str = f"_{self.ohlc_len}ohlc" if self.ohlc_len != self.ws else ""
        raw_suffix = (
            "" if self.ts_scale == "image_scale"
            else "_raw_price" if self.ts_scale == "ret_scale"
            else "_vol_scale"
        )
        fname = (
            f"mean_std_ts1d_{self.ws}d{self.freq}_vbTrue_maTrue_"
            f"{self.year}{ohlc_len_str}{raw_suffix}.npz"
        )
        mean_std_path = op.join(
            dcf.STOCKS_SAVEPATH,
            f"stocks_{self.country}_ts",
            "dataset_all",
            fname
        )

        if op.exists(mean_std_path):
            x = np.load(mean_std_path, allow_pickle=True)
            return [x["mean"], x["std"]]

        # apply transformations first if needed
        if self.ts_scale == "image_scale":
            for i in range(self.images.shape[0]):
                self.images[i] = self._minmax_scale_ts1d(self.images[i])
        elif self.ts_scale == "vol_scale":
            for i in range(self.images.shape[0]):
                self.images[i] = self._vol_scale_ts1d(self.images[i]) / np.sqrt(self.label_dict["EWMA_vol"][i])

        mean = np.nanmean(self.images, axis=(0, 2))
        std = np.nanstd(self.images, axis=(0, 2))
        np.savez(mean_std_path, mean=mean, std=std)
        return [mean, std]

    def _minmax_scale_ts1d(self, image: np.ndarray) -> np.ndarray:
        """
        Scale the input channels by min-max scaling to [0, 1].
        """
        out = image.copy()
        ohlcma = out[:5]
        rng_1 = np.nanmax(ohlcma) - np.nanmin(ohlcma)
        if rng_1 != 0:
            out[:5] = (ohlcma - np.nanmin(ohlcma)) / rng_1

        rng_2 = np.nanmax(out[5]) - np.nanmin(out[5])
        if rng_2 != 0:
            out[5] = (out[5] - np.nanmin(out[5])) / rng_2
        return out

    def _vol_scale_ts1d(self, image: np.ndarray) -> np.ndarray:
        """
        Convert absolute prices to approximate returns by dividing consecutive columns.
        """
        out = image.copy()
        out[:, 0] = 0
        for i in range(1, 5):
            out[:, i] = image[:, i] / image[0, i - 1] - 1
        return out

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, idx: int) -> dict:
        image = self.images[idx].copy()
        # re-check scale if needed
        if self.ts_scale == "image_scale":
            image = self._minmax_scale_ts1d(image)
        elif self.ts_scale == "vol_scale":
            image = self._vol_scale_ts1d(image) / np.sqrt(self.label_dict["EWMA_vol"][idx])

        image = (image - self.demean[0].reshape(6, 1)) / self.demean[1].reshape(6, 1)
        image = np.nan_to_num(image, nan=0, posinf=0, neginf=0)

        return {
            "image": image,
            "label": self.label[idx],
            "ret_val": self.label_dict[self.ret_val_name][idx],
            "ending_date": self.label_dict["Date"][idx],
            "StockID": self.label_dict["StockID"][idx],
            "MarketCap": self.label_dict["MarketCap"][idx]
        }
