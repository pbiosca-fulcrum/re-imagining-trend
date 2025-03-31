"""
chart_dataset.py

Defines a Torch Dataset for 2D CNN images (or 1D signals). Here we've modified it
to expect each chart image as an individual PNG file referenced in the label Feather,
matching the approach in generate_chart.py after the new changes.
"""

from typing import Optional, List
import os.path as op
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from src.data import dgp_config as dcf
from src.data import equity_data as eqd
from src.utils import utilities as ut
from src.data.chart_library import DrawChartError


class EquityDataset(Dataset):
    """
    Torch Dataset for 2D CNN usage: it now loads one PNG image per sample, reading
    the "image_path" column from the label DataFrame.

    The old memory-mapped approach has been removed. Instead, each chart image
    is stored individually under `images_rebuilt_from_dataset/`.
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
        annual_stocks_num: any = "all",
        tstat_threshold: float = 0,
        stockid_filter: Optional[List[str]] = None,
        remove_tail: bool = False,
        ohlc_len: Optional[int] = None,
        regression_label: Optional[str] = None,
        delayed_ret: int = 0
    ) -> None:
        """
        Args:
            window_size: The number of days in each chart.
            predict_window: The horizon for next returns (5=weekly,20=monthly, etc.).
            freq: "week","month","quarter","year".
            year: The target year of data.
            country: Usually "USA".
            has_volume_bar: If True, each chart has a volume sub-chart.
            has_ma: If True, each chart includes an MA overlay.
            chart_type: "bar", "pixel", or "centered_pixel".
            annual_stocks_num: "all" or integer top by market cap.
            tstat_threshold: optional filter by t-stat percentile (0 => no filter).
            stockid_filter: optional filter for a specific stock list.
            remove_tail: remove tail data near year-end if True.
            ohlc_len: The actual chart's bar count. Usually same as window_size.
            regression_label: If not None, do regression ("raw_ret", "vol_adjust_ret"), else classification.
            delayed_ret: If you want delayed returns (0..5).
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

        # E.g., for predict_window=5 and freq="week", ret_val_name="next_week_ret_0delay"
        self.ret_val_name = f"next_{dcf.FREQ_DICT[self.pw]}_ret_{delayed_ret}delay"

        # 1) Load label data (Feather file) which has "image_path" for each row
        self.labels_df = self.load_annual_data_labels()

        # 2) Convert "Ret" column into classification or regression label
        self.label = self.get_label_value()

        # 3) Filter the dataset based on user-specified thresholds, stock picks, or time tail removal
        self.filter_data(
            annual_stocks_num=annual_stocks_num,
            stockid_filter=stockid_filter,
            tstat_threshold=tstat_threshold,
            remove_tail=remove_tail
        )

        # 4) Compute dataset-level mean/std for normalization
        self.demean = self._get_insample_mean_std()

    def load_annual_data_labels(self) -> pd.DataFrame:
        """
        Load the label Feather file that references each PNG's path.
        We only keep rows for the specified year in question.
        """
        label_path = self._get_label_feather_path()
        if not op.isfile(label_path):
            raise FileNotFoundError(f"No label file found at {label_path}")

        df = pd.read_feather(label_path)
        df["StockID"] = df["StockID"].astype(str)
        # Some rows might not correspond to the exact year if we do multi-year generation;
        # so keep only the target year.
        df = df[df["Date"].dt.year == self.year]
        df = df.reset_index(drop=True)
        return df

    def _get_label_feather_path(self) -> str:
        """
        Construct the label Feather path based on chart configs.
        """
        chart_type_str = "" if self.chart_type == "bar" else f"{self.chart_type}_"
        vb_str = "has_vb" if self.has_vb else "no_vb"
        # If you specifically track the MA lags in the filename, adapt as needed:
        ma_str = "[5]"
        freq_str = self.freq if self.ohlc_len == self.ws else "month"
        dataset_name = (
            f"{chart_type_str}{self.ws}d_{freq_str}_{vb_str}_{ma_str}_ma_{self.year}"
        )
        if self.ohlc_len != self.ws:
            dataset_name += f"_{self.ohlc_len}ohlc"

        labels_filename = op.join(
            dcf.STOCKS_SAVEPATH,
            f"stocks_{self.country}",
            "dataset_all",
            f"{dataset_name}_labels.feather"
        )
        return labels_filename

    def get_label_value(self) -> np.ndarray:
        """
        Convert raw returns to classification or regression label.
        We rely on self.ret_val_name (like "next_week_ret_0delay").
        """
        # If that column is missing, fallback to zeros and warn
        if self.ret_val_name not in self.labels_df.columns:
            print(f"[WARN] The column '{self.ret_val_name}' is missing. Setting label=0.0 .")
            return np.zeros(len(self.labels_df), dtype=np.float32)

        ret_array = self.labels_df[self.ret_val_name].values
        if self.regression_label is None:
            # classification => label is 0 or 1
            return np.where(ret_array > 0, 1, 0).astype(np.int8)
        if self.regression_label == "raw_ret":
            return ret_array.astype(np.float32)
        if self.regression_label == "vol_adjust_ret":
            vol_arr = self.labels_df.get("EWMA_vol", pd.Series([np.nan]*len(ret_array))).values
            vol_arr = np.where((vol_arr == 0) | np.isnan(vol_arr), 1e9, vol_arr)  # avoid div0
            return (ret_array / vol_arr).astype(np.float32)
        # fallback
        return ret_array.astype(np.float32)

    def filter_data(
        self,
        annual_stocks_num: any,
        stockid_filter: Optional[List[str]],
        tstat_threshold: float,
        remove_tail: bool
    ) -> None:
        """
        Filter the data based on user constraints, removing or adjusting rows in self.labels_df.
        """
        df = self.labels_df.copy()
        # Filter by valid label
        # If "EWMA_vol" is missing, default 1 so it's not zero. Then we check isfinite
        keep_mask = np.isfinite(self.label)
        if "EWMA_vol" in df.columns:
            keep_mask &= df["EWMA_vol"] != 0

        df = df[keep_mask]
        new_label = self.label[keep_mask]

        # If we want top N by marketcap:
        if annual_stocks_num != "all":
            top_stock_list = self._pick_top_stocks_by_marketcap(
                df, annual_stocks_num, stockid_filter
            )
            df = df[df["StockID"].isin(top_stock_list)]
            new_label = new_label[df.index]
        elif stockid_filter is not None:
            # Just filter by stock list
            df = df[df["StockID"].isin(stockid_filter)]
            new_label = new_label[df.index]

        # If we want a tstat threshold
        if tstat_threshold != 0:
            tstats = df.get(self.ret_val_name, pd.Series([0]*len(df))) / df.get("EWMA_vol", 1)
            threshold_val = np.nanpercentile(np.abs(tstats), tstat_threshold)
            keep_idx = np.abs(tstats) > threshold_val
            df = df[keep_idx]
            new_label = new_label[keep_idx]

        # If remove_tail
        if remove_tail:
            tail_date_map = {5: "12/24", 20: "12/1", 60: "10/1"}
            last_day = tail_date_map.get(self.pw, "12/1")
            cut_off = pd.Timestamp(f"{last_day}/{self.year}")
            keep_idx = df["Date"] < cut_off
            df = df[keep_idx]
            new_label = new_label[keep_idx]

        # Now reset the index on the DataFrame. This is safe.
        df = df.reset_index(drop=True)

        # new_label is a NumPy array, so we do NOT call .reset_index() on it
        self.label = new_label
        self.labels_df = df

    @staticmethod
    def _pick_top_stocks_by_marketcap(
        df: pd.DataFrame, annual_stocks_num: int, stockid_filter: Optional[List[str]]
    ) -> List[str]:
        """
        Return a list of StockIDs for the top 'annual_stocks_num' by MarketCap
        (optionally filtered by stockid_filter if provided).
        """
        midyear = df[df["Date"].dt.month == 6]
        if stockid_filter:
            midyear = midyear[midyear["StockID"].isin(stockid_filter)]
        midyear = midyear.sort_values(by="MarketCap", ascending=False)
        unique_ids = midyear["StockID"].unique().tolist()
        if len(unique_ids) > annual_stocks_num:
            return unique_ids[:annual_stocks_num]
        return unique_ids

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> dict:
        """
        Return a single sample as a dict with:
          - 'image': (C,H,W) normalized tensor,
          - 'label': the classification/regression label,
          - 'ret_val': the raw return or the base 'Ret' column,
          - 'ending_date': the row's date,
          - 'StockID': the row's stock,
          - 'MarketCap': the row's MktCap.
        """
        row = self.labels_df.iloc[idx]
        # Load the PNG from image_path
        image_path = row["image_path"]
        with Image.open(image_path) as pil_img:
            # Convert to grayscale
            pil_img = pil_img.convert("L")
            image_np = np.array(pil_img, dtype=np.uint8)

        # Convert to float, normalize to [0..1], then standardize
        image_float = image_np.astype(np.float32) / 255.0
        image_float = (image_float - self.demean[0]) / (self.demean[1] + 1e-12)

        # Add channel dimension => shape (1, H, W) for CNN
        image_tensor = torch.from_numpy(image_float).unsqueeze(0)

        out = {
            "image": image_tensor,
            "label": self.label[idx],
            "ret_val": row.get(self.ret_val_name, 0.0),
            "ending_date": row["Date"],
            "StockID": row["StockID"],
            "MarketCap": row.get("MarketCap", 0.0)
        }
        return out

    def _get_insample_mean_std(self) -> List[float]:
        """
        Approximate the dataset's per-pixel mean/std by sampling up to 5000 images.
        """
        n_samples = min(5000, len(self.labels_df))
        if n_samples == 0:
            return [0.0, 1.0]  # fallback

        pixel_vals = []
        for i in range(n_samples):
            row = self.labels_df.iloc[i]
            image_path = row["image_path"]
            with Image.open(image_path) as pil_img:
                img_np = np.array(pil_img.convert("L"), dtype=np.float32)
                pixel_vals.append(img_np.ravel())

        all_pixels = np.concatenate(pixel_vals)
        mean_val = all_pixels.mean() / 255.0
        std_val = all_pixels.std() / 255.0
        return [float(mean_val), float(std_val)]


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
