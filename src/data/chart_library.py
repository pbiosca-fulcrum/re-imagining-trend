"""
chart_library.py

Contains logic to draw OHLC charts as images (bar, pixel, or centered_pixel).
These images feed into the CNN model as training data.
"""

import math
import numpy as np
from PIL import Image, ImageDraw

from src.data import dgp_config as dcf


class DrawChartError(Exception):
    """Custom Exception for chart drawing errors."""
    pass


class DrawOHLC:
    """
    A class that takes in a small DataFrame chunk of OHLC data and
    renders it as an image for CNN processing.
    """

    def __init__(
        self,
        df,
        has_volume_bar: bool = False,
        ma_lags=None,
        chart_type: str = "bar"
    ) -> None:
        if round(df.iloc[0]["Close"], 3) != 1.000:
            raise DrawChartError("Close on first day not equal to 1.")
        self.has_volume_bar = has_volume_bar
        self.vol = df["Vol"] if has_volume_bar else None
        self.ma_lags = ma_lags
        self.ma_name_list = [f"ma{ml}" for ml in (ma_lags or [])]
        self.chart_type = chart_type
        assert self.chart_type in ["bar", "pixel", "centered_pixel"]

        if self.chart_type == "centered_pixel":
            self.df = self.centered_prices(df)
        else:
            self.df = df[["Open", "High", "Low", "Close"] + self.ma_name_list].abs()

        self.ohlc_len = len(df)
        assert self.ohlc_len in [5, 20, 60]
        self.minp = self.df.min().min()
        self.maxp = self.df.max().max()

        self.ohlc_width = dcf.IMAGE_WIDTH[self.ohlc_len]
        self.ohlc_height = dcf.IMAGE_HEIGHT[self.ohlc_len]
        self.volume_height = int(self.ohlc_height / 5) if self.has_volume_bar else 0

        first_center = (dcf.BAR_WIDTH - 1) / 2.0
        self.centers = np.arange(
            first_center,
            first_center + dcf.BAR_WIDTH * self.ohlc_len,
            dcf.BAR_WIDTH,
            dtype=int
        )

    def centered_prices(self, df) -> np.ndarray:
        """
        For the 'centered_pixel' approach, recenter the prices around today's close.
        """
        cols = ["Open", "High", "Low", "Close", "Prev_Close"] + self.ma_name_list
        new_df = df[cols].copy()
        new_df[cols] = new_df[cols].div(df["Close"], axis=0)  # relative to close
        new_df[cols] = new_df[cols].sub(df["Close"], axis=0)
        new_df.loc[new_df.index != 0, self.ma_name_list] = 0
        return new_df

    def draw_image(self, pattern_list=None) -> Image.Image:
        """
        Main function to draw either bar, pixel, or centered_pixel chart as grayscale.
        """
        if (self.maxp == self.minp) or math.isnan(self.maxp) or math.isnan(self.minp):
            return None
        try:
            if self.__ret_to_yaxis(self.minp) != 0:
                pass  # ignoring strict assertion
        except ValueError:
            return None

        if self.chart_type == "centered_pixel":
            ohlc = self.__draw_centered_pixel_chart()
        else:
            ohlc = self.__draw_ohlc()

        if self.vol is not None:
            volume_bar = self.__draw_vol()
            image = Image.new(
                "L",
                (
                    self.ohlc_width,
                    self.ohlc_height + self.volume_height + dcf.VOLUME_CHART_GAP
                )
            )
            image.paste(ohlc, (0, self.volume_height + dcf.VOLUME_CHART_GAP))
            image.paste(volume_bar, (0, 0))
        else:
            image = ohlc

        # optional patterns
        if pattern_list is not None:
            draw = ImageDraw.Draw(image)
            cur_day = 0
            for pat, length in pattern_list:
                if pat is None:
                    cur_day += length
                else:
                    draw.line([
                        self.centers[cur_day],
                        0,
                        self.centers[cur_day],
                        self.ohlc_height - 1
                    ], fill=dcf.CHART_COLOR)
                    cur_day += length
                    if cur_day < self.ohlc_len:
                        draw.line([
                            self.centers[cur_day],
                            0,
                            self.centers[cur_day],
                            self.ohlc_height - 1
                        ], fill=dcf.CHART_COLOR)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image

    def __ret_to_yaxis(self, val: float) -> int:
        """
        Convert a price-level to pixel coordinate.
        """
        pixels_per_unit = (self.ohlc_height - 1.0) / (self.maxp - self.minp)
        return int(round((val - self.minp) * pixels_per_unit))

    def __draw_vol(self) -> Image.Image:
        """
        Draw volume sub-chart below the main OHLC chart.
        """
        from PIL import Image, ImageDraw
        volume_bar = Image.new("L", (self.ohlc_width, self.volume_height), dcf.BACKGROUND_COLOR)
        pixels = volume_bar.load()
        max_volume = np.max(self.vol.abs())
        if not np.isnan(max_volume) and max_volume != 0:
            pixels_per_volume = self.volume_height / max_volume
            draw = ImageDraw.Draw(volume_bar)
            for day in range(self.ohlc_len):
                if np.isnan(self.vol.iloc[day]):
                    continue
                vol_height = int(round(abs(self.vol.iloc[day]) * pixels_per_volume))
                if self.chart_type == "bar":
                    draw.line([
                        self.centers[day], 0,
                        self.centers[day], vol_height - 1
                    ], fill=dcf.CHART_COLOR)
                else:  # "pixel" or "centered_pixel"
                    pixels[self.centers[day], vol_height - 1] = dcf.CHART_COLOR
        return volume_bar

    def __draw_ohlc(self) -> Image.Image:
        """
        Draw standard bar or pixel chart with open/high/low/close.
        """
        from PIL import Image, ImageDraw
        ohlc = Image.new("L", (self.ohlc_width, self.ohlc_height), dcf.BACKGROUND_COLOR)
        pixels = ohlc.load()

        # draw moving averages
        for ma_col in self.ma_name_list:
            draw = ImageDraw.Draw(ohlc)
            for day in range(self.ohlc_len - 1):
                if not np.isnan(self.df[ma_col].iloc[day]) and not np.isnan(self.df[ma_col].iloc[day + 1]):
                    if self.chart_type == "bar":
                        draw.line((
                            self.centers[day], self.__ret_to_yaxis(self.df[ma_col].iloc[day]),
                            self.centers[day + 1], self.__ret_to_yaxis(self.df[ma_col].iloc[day + 1])
                        ), width=1, fill=dcf.CHART_COLOR)
                    else:  # "pixel"
                        pixels[self.centers[day], self.__ret_to_yaxis(self.df[ma_col].iloc[day])] = dcf.CHART_COLOR
            # last day
            last_px = self.__ret_to_yaxis(self.df[ma_col].iloc[self.ohlc_len - 1])
            if 0 <= last_px < self.ohlc_height:
                pixels[self.centers[self.ohlc_len - 1], last_px] = dcf.CHART_COLOR

        # now the OHLC bars
        for day in range(self.ohlc_len):
            highp = self.df["High"].iloc[day]
            lowp = self.df["Low"].iloc[day]
            closep = self.df["Close"].iloc[day]
            openp = self.df["Open"].iloc[day]
            if np.isnan(highp) or np.isnan(lowp):
                continue
            left = int(math.ceil(self.centers[day] - dcf.BAR_WIDTH / 2))
            right = int(math.floor(self.centers[day] + dcf.BAR_WIDTH / 2))

            line_left = int(math.ceil(self.centers[day] - dcf.LINE_WIDTH / 2))
            line_right = int(math.floor(self.centers[day] + dcf.LINE_WIDTH / 2))

            line_bottom = self.__ret_to_yaxis(lowp)
            line_up = self.__ret_to_yaxis(highp)

            if self.chart_type == "bar":
                for i in range(line_left, line_right + 1):
                    for j in range(line_bottom, line_up + 1):
                        pixels[i, j] = dcf.CHART_COLOR
            else:  # "pixel"
                pixels[self.centers[day], line_bottom] = dcf.CHART_COLOR
                pixels[self.centers[day], line_up] = dcf.CHART_COLOR

            if not np.isnan(openp):
                open_line = self.__ret_to_yaxis(openp)
                for i in range(left, self.centers[day] + 1):
                    pixels[i, open_line] = dcf.CHART_COLOR

            if not np.isnan(closep):
                close_line = self.__ret_to_yaxis(closep)
                for i in range(self.centers[day] + 1, right + 1):
                    pixels[i, close_line] = dcf.CHART_COLOR

        return ohlc

    def __draw_centered_pixel_chart(self) -> Image.Image:
        """
        A specialized chart type where everything is recentered around the close, drawn as single-pixel lines.
        """
        from PIL import Image
        ohlc = Image.new("L", (self.ohlc_width, self.ohlc_height), dcf.BACKGROUND_COLOR)
        pixels = ohlc.load()

        for day in range(self.ohlc_len):
            highp = self.df["High"].iloc[day]
            lowp = self.df["Low"].iloc[day]
            prev_closep = self.df["Prev_Close"].iloc[day]
            openp = self.df["Open"].iloc[day]
            if np.isnan(highp) or np.isnan(lowp):
                continue
            pixels[self.centers[day], self.__ret_to_yaxis(highp)] = dcf.CHART_COLOR
            pixels[self.centers[day], self.__ret_to_yaxis(lowp)] = dcf.CHART_COLOR

            left = int(math.ceil(self.centers[day] - dcf.BAR_WIDTH / 2))
            right = int(math.floor(self.centers[day] + dcf.BAR_WIDTH / 2))

            if not np.isnan(openp):
                open_line = self.__ret_to_yaxis(openp)
                for i in range(left, self.centers[day] + 1):
                    pixels[i, open_line] = dcf.CHART_COLOR

            if not np.isnan(prev_closep):
                prev_close_line = self.__ret_to_yaxis(prev_closep)
                for i in range(left, right + 1):
                    pixels[i, prev_close_line] = dcf.CHART_COLOR

        # optionally could also draw ma lines
        return ohlc
