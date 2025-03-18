"""
dgp_config.py

Holds global constants for data generation and chart images:
paths, frequency mappings, image sizes, etc.
"""

import os
import os.path as op
from pathlib import Path
from src.utils.config import WORK_DIR


def get_dir(dir_path: str) -> str:
    """
    Ensure directory existence, create if missing, and return the path.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path


DATA_DIR = get_dir(op.join(WORK_DIR, "data"))
PROCESSED_DATA_DIR = get_dir(op.join(DATA_DIR, "processed_data"))
STOCKS_SAVEPATH = os.path.join(DATA_DIR, "stocks_dataset")
RAW_DATA_DIR = op.join(STOCKS_SAVEPATH, "raw_data")

print(f"DATA_DIR: {DATA_DIR}")
print(f"PROCESSED_DATA_DIR: {PROCESSED_DATA_DIR}")
print(f"STOCKS_SAVEPATH: {STOCKS_SAVEPATH}")
print(f"RAW_DATA_DIR: {RAW_DATA_DIR}")

CACHE_DIR = Path("../CACHE_DIR")
PORTFOLIO = Path("../CACHE_DIR/PORTFOLIO")
if not os.path.isdir(PORTFOLIO):
    os.makedirs(PORTFOLIO, exist_ok=True)

BAR_WIDTH = 3
LINE_WIDTH = 1
VOLUME_CHART_GAP = 1
BACKGROUND_COLOR = 0
CHART_COLOR = 255

# For 2D chart images
IMAGE_WIDTH = {5: BAR_WIDTH * 5, 20: BAR_WIDTH * 20, 60: BAR_WIDTH * 60}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}

FREQ_DICT = {
    5: "week",
    20: "month",
    60: "quarter",
    65: "quarter",  # used in older code
    260: "year"
}

# Potential set of international countries for expansions
INTERNATIONAL_COUNTRIES = [
    "Japan", "UnitedKingdom", "China", "SouthKorea", "India",
    "Canada", "Germany", "Australia", "HongKong", "France", "Singapore",
    "Italy", "Sweden", "Switzerland", "Netherlands", "Norway", "Spain",
    "Belgium", "Greece", "Denmark", "Russia", "Finland", "NewZealand",
    "Austria", "Portugal", "Ireland"
]
