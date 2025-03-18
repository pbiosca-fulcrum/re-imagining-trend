"""
config.py

Houses global config constants used across the project, such as directory paths and default hyperparameters.
"""

import os


def get_dir(dir_path: str) -> str:
    """
    Ensure the directory exists, create if needed, and return its path.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path


# Project structure
# WORK_DIR = get_dir("../WORK_SPACE")
WORK_DIR = get_dir(".")
EXP_DIR = get_dir(os.path.join(WORK_DIR, "new_model_res"))
PORTFOLIO_DIR = get_dir(os.path.join(EXP_DIR, "portfolio"))
LOG_DIR = get_dir(os.path.join(EXP_DIR, "log"))
LATEX_DIR = get_dir(os.path.join(EXP_DIR, "latex"))
LIGHT_CSV_RES_DIR = get_dir(os.path.join(WORK_DIR, "torch_ta/ta/csv_res/"))

# Default hyperparameters
BATCH_SIZE = 128
NUM_WORKERS = 1
TRUE_DATA_CNN_INPLANES = 64

# For 2D CNN model
BENCHMARK_MODEL_LAYERNUM_DICT = {5: 2, 20: 3, 60: 4}
EMP_CNN_BL_SETTING = {
    5: ([(5, 3)] * 10, [(1, 1)] * 10, [(1, 1)] * 10, [(2, 1)] * 10),
    20: ([(5, 3)] * 10, [(3, 1)] + [(1, 1)] * 9, [(2, 1)] + [(1, 1)] * 9, [(2, 1)] * 10),
    60: ([(5, 3)] * 10, [(3, 1)] + [(1, 1)] * 9, [(3, 1)] + [(1, 1)] * 9, [(2, 1)] * 10),
}

# For 1D CNN model
TS1D_LAYERNUM_DICT = {5: 1, 20: 2, 60: 3}
EMP_CNN1d_BL_SETTING = {
    5: ([3], [1], [1], [2]),
    20: ([3, 3], [1, 1], [1, 1], [2, 2]),
    60: ([3, 3, 3], [1, 1, 1], [1, 1, 1], [2, 2, 2]),
}

# For splitting sample by year
IS_YEARS = list(range(1993, 2001))
OOS_YEARS = list(range(2001, 2020))

# Benchmark model naming (for referencing in experiment code)
BENCHMARK_MODEL_NAME_DICT = {
    5: "D5L2F53S1F53S1C64MP11",
    20: "D20L3F53S1F53S1F53S1C64MP111",
    60: "D60L4F53S1F53S1F53S1F53S1C64MP1111",
}
