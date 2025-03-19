"""
utilities.py

General-purpose helper functions for the entire project.
"""

import math
import os
import numpy as np
import pandas as pd
import torch
import pickle as pickle


two_sided_tstat_threshold_dict = {0.1: 1.645, 0.05: 1.96, 0.01: 2.575}
one_sided_tstat_threshold_dict = {0.1: 1.28, 0.05: 1.645, 0.01: 2.33}


def get_dir(dir_path: str) -> str:
    """
    Create and return a directory path if it doesn't exist.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def df_empty(columns, dtypes, index=None):
    """
    Create an empty DataFrame with the specified columns/dtypes.
    """
    assert len(columns) == len(dtypes)
    df = pd.DataFrame(index=index)
    for c, d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df


def binary_one_hot(y, device=None):
    """
    Convert a set of class labels into one-hot encoded form for 2 classes.
    """
    y = y.to("cpu")
    y_onehot = torch.FloatTensor(y.shape[0], 2)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    y_onehot = y_onehot.to(y.dtype)
    if device:
        y_onehot = y_onehot.to(device)
    return y_onehot


def cross_entropy_loss(pred_prob, true_label):
    """
    Manual cross-entropy for arrays of shape (n_samples,).
    """
    pred_prob = np.array(pred_prob)
    x = np.zeros((len(pred_prob), 2))
    x[:, 1] = pred_prob
    x[:, 0] = 1 - x[:, 1]

    true_label = np.array(true_label)
    y = np.zeros((len(true_label), 2))
    y[np.arange(true_label.size), true_label] = 1

    loss = -np.sum(y * np.log(x)) / len(pred_prob)
    return loss


def rank_corr(df: pd.DataFrame, col1: str, col2: str, method: str = "spearman") -> float:
    """
    Return correlation between two columns, optionally using rank-based approach.
    """
    if method == "spearman":
        c1 = df[col1].rank(method="average", ascending=False)
        c2 = df[col2].rank(method="average", ascending=False)
        return c2.corr(c1, method="spearman")
    return df[col1].corr(df[col2], method=method)


def rank_normalization(c: pd.Series) -> pd.Series:
    """
    Map series into [-1, 1] by rank.
    """
    r = c.rank(ascending=True)
    return 2.0 * (r - r.min()) / (r.max() - r.min()) - 1.0


def calculate_test_log(pred_prob, label):
    """
    Evaluate predictions vs. true labels to get cross-entropy, MCC, accuracy, etc.
    """
    pred = np.where(pred_prob > 0.5, 1, 0)
    n = len(pred)
    TP = np.nansum(pred * label) / n
    TN = np.nansum((pred - 1) * (label - 1)) / n
    FP = abs(np.nansum(pred * (label - 1))) / n
    FN = abs(np.nansum((pred - 1) * label)) / n
    test_log = {
        "diff": (TP + FP) - (TN + FN),
        "loss": cross_entropy_loss(pred_prob, label),
        "accy": TP + TN
    }
    denom = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    if denom == 0:
        test_log["MCC"] = np.nan
    else:
        test_log["MCC"] = (TP * TN - FP * FN) / math.sqrt(denom)
    return test_log


def save_pkl_obj(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pkl_obj(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def to_latex_w_turnover(pf_df: pd.DataFrame, cut: int = 10) -> str:
    """
    Convert a  decile portfolio result DataFrame to a latex block, including turnover as an extra row or note.
    """
    pf_df = pf_df.rename(columns={"ret": "Ret", "std": "Std"})
    pf_df = pf_df.round(3)
    pf_df = pf_df.set_index(
        pd.Index(["Low"] + list(range(2, int(cut))) + ["High", "H-L", "Turnover"])
    )
    latex = (pf_df.iloc[: cut + 1]).to_latex()
    # add a line for turnover
    latex_list = latex.splitlines()
    latex_list.insert(len(latex_list) - 2, "\\hline")
    line = f"\\multicolumn{{4}}{{c}}{{Turnover: {int(pf_df.loc['Turnover', 'SR']*100)}\\%}}\\\\"
    latex_list.insert(len(latex_list) - 2, line)
    return "\n".join(latex_list)


def pvalue_surfix(pv: float) -> str:
    """
    Return star string based on pvalue significance.
    """
    if pv < 0.01:
        return "***"
    if pv < 0.05:
        return "**"
    if pv < 0.1:
        return "*"
    return ""


def add_star_by_pvalue(value: float, pvalue: float, decimal: int = 2) -> str:
    return f"{value:.{decimal}f}" + pvalue_surfix(pvalue)


def star_significant_value_by_sample_num(val: float, sample_num: int, one_sided: bool = True, decimal: int = 2) -> str:
    """
    Provide a star-labeled string based on t-stat significance with approximate z thresholds.
    """
    tstat = val * math.sqrt(sample_num)
    return add_stars_to_value_by_tstat(val, tstat, one_sided, decimal)


def add_stars_to_value_by_tstat(value: float, tstat: float, one_sided: bool, decimal: int) -> str:
    t = abs(tstat)
    if one_sided:
        if tstat > one_sided_tstat_threshold_dict[0.01]:
            return f"{value:.{decimal}f}***"
        elif tstat > one_sided_tstat_threshold_dict[0.05]:
            return f"{value:.{decimal}f}**"
        elif tstat > one_sided_tstat_threshold_dict[0.1]:
            return f"{value:.{decimal}f}*"
        return f"{value:.{decimal}f}"
    else:
        if t > two_sided_tstat_threshold_dict[0.01]:
            return f"{value:.{decimal}f}***"
        elif t > two_sided_tstat_threshold_dict[0.05]:
            return f"{value:.{decimal}f}**"
        elif t > two_sided_tstat_threshold_dict[0.1]:
            return f"{value:.{decimal}f}*"
        return f"{value:.{decimal}f}"
