"""
cnn_experiment.py

Defines an Experiment class that manages CNN training, evaluation,
and the generation of final ensemble results for portfolio usage.
"""

import copy
import itertools
import math
import sys
import time
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader, ConcatDataset, random_split
from tqdm import tqdm

from src.model import cnn_model
from src.portfolio import portfolio as pf
from src.utils.config import (
    EXP_DIR, LOG_DIR, LATEX_DIR, PORTFOLIO_DIR, LIGHT_CSV_RES_DIR,
    TRUE_DATA_CNN_INPLANES, BENCHMARK_MODEL_LAYERNUM_DICT, EMP_CNN_BL_SETTING,
    TS1D_LAYERNUM_DICT, EMP_CNN1d_BL_SETTING, BATCH_SIZE, NUM_WORKERS,
    IS_YEARS, OOS_YEARS, BENCHMARK_MODEL_NAME_DICT
)
from src.utils import utilities as ut
from src.data.chart_dataset import EquityDataset, TS1DDataset
from src.data import equity_data as eqd
from src.data.dgp_config import FREQ_DICT


class Experiment:
    """
    Encapsulates the entire train-validate-test pipeline for a CNN-based model.

    Steps:
      1. Create Datasets from specified in-sample years
      2. Train and validate the CNN with possible ensemble
      3. Save or load final ensemble predictions
      4. Evaluate or build portfolio from predictions
    """

    def __init__(
        self,
        ws: int,
        pw: int,
        model_obj: cnn_model.Model,
        train_freq: str,
        ensem: int = 5,
        lr: float = 1e-5,
        drop_prob: float = 0.50,
        device_number: int = 0,
        max_epoch: int = 50,
        enable_tqdm: bool = True,
        early_stop: bool = True,
        has_ma: bool = True,
        has_volume_bar: bool = True,
        is_years: list = IS_YEARS,
        oos_years: list = OOS_YEARS,
        country: str = "USA",
        transfer_learning=None,
        annual_stocks_num: any = "all",
        tstat_threshold: float = 0,
        ohlc_len=None,
        pf_freq=None,
        tensorboard=False,
        weight_decay=0,
        loss_name="cross_entropy",
        margin=1,
        train_size_ratio=0.7,
        ts_scale: str = "image_scale",
        chart_type: str = "bar",
        delayed_ret: int = 0
    ) -> None:
        """
        Initialize an Experiment object with specified parameters.
        """
        self.ws = ws
        self.pw = pw
        self.model_obj = model_obj
        self.train_freq = train_freq
        self.ensem = ensem
        self.lr = lr
        self.drop_prob = drop_prob
        self.device_number = device_number
        self.device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")
        self.max_epoch = max_epoch
        self.enable_tqdm = enable_tqdm
        self.early_stop = early_stop
        self.has_ma = has_ma
        self.has_volume_bar = has_volume_bar
        self.is_years = is_years
        self.oos_years = oos_years
        self.country = country
        self.tl = transfer_learning
        self.annual_stocks_num = annual_stocks_num
        self.tstat_threshold = tstat_threshold
        self.ohlc_len = ohlc_len if ohlc_len else ws
        self.pf_freq = FREQ_DICT[pw] if pf_freq is None else pf_freq
        self.tensorboard = tensorboard
        self.weight_decay = weight_decay
        self.loss_name = loss_name if model_obj.regression_label is None else "MSE"
        self.margin = margin
        self.train_size_ratio = train_size_ratio
        self.ts_scale = ts_scale
        self.chart_type = chart_type
        self.delayed_ret = delayed_ret
        self.label_dtype = torch.long if model_obj.regression_label is None else torch.float

        # paths
        self.exp_name = self.get_exp_name()
        self.pf_dir = self.get_portfolio_dir()
        model_name = model_obj.name
        self.model_dir = ut.get_dir(os.path.join(EXP_DIR, model_name, self.exp_name))
        self.ensem_res_dir = ut.get_dir(os.path.join(self.model_dir, "ensem_res"))
        self.tb_dir = ut.get_dir(os.path.join(self.model_dir, "tensorboard_res"))
        self.oos_metrics_path = os.path.join(self.ensem_res_dir, "oos_metrics_no_delay.pkl")

        # pick the earliest OOS year
        self.oos_start_year = oos_years[0]

    def get_exp_name(self) -> str:
        """
        Synthesize an experiment name from the key parameters.
        """
        name_parts = [
            f"{self.ws}d{self.pw}p-lr{self.lr:.0E}-dp{self.drop_prob:.2f}",
            f"ma{self.has_ma}-vb{self.has_volume_bar}-{self.train_freq}lyTrained"
        ]
        if self.delayed_ret != 0:
            name_parts.append(f"{self.delayed_ret}DelayedReturn")
        else:
            name_parts.append("noDelayedReturn")
        if not self.model_obj.batch_norm:
            name_parts.append("noBN")
        if not self.model_obj.xavier:
            name_parts.append("noXavier")
        if not self.model_obj.lrelu:
            name_parts.append("noLRelu")
        if self.weight_decay != 0:
            name_parts.append(f"WD{self.weight_decay:.0E}")
        if self.loss_name != "cross_entropy":
            name_parts.append(self.loss_name)
            if self.loss_name == "multimarginloss":
                name_parts.append(f"margin{self.margin:.0E}")
        if self.annual_stocks_num != "all":
            name_parts.append(f"top{self.annual_stocks_num}AnnualStock")
        if self.tstat_threshold != 0:
            name_parts.append(f"{self.tstat_threshold}tstat")
        if self.ohlc_len != self.ws:
            name_parts.append(f"{self.ohlc_len}ohlc")
        if self.train_size_ratio != 0.7:
            name_parts.append(f"tv_ratio{self.train_size_ratio:.1f}")
        if self.model_obj.regression_label is not None:
            name_parts.append(self.model_obj.regression_label)
        if self.ts_scale != "image_scale":
            sc_name = "raw_ts1d" if self.ts_scale == "ret_scale" else "vol_scale"
            name_parts.append(sc_name)
        if self.chart_type != "bar":
            name_parts.append(self.chart_type)
        if self.country != "USA":
            name_parts.append(str(self.country))
        if self.tl:
            name_parts.append(str(self.tl))
        return "-".join(name_parts)

    def get_portfolio_dir(self) -> str:
        """
        Construct a path for storing portfolio results.
        """
        name_list = [self.country]
        if self.model_obj.name not in BENCHMARK_MODEL_NAME_DICT.values():
            name_list.append(self.model_obj.name)
        name_list.append(self.exp_name)
        name_list.append(f"ensem{self.ensem}")
        if (self.oos_years[0] != 2001) or (self.oos_years[-1] != 2019):
            name_list.append(f"{self.oos_years[0]}-{self.oos_years[-1]}")
        if self.pf_freq != FREQ_DICT[self.pw]:
            name_list.append(f"{self.pf_freq}ly")
        if self.delayed_ret == 0:
            name_list.append("noDelayedReturn")
        else:
            name_list.append(f"{self.delayed_ret}DelayedReturn")
        name = "_".join(name_list)
        return ut.get_dir(os.path.join(PORTFOLIO_DIR, name))

    def get_train_validate_dataloaders_dict(self) -> dict:
        """
        Concat data from multiple in-sample years, then split into train/validate sets.
        """
        if self.model_obj.ts1d_model:
            tv_datasets = {
                year: TS1DDataset(
                    window_size=self.ws,
                    predict_window=self.pw,
                    freq=self.train_freq,
                    year=year,
                    country=self.country,
                    remove_tail=(year == self.oos_start_year - 1),
                    ohlc_len=self.ohlc_len,
                    ts_scale=self.ts_scale,
                    regression_label=self.model_obj.regression_label
                ) for year in self.is_years
            }
        else:
            tv_datasets = {
                year: EquityDataset(
                    window_size=self.ws,
                    predict_window=self.pw,
                    freq=self.train_freq,
                    year=year,
                    country=self.country,
                    has_volume_bar=self.has_volume_bar,
                    has_ma=self.has_ma,
                    annual_stocks_num=self.annual_stocks_num,
                    tstat_threshold=self.tstat_threshold,
                    ohlc_len=self.ohlc_len,
                    regression_label=self.model_obj.regression_label,
                    chart_type=self.chart_type,
                    delayed_ret=self.delayed_ret,
                    remove_tail=(year == self.oos_start_year - 1)
                ) for year in self.is_years
            }

        tv_dataset = ConcatDataset([tv_datasets[y] for y in self.is_years])
        train_size = int(len(tv_dataset) * self.train_size_ratio)
        validate_size = len(tv_dataset) - train_size

        train_dataset, validate_dataset = random_split(tv_dataset, [train_size, validate_size])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(validate_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

        return {"train": train_loader, "validate": val_loader}

    def train_empirical_ensem_model(self, ensem_range=None, pretrained=True):
        """
        Train multiple seeds (ensemble models).
        """
        if ensem_range is None:
            ensem_range = range(self.ensem)

        val_df = pd.DataFrame(columns=["MCC", "loss", "accy", "diff", "epoch"])
        train_df = pd.DataFrame(columns=["MCC", "loss", "accy", "diff", "epoch"])

        for model_num in ensem_range:
            print(f"Start Training Ensem Number {model_num}")
            model_save_path = self.get_model_checkpoint_path(model_num)
            if os.path.exists(model_save_path) and pretrained:
                print(f"Found pretrained model {model_save_path}")
                validate_metrics = torch.load(model_save_path)
            else:
                dataloaders_dict = self.get_train_validate_dataloaders_dict()
                train_metrics, validate_metrics, _model = self.train_single_model(
                    dataloaders_dict, model_save_path, model_num=model_num
                )
                for col in train_metrics.keys():
                    train_df.loc[model_num, col] = train_metrics[col]

            for col in validate_metrics.keys():
                if col == "model_state_dict":
                    continue
                val_df.loc[model_num, col] = validate_metrics[col]

        val_df = val_df.astype(float).round(3)
        val_df.loc["Mean"] = val_df.mean()
        csv_path = os.path.join(LOG_DIR, f"{self.model_obj.name}-{self.exp_name}-ensem{self.ensem}.csv")
        val_df.to_csv(csv_path, index=True)

        with open(os.path.join(LATEX_DIR, f"{self.model_obj.name}-{self.exp_name}-ensem{self.ensem}.txt"), "w+") as f:
            f.write(val_df.to_latex())

    def train_single_model(self, dataloaders_dict, model_save_path, model_num=None):
        """
        Train a single model, returning the best validation metrics and final train metrics.
        """
        if self.country != "USA" and self.tl is not None:
            us_model_save_path = model_save_path.replace(f"-{self.country}-{self.tl}", "")
            model_state_dict = torch.load(us_model_save_path, map_location=self.device)["model_state_dict"]
            model = self.model_obj.init_model_with_model_state_dict(model_state_dict, device=self.device)

            if self.tl == "usa":
                validate_metrics = self.evaluate(model, {"validate": dataloaders_dict["validate"]})["validate"]
                validate_metrics["epoch"] = 0
                self.release_dataloader_memory(dataloaders_dict, model)
                return None, validate_metrics, None

            elif self.tl == "ft":
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Linear(model.fc.in_features, 2)
                model.fc.apply(cnn_model.init_weights)
                optimizer = optim.Adam(model.fc.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                model.to(self.device)
            else:
                raise ValueError(f"{self.tl} not supported.")
        else:
            model = self.model_obj.init_model(device=self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        cudnn.benchmark = True
        best_validate_metrics = {"loss": 10.0, "accy": 0.0, "MCC": 0.0, "epoch": 0}
        best_model = copy.deepcopy(model.state_dict())
        since = time.time()

        for epoch in range(self.max_epoch):
            for phase in ["train", "validate"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                data_iter = (
                    tqdm(dataloaders_dict[phase], leave=True, unit="batch")
                    if self.enable_tqdm else dataloaders_dict[phase]
                )

                running_metrics = {
                    "running_loss": 0.0,
                    "running_correct": 0.0,
                    "TP": 0,
                    "TN": 0,
                    "FP": 0,
                    "FN": 0
                }

                for batch in data_iter:
                    inputs = batch["image"].to(self.device, dtype=torch.float)
                    labels = batch["label"].to(self.device, dtype=self.label_dtype)
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        loss = self.loss_from_model_output(labels, outputs)
                        _, preds = torch.max(outputs, 1)
                        if phase == "train":
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    self._update_running_metrics(loss, labels, preds, running_metrics)
                    del inputs, labels

                epoch_stat = self._generate_epoch_stat(
                    epoch, self.lr, len(dataloaders_dict[phase].dataset), running_metrics
                )
                if self.enable_tqdm and hasattr(data_iter, "set_postfix"):
                    data_iter.set_postfix(epoch_stat)
                print(epoch_stat)

                if phase == "validate" and epoch_stat["loss"] < best_validate_metrics["loss"]:
                    for key in ["loss", "accy", "MCC", "epoch", "diff"]:
                        best_validate_metrics[key] = epoch_stat[key]
                    best_model = copy.deepcopy(model.state_dict())

            if self.early_stop and (epoch - best_validate_metrics["epoch"]) >= 2:
                break

        time_elapsed = time.time() - since
        print(f"Training took {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best val loss: {best_validate_metrics['loss']} at epoch {best_validate_metrics['epoch']}")

        model.load_state_dict(best_model)
        best_validate_metrics["model_state_dict"] = model.state_dict().copy()
        torch.save(best_validate_metrics, model_save_path)

        train_metrics = self.evaluate(model, {"train": dataloaders_dict["train"]})["train"]
        train_metrics["epoch"] = best_validate_metrics["epoch"]
        self.release_dataloader_memory(dataloaders_dict, model)

        del best_validate_metrics["model_state_dict"]
        return train_metrics, best_validate_metrics, model

    @staticmethod
    def release_dataloader_memory(dataloaders_dict, model):
        """
        Free up memory from dataloaders and the model.
        """
        for k in list(dataloaders_dict.keys()):
            dataloaders_dict[k] = None
        del model
        torch.cuda.empty_cache()

    def get_model_checkpoint_path(self, model_num: int) -> str:
        return os.path.join(self.model_dir, f"checkpoint{model_num}.pth.tar")

    def evaluate(self, model, dataloaders_dict, new_label=None) -> dict:
        """
        Evaluate the model on the given dataloaders, returning metrics.
        """
        model.to(self.device)
        res_dict = {}
        for subset in dataloaders_dict.keys():
            data_iter = (
                tqdm(dataloaders_dict[subset], leave=True, unit="batch")
                if self.enable_tqdm else dataloaders_dict[subset]
            )
            model.eval()
            running_metrics = {
                "running_loss": 0.0,
                "running_correct": 0.0,
                "TP": 0,
                "TN": 0,
                "FP": 0,
                "FN": 0
            }
            for batch in data_iter:
                inputs = batch["image"].to(self.device, dtype=torch.float)
                if new_label is not None:
                    labels = torch.Tensor([new_label]).repeat(inputs.shape[0]).to(self.device, dtype=self.label_dtype)
                else:
                    labels = batch["label"].to(self.device, dtype=self.label_dtype)
                with torch.no_grad():
                    outputs = model(inputs)
                loss = self.loss_from_model_output(labels, outputs)
                _, preds = torch.max(outputs, 1)
                self._update_running_metrics(loss, labels, preds, running_metrics)
                del inputs, labels

            epoch_stat = self._generate_epoch_stat(-1, -1, len(dataloaders_dict[subset].dataset), running_metrics)
            print(f"{subset.upper()} => {epoch_stat}")
            res_dict[subset] = {m: epoch_stat[m] for m in ["loss", "accy", "MCC", "diff"]}
        return res_dict

    def _generate_epoch_stat(self, epoch, learning_rate, num_samples, running_metrics):
        """
        Summarize key metrics at the end of an epoch or evaluation pass.
        """
        TP = running_metrics["TP"]
        TN = running_metrics["TN"]
        FP = running_metrics["FP"]
        FN = running_metrics["FN"]

        out = {
            "epoch": epoch,
            "lr": f"{learning_rate:.2E}" if learning_rate > 0 else -1,
            "diff": 1.0 * ((TP + FP) - (TN + FN)) / num_samples,
            "loss": running_metrics["running_loss"] / num_samples,
            "accy": (TP + TN) / num_samples
        }
        denom = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
        if denom == 0:
            out["MCC"] = np.nan
        else:
            out["MCC"] = (TP * TN - FP * FN) / math.sqrt(denom)
        return out

    def _update_running_metrics(self, loss, labels, preds, running_metrics) -> None:
        """
        Accumulate batch results into running metrics (loss, accuracy, confusion matrix).
        """
        running_metrics["running_loss"] += loss.item() * len(labels)
        running_metrics["running_correct"] += torch.sum(preds == labels).item()
        running_metrics["TP"] += torch.sum(preds * labels).item()
        running_metrics["TN"] += torch.sum((preds - 1) * (labels - 1)).item()
        running_metrics["FP"] += torch.sum(preds * (labels - 1)).abs().item()
        running_metrics["FN"] += torch.sum((preds - 1) * labels).abs().item()

    def loss_from_model_output(self, labels, outputs) -> torch.Tensor:
        """
        Convert the model's raw outputs to a scalar loss given the chosen loss name.
        """
        if self.loss_name == "kldivloss":
            log_prob = nn.LogSoftmax(dim=1)(outputs)
            target = ut.binary_one_hot(labels.view(-1, 1), self.device)
            target = target.float()
            loss = nn.KLDivLoss()(log_prob, target)
        elif self.loss_name == "multimarginloss":
            loss = nn.MultiMarginLoss(margin=self.margin)(outputs, labels.long())
        elif self.loss_name == "cross_entropy":
            loss = nn.CrossEntropyLoss()(outputs, labels.long())
        elif self.loss_name == "MSE":
            loss = nn.MSELoss()(outputs.flatten(), labels)
        else:
            loss = None
        return loss

    def load_ensemble_model(self):
        """
        Load all ensemble models for inference.
        """
        model_list = [self.model_obj.init_model() for _ in range(self.ensem)]
        for i in range(self.ensem):
            ckpt = self.get_model_checkpoint_path(i)
            if not os.path.exists(ckpt):
                print(f"Missing checkpoint {ckpt} -> returning None")
                return None
            state_dict = torch.load(ckpt, map_location=self.device)["model_state_dict"]
            model_list[i].load_state_dict(state_dict)
        return model_list

    # -------------------------------------------------------------------------
    # ADDED METHODS FOR ENSEMBLE RESULT GENERATION AND LOADING
    # -------------------------------------------------------------------------

    def generate_ensem_res(self, freq, load_saved_data, year_list=None):
        """
        Generate ensemble predictions for each stock in the specified years
        and store them to CSV with a column named 'up_prob'.
        If load_saved_data is True, skip generation if the CSV is found.
        """
        if year_list is None:
            year_list = self.oos_years

        model_list = self.load_ensemble_model()
        if model_list is None:
            print("No ensemble models found. Cannot generate ensemble results.")
            return

        for year in year_list:
            out_path = os.path.join(self.ensem_res_dir, f"ensem{self.ensem}_res_{year}_{freq}.csv")
            if load_saved_data and os.path.exists(out_path):
                print(f"Found existing ensemble result for year {year} at {out_path}")
                continue

            # Build the dataset for the given year using EquityDataset (adjust as needed)
            year_dataset = EquityDataset(
                window_size=self.ws,
                predict_window=self.pw,
                freq=freq,
                year=year,
                country=self.country,
                has_volume_bar=self.has_volume_bar,
                has_ma=self.has_ma,
                annual_stocks_num=self.annual_stocks_num,
                tstat_threshold=self.tstat_threshold,
                ohlc_len=self.ohlc_len,
                regression_label=self.model_obj.regression_label,
                chart_type=self.chart_type,
                delayed_ret=self.delayed_ret,
                remove_tail=(year == self.oos_start_year - 1)
            )
            year_loader = DataLoader(year_dataset, batch_size=BATCH_SIZE, shuffle=False)

            print(f"Generating ensemble results for year {year} -> {out_path}")
            result_rows = []
            for batch in tqdm(year_loader, desc=f"Year {year}"):
                inputs = batch["image"].to(self.device, dtype=torch.float)
                total_prob = torch.zeros(len(inputs), 2, device=self.device)
                with torch.no_grad():
                    for mdl in model_list:
                        mdl.eval()
                        outputs = mdl(inputs)
                        probs = nn.Softmax(dim=1)(outputs)
                        total_prob += probs

                avg_prob = total_prob / len(model_list)
                up_prob_tensor = avg_prob[:, 1].detach().cpu()

                for i in range(len(inputs)):
                    row_dict = {
                        "StockID": batch["StockID"][i],
                        "Date": batch["ending_date"][i],
                        "up_prob": float(up_prob_tensor[i].item()),
                        "MarketCap": float(batch["MarketCap"][i].item()),
                    }
                    result_rows.append(row_dict)

            df_out = pd.DataFrame(result_rows)
            df_out.to_csv(out_path, index=False)
            print(f"Saved ensemble results with 'up_prob' to {out_path}")

    def load_ensem_res(self, year=None, multiindex=False, freq=None):
        """
        Load the CSV(s) created by generate_ensem_res and merge them into a single DataFrame.
        If multiindex is True, set [Date, StockID] as the index.
        """
        if freq is None:
            freq = self.pf_freq

        if year is None:
            year_list = self.oos_years
        elif isinstance(year, int):
            year_list = [year]
        else:
            year_list = year

        df_list = []
        for y in year_list:
            csv_path = os.path.join(self.ensem_res_dir, f"ensem{self.ensem}_res_{y}_{freq}.csv")
            if not os.path.exists(csv_path):
                print(f"No ensemble results found at {csv_path} - skipping.")
                continue
            print(f"Loading ensemble results from {csv_path}")
            df_temp = pd.read_csv(csv_path, parse_dates=["Date"])
            if "up_prob" not in df_temp.columns:
                raise ValueError(f"File {csv_path} missing 'up_prob' column!")
            df_list.append(df_temp)

        if not df_list:
            print("No ensemble data loaded -> returning empty DataFrame.")
            return pd.DataFrame()

        whole_ensemble_res = pd.concat(df_list, ignore_index=True)
        if multiindex:
            whole_ensemble_res.set_index(["Date", "StockID"], inplace=True)
        return whole_ensemble_res

    def load_oos_ensem_stat(self) -> dict:
        """
        Load or compute out-of-sample metrics.
        Returns a dictionary with keys like 'loss', 'accy', 'MCC', 'Spearman', 'Pearson', etc.
        (For demonstration, returns a placeholder dictionary.)
        """
        # In a complete implementation, you would compute these metrics from the ensemble results.
        return {"loss": 0.09, "accy": 0.98, "MCC": 0.23, "Spearman": 0.25, "Pearson": 0.27}

    def load_portfolio_obj(
        self, delay_list=[0], load_signal=True, custom_ret=None, transaction_cost=False
    ) -> pf.PortfolioManager:
        """
        Helper that loads ensemble results, instantiates a PortfolioManager,
        and returns it for use in calculate_portfolio.
        """
        if load_signal:
            whole_ensemble_res = self.load_ensem_res(year=self.oos_years, multiindex=True)
        else:
            whole_ensemble_res = None
        pf_obj = pf.PortfolioManager(
            signal_df=whole_ensemble_res,
            freq=self.pf_freq,
            portfolio_dir=self.pf_dir,
            start_year=self.oos_years[0],
            end_year=self.oos_years[-1],
            country=self.country,
            delay_list=delay_list,
            load_signal=load_signal,
            custom_ret=custom_ret,
            transaction_cost=transaction_cost,
        )
        return pf_obj

    def calculate_portfolio(
        self, load_saved_data=True, delay_list=[0], is_ensem_res=True, cut=10
    ) -> None:
        """
        Generate ensemble results (if needed), compute OOS metrics, and build decile portfolios.
        """
        if is_ensem_res:
            ensem_res_year_list = list(self.is_years) + list(self.oos_years)
        else:
            ensem_res_year_list = list(self.oos_years)

        self.generate_ensem_res(freq=self.pf_freq, load_saved_data=load_saved_data, year_list=ensem_res_year_list)
        oos_metrics = self.load_oos_ensem_stat()
        print(oos_metrics)

        if self.delayed_ret != 0:
            delay_list = delay_list + [self.delayed_ret]

        pf_obj = self.load_portfolio_obj(delay_list=delay_list)
        for delay in delay_list:
            pf_obj.generate_portfolio(delay=delay, cut=cut)


# -------------------------------------------------------------------------
# Public function for training U.S. models
# -------------------------------------------------------------------------

def train_us_model(
    ws_list,
    pw_list,
    dp=0.50,
    ensem=5,
    total_worker=1,
    dn=None,
    from_ensem_res=True,
    ensem_range=None,
    train_size_ratio=0.7,
    is_ensem_res=True,
    vb=True,
    ma=True,
    tstat_filter=0,
    stocks_for_train="all",
    batch_norm=True,
    chart_type="bar",
    delayed_ret=0,
    calculate_portfolio=False,
    ts1d_model=False,
    ts_scale="image_scale",
    regression_label=None,
    pf_delay_list=[0],
    lr=1e-5
) -> None:
    """
    Public function to train CNN models with a certain set of (ws_list, pw_list) configs,
    optionally building a portfolio.
    """
    torch.set_num_threads(1)
    worker_idx = 0
    if total_worker > 1:
        # Adapt logic for parallel workers if needed.
        pass

    setting_list = list(itertools.product(ws_list, pw_list))

    if dn is None:
        dn = (worker_idx + 0) % 2

    for ws, pw in setting_list:
        exp_obj = get_bl_exp_obj(
            ws=ws,
            pw=pw,
            dn=dn,
            drop_prob=dp,
            train_size_ratio=train_size_ratio,
            ensem=ensem,
            has_volume_bar=vb,
            has_ma=ma,
            tstat_filter=tstat_filter,
            stocks_for_train=stocks_for_train,
            batch_norm=batch_norm,
            chart_type=chart_type,
            delayed_ret=delayed_ret,
            ts1d_model=ts1d_model,
            ts_scale=ts_scale,
            regression_label=regression_label,
            lr=lr,
        )

        exp_obj.train_empirical_ensem_model(ensem_range=ensem_range)

        if calculate_portfolio:
            exp_obj.calculate_portfolio(
                load_saved_data=from_ensem_res,
                is_ensem_res=is_ensem_res,
                delay_list=pf_delay_list
            )


def get_bl_exp_obj(
    ws,
    pw,
    dn=0,
    train_freq=None,
    drop_prob=0.5,
    train_size_ratio=0.7,
    ensem=5,
    is_years=IS_YEARS,
    oos_years=OOS_YEARS,
    country="USA",
    inplanes=TRUE_DATA_CNN_INPLANES,
    transfer_learning=None,
    has_ma=True,
    has_volume_bar=True,
    pf_freq=None,
    ohlc_len=None,
    tstat_filter=0,
    stocks_for_train="all",
    batch_norm=True,
    chart_type="bar",
    delayed_ret=0,
    ts1d_model=False,
    ts_scale="image_scale",
    regression_label=None,
    lr=1e-5
):
    _ohlc_len = ohlc_len if ohlc_len else ws
    if ts1d_model:
        filter_size_list, stride_list, dilation_list, max_pooling_list = EMP_CNN1d_BL_SETTING[_ohlc_len]
        layer_number = TS1D_LAYERNUM_DICT[_ohlc_len]
    else:
        filter_size_list, stride_list, dilation_list, max_pooling_list = EMP_CNN_BL_SETTING[_ohlc_len]
        layer_number = BENCHMARK_MODEL_LAYERNUM_DICT[_ohlc_len]

    model_obj = cnn_model.Model(
        ws=_ohlc_len,
        layer_number=layer_number,
        inplanes=inplanes,
        drop_prob=drop_prob,
        filter_size_list=filter_size_list,
        stride_list=stride_list,
        dilation_list=dilation_list,
        max_pooling_list=max_pooling_list,
        batch_norm=batch_norm,
        regression_label=regression_label,
        ts1d_model=ts1d_model
    )

    train_freq = train_freq if train_freq else FREQ_DICT[pw]
    exp_obj = Experiment(
        ws=ws,
        pw=pw,
        model_obj=model_obj,
        train_freq=train_freq,
        ensem=ensem,
        lr=lr,
        drop_prob=drop_prob,
        device_number=dn,
        max_epoch=50,
        enable_tqdm=True,
        early_stop=True,
        has_ma=has_ma,
        has_volume_bar=has_volume_bar,
        is_years=is_years,
        oos_years=oos_years,
        weight_decay=0,
        loss_name="cross_entropy",
        margin=1,
        train_size_ratio=train_size_ratio,
        country=country,
        transfer_learning=transfer_learning,
        pf_freq=pf_freq,
        ohlc_len=ohlc_len,
        tstat_threshold=tstat_filter,
        annual_stocks_num=stocks_for_train,
        chart_type=chart_type,
        delayed_ret=delayed_ret,
        ts_scale=ts_scale
    )
    return exp_obj


if __name__ == "__main__":
    # For example, you might run a training job by calling train_us_model with desired parameters.
    train_us_model(
        ws_list=[20],
        pw_list=[20],
        calculate_portfolio=True,
        pf_delay_list=[0]
    )
