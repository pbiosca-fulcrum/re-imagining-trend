"""
cnn_model.py

Defines a flexible CNN architecture (2D or 1D) for classification/regression tasks.

**Key Modifications**:
- Default arguments for batch_norm=True, xavier=True, lrelu=True,
  and bn_loc="bn_bf_relu" to match Kelly’s approach.
- Default drop_prob=0.50 remains, as in Kelly’s code.
- If you want to do 1D or 2D, you pass `ts1d_model=True/False`.

Other details: 
- We keep final output = 2 units for classification, or 1 unit for regression_label.
"""

import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary

from src.utils.config import TRUE_DATA_CNN_INPLANES


def init_weights(m: nn.Module) -> None:
    """
    Xavier initialization for Conv/Linear layers.
    """
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class Flatten(nn.Module):
    """
    Flatten module to convert final 2D/3D tensor to (batch_size, -1) for FC layers.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Model:
    """
    A container that sets up either a 2D or 1D CNN architecture, storing hyperparams,
    and automatically adjusts for volume-bar image height if volume_bar=True.
    
    Attributes:
    ws (int): The base chart 'window_size' (e.g. 20).
    layer_number (int): How many conv layers to stack.
    inplanes (int): Base number of output channels for the first conv block.
    drop_prob (float): Dropout probability at the end of conv layers.
    filter_size_list (list): Kernel sizes for each conv layer.
    stride_list (list): Strides for each conv layer.
    dilation_list (list): Dilations for each conv layer.
    max_pooling_list (list): (height, width) pooling factors for each conv layer.
    batch_norm (bool): Whether to apply BatchNorm after each conv.
    xavier (bool): If True, apply xavier_uniform_ to weights.
    lrelu (bool): If True, use LeakyReLU; else standard ReLU.
    bn_loc (str): (Unused in this simple example. Kept for config.)
    conv_layer_chanls (list or None): If provided, custom channels per layer. Otherwise auto.
    regression_label (str or None): If not None, do regression (output=1). Else classification (output=2).
    ts1d_model (bool): If True, we're building the 1D version of the model (not used here).
    volume_bar (bool): If True, the chart images have a height offset for volume sub-chart.

    Key defaults to match Kelly’s typical setup:
      - batch_norm=True
      - xavier=True
      - lrelu=True
      - bn_loc="bn_bf_relu"
      - drop_prob=0.5
    """

    def __init__(
        self,
        ws: int,
        layer_number: int,
        inplanes: int = TRUE_DATA_CNN_INPLANES,
        drop_prob: float = 0.50,
        filter_size=None,
        stride=None,
        dilation=None,
        max_pooling=None,
        filter_size_list=None,
        stride_list=None,
        dilation_list=None,
        max_pooling_list=None,
        batch_norm: bool = True,
        xavier: bool = True,
        lrelu: bool = True,
        bn_loc: str = "bn_bf_relu",
        conv_layer_chanls=None,
        regression_label=None,
        ts1d_model: bool = False,
        volume_bar: bool = True,  # optional for 2D chart images
    ):
        self.ws = ws
        self.layer_number = layer_number
        self.inplanes = inplanes
        self.drop_prob = drop_prob
        self.filter_size_list = filter_size_list or [filter_size] * layer_number
        self.stride_list = stride_list or [stride] * layer_number
        self.dilation_list = dilation_list or [dilation] * layer_number
        self.max_pooling_list = max_pooling_list or [max_pooling] * layer_number
        self.batch_norm = batch_norm
        self.xavier = xavier
        self.lrelu = lrelu
        self.bn_loc = bn_loc
        self.conv_layer_chanls = conv_layer_chanls
        self.regression_label = regression_label
        self.ts1d_model = ts1d_model
        self.volume_bar = volume_bar

        # For 1D vs 2D, we handle differently in init_model
        if self.ts1d_model:
            # 1D logic: we might handle padding as integer
            self.padding_list = [fs // 2 if fs else 1 for fs in self.filter_size_list]
        else:
            # 2D logic: handle (height, width) pairs
            self.padding_list = [
                (int(fs[0] / 2), int(fs[1] / 2))
                for fs in self.filter_size_list
                if fs
            ]

        # Construct a descriptive name
        self.name = self._get_full_model_name()

    def _get_full_model_name(self) -> str:
        arch = f"D{self.ws}L{self.layer_number}"
        vb_suffix = "_vb" if self.volume_bar else ""
        return arch + vb_suffix

    def init_model(self, device=None, state_dict=None) -> nn.Module:
        """
        Initialize the PyTorch model object (CNNModel or CNN1DModel).
        """
        if self.ts1d_model:
            model = CNN1DModel(
                layer_number=self.layer_number,
                ws=self.ws,
                inplanes=self.inplanes,
                drop_prob=self.drop_prob,
                filter_size_list=self.filter_size_list,
                stride_list=self.stride_list,
                padding_list=self.padding_list,
                dilation_list=self.dilation_list,
                max_pooling_list=self.max_pooling_list,
                regression_label=self.regression_label,
            )
        else:
            model = CNNModel(
                layer_number=self.layer_number,
                ws=self.ws,
                inplanes=self.inplanes,
                drop_prob=self.drop_prob,
                filter_size_list=self.filter_size_list,
                stride_list=self.stride_list,
                padding_list=self.padding_list,
                dilation_list=self.dilation_list,
                max_pooling_list=self.max_pooling_list,
                batch_norm=self.batch_norm,
                xavier=self.xavier,
                lrelu=self.lrelu,
                bn_loc=self.bn_loc,
                conv_layer_chanls=self.conv_layer_chanls,
                regression_label=self.regression_label,
                volume_bar=self.volume_bar,
            )

        if state_dict is not None:
            model.load_state_dict(state_dict)

        if device is not None:
            model.to(device)

        return model

    def init_model_with_model_state_dict(self, model_state_dict, device=None) -> nn.Module:
        """
        Load an existing model_state_dict into a fresh instance of the model.
        """
        model = self.init_model(device=device)
        model.load_state_dict(model_state_dict)
        return model


class CNNModel(nn.Module):
    """
    2D CNN Implementation for chart image inputs (bar/pixel).

    By default:
     - 'batch_norm=True' means we add BN after conv,
     - 'lrelu=True' uses LeakyReLU,
     - 'bn_loc="bn_bf_relu"' as a typical config,
     - 'xavier=True' calls init_weights on the entire model.
    """

    def __init__(
        self,
        layer_number: int,
        ws: int,
        inplanes: int,
        drop_prob: float,
        filter_size_list,
        stride_list,
        padding_list,
        dilation_list,
        max_pooling_list,
        batch_norm: bool,
        xavier: bool,
        lrelu: bool,
        bn_loc: str,
        conv_layer_chanls,
        regression_label=None,
        volume_bar: bool = True,
    ):
        super().__init__()
        self.layer_number = layer_number
        self.ws = ws
        self.inplanes = inplanes
        self.drop_prob = drop_prob
        self.filter_size_list = filter_size_list
        self.stride_list = stride_list
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.max_pooling_list = max_pooling_list
        self.batch_norm = batch_norm
        self.xavier = xavier
        self.lrelu = lrelu
        self.bn_loc = bn_loc
        self.conv_layer_chanls = conv_layer_chanls
        self.regression_label = regression_label
        self.volume_bar = volume_bar

        self.conv_layers = self._init_conv_layers()
        fc_size = self._get_conv_layers_flatten_size()
        if regression_label:
            self.fc = nn.Linear(fc_size, 1)
        else:
            self.fc = nn.Linear(fc_size, 2)

        if xavier:
            self.apply(init_weights)

    def _init_conv_layers(self) -> nn.Sequential:
        """
        Build the series of 2D convolutional blocks (Conv->BN->ReLU->Pool).
        """
        if self.conv_layer_chanls is None:
            conv_layer_chanls = [self.inplanes * (2 ** i) for i in range(self.layer_number)]
        else:
            conv_layer_chanls = self.conv_layer_chanls

        layers = []
        in_ch = 1
        for i, out_ch in enumerate(conv_layer_chanls):
            block = self._conv_block_2d(
                in_ch=in_ch,
                out_ch=out_ch,
                fs=self.filter_size_list[i],
                st=self.stride_list[i],
                pad=self.padding_list[i],
                dil=self.dilation_list[i],
                mp=self.max_pooling_list[i],
            )
            layers.append(block)
            in_ch = out_ch

        layers.append(Flatten())
        layers.append(nn.Dropout(p=self.drop_prob))
        return nn.Sequential(*layers)

    def _conv_block_2d(
        self, in_ch: int, out_ch: int, fs, st, pad, dil, mp
    ) -> nn.Sequential:
        """
        Create a single 2D conv block with optional BN, ReLU, and MaxPool.
        """
        block_list = []
        conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=fs,
            stride=st,
            padding=pad,
            dilation=dil
        )
        block_list.append(conv)

        if self.batch_norm:
            block_list.append(nn.BatchNorm2d(out_ch))

        block_list.append(nn.LeakyReLU() if self.lrelu else nn.ReLU())

        if mp != (1, 1):
            block_list.append(nn.MaxPool2d(mp, ceil_mode=True))

        return nn.Sequential(*block_list)

    def _get_conv_layers_flatten_size(self) -> int:
        """
        Pass a dummy input of the correct (height, width) into conv_layers
        so we know how many features come out after the last conv block.

        If ws=20 with volume_bar=True, the actual height is 64 + extra for volume, etc.
        """
        from src.data.dgp_config import IMAGE_HEIGHT, IMAGE_WIDTH, VOLUME_CHART_GAP

        base_h = IMAGE_HEIGHT[self.ws]
        if self.volume_bar:
            base_h += int(base_h / 5) + VOLUME_CHART_GAP

        base_w = IMAGE_WIDTH[self.ws]
        dummy = torch.rand((1, 1, base_h, base_w))
        x = self.conv_layers(dummy)
        return x.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


class CNN1DModel(nn.Module):
    """
    1D CNN Implementation for time-series usage (O/H/L/C/Vol lines).
    Typically, shape is (batch, 6, window_size).
    """

    def __init__(
        self,
        layer_number: int,
        ws: int,
        inplanes: int,
        drop_prob: float,
        filter_size_list,
        stride_list,
        padding_list,
        dilation_list,
        max_pooling_list,
        regression_label=None,
    ):
        super().__init__()
        self.layer_number = layer_number
        self.ws = ws
        self.inplanes = inplanes
        self.drop_prob = drop_prob
        self.filter_size_list = filter_size_list
        self.stride_list = stride_list
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.max_pooling_list = max_pooling_list
        self.regression_label = regression_label

        self.conv_layers = self._init_ts1d_conv_layers()
        fc_size = self._get_ts1d_conv_layers_flatten_size()

        if regression_label:
            self.fc = nn.Linear(fc_size, 1)
        else:
            self.fc = nn.Linear(fc_size, 2)

        self.apply(init_weights)

    def _init_ts1d_conv_layers(self) -> nn.Sequential:
        conv_layer_chanls = [self.inplanes * (2 ** i) for i in range(self.layer_number)]
        layers = []
        prev_chanl = 6  # open/high/low/close/ma/vol
        for i, out_ch in enumerate(conv_layer_chanls):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=prev_chanl,
                        out_channels=out_ch,
                        kernel_size=self.filter_size_list[i],
                        stride=self.stride_list[i],
                        padding=self.padding_list[i],
                        dilation=self.dilation_list[i],
                    ),
                    nn.BatchNorm1d(out_ch),
                    nn.LeakyReLU(),
                    nn.MaxPool1d(self.max_pooling_list[i], ceil_mode=True),
                )
            )
            prev_chanl = out_ch

        layers.append(Flatten())
        layers.append(nn.Dropout(p=self.drop_prob))
        return nn.Sequential(*layers)

    def _get_ts1d_conv_layers_flatten_size(self):
        dummy_input = torch.rand((1, 6, self.ws))
        x = self.conv_layers(dummy_input)
        return x.shape[1]

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x
