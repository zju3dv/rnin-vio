"""
* This file is part of RNIN-VIO
*
* Copyright (c) ZJU-SenseTime Joint Lab of 3D Vision. All Rights Reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

import torch
import torch.nn as nn
from torch.nn.init import orthogonal_

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(planes, planes * self.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(planes * self.expansion),
        )
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.convs(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.clone() + identity
        out = self.relu(out)

        return out

class FcBlock(nn.Module):
    def __init__(self, in_dim, out_dim, mid_dim=256, dropout=0.2):
        super(FcBlock, self).__init__()
        self.mid_dim = mid_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        # fc layers
        self.fcs = nn.Sequential(
            nn.Linear(self.in_dim, self.mid_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(self.mid_dim, self.out_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fcs(x)
        return x

class ResNetLSTMSeqNet(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super(ResNetLSTMSeqNet, self).__init__()
        data_window_config = dict([
            ("past_data_size", int(cfg['model_param']['past_time'] * cfg['data']['imu_freq'])),
            ("window_size", int(cfg['model_param']['window_time'] * cfg['data']['imu_freq'])),
            ("future_data_size", int(cfg['model_param']['future_time'] * cfg['data']['imu_freq'])),
            ("step_size", int(cfg['data']['imu_freq'] / cfg['data']['sample_freq'])), ])
        input_dim = cfg['model_param']['input_dim']
        output_dim = cfg['model_param']['output_dim']
        layer_sizes = cfg['model_param']['layer_sizes']
        self.lstm_size = cfg['model_param']['lstm_size']
        self.lstm_dropout = cfg['model_param']['lstm_dropout']
        self.num_layers = cfg['model_param']['lstm_layers']
        self.win_size = data_window_config["window_size"] + data_window_config["past_data_size"] + data_window_config["future_data_size"]
        self.num_direction = 1
        self.res_net_out_channel = 128
        self.resnet_code = self.res_net_out_channel * int(self.win_size / 16 + 1)

        self.base_plane = 64
        self.inplanes = self.base_plane
        # Input module
        self.input_block = nn.Sequential(
            nn.Conv1d(
                input_dim, self.base_plane, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm1d(self.base_plane),
            nn.ReLU(inplace=True),
        )
        # Residual groups
        self.residual_groups = nn.Sequential(
            self.stack_res_layres(ResBlock, 64, layer_sizes[0], stride=1),
            self.stack_res_layres(ResBlock, 128, layer_sizes[1], stride=2),
            self.stack_res_layres(ResBlock, 256, layer_sizes[2], stride=2),
            self.stack_res_layres(ResBlock, 512, layer_sizes[3], stride=2),
        )
        self.resnet_post_pro = nn.Sequential(
            nn.Conv1d(
                512, self.res_net_out_channel, kernel_size=1, bias=False
            ),
            nn.BatchNorm1d(self.res_net_out_channel),
        )

        # LSTM
        self.lstm = nn.LSTM(self.resnet_code, self.lstm_size, self.num_layers,
                            batch_first=True, dropout=self.lstm_dropout, bidirectional=False)

        # Output module
        self.output_block1 = FcBlock(self.lstm_size, output_dim)  # dp mean
        self.output_block2 = FcBlock(self.lstm_size, output_dim)  # dp cov

        self.initialize()

    def freeze_cov(self):
        for param in self.output_block2.parameters():
            param.requires_grad = False

    def unfreeze(self):
        # unfreeze all:
        for param in self.parameters():
            param.requires_grad = True

    def stack_res_layres(self, block, planes, layer_sizes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion)
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride=stride, downsample=downsample)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, layer_sizes):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                # layer 1
                orthogonal_(m.weight_ih_l0)
                orthogonal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                # Set forget gate bias to 1 (remember)
                n = m.bias_hh_l0.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l0.data[start:end].fill_(1.)

                # layer 2
                if self.num_layers > 1:
                    orthogonal_(m.weight_ih_l1)
                    orthogonal_(m.weight_hh_l1)
                    m.bias_ih_l1.data.zero_()
                    m.bias_hh_l1.data.zero_()
                    n = m.bias_hh_l1.size(0)
                    start, end = n // 4, n // 2
                    m.bias_hh_l1.data[start:end].fill_(1.)

    def init_hidden(self, x, batch_size, first_batch=False):

        weight = next(self.parameters()).data
        if first_batch:
            if torch.cuda.is_available():
                hidden = (weight.new(batch_size, self.num_layers * self.num_direction, self.lstm_size).zero_().to(x.device),
                          weight.new(batch_size, self.num_layers * self.num_direction, self.lstm_size).zero_().to(x.device))
            else:
                hidden = (weight.new(batch_size, self.num_layers * self.num_direction, self.lstm_size).zero_(),
                          weight.new(batch_size, self.num_layers * self.num_direction, self.lstm_size).zero_())
        else:
            if torch.cuda.is_available():
                hidden = (weight.new(self.num_layers * self.num_direction, batch_size, self.lstm_size).zero_().to(x.device),
                          weight.new(self.num_layers * self.num_direction, batch_size, self.lstm_size).zero_().to(x.device))
            else:
                hidden = (weight.new(self.num_layers * self.num_direction, batch_size, self.lstm_size).zero_(),
                          weight.new(self.num_layers * self.num_direction, batch_size, self.lstm_size).zero_())

        return hidden

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, compute_type=None, hn=None, cn=None):
        self.lstm.flatten_parameters()
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))

        # Resnet_encode
        x = self.input_block(x)
        x = self.residual_groups(x)
        embed = self.resnet_post_pro(x)

        # LSTM
        embed = embed.view(batch_size, seq_len, -1)
        if hn == None or cn == None:
            (hn, cn) = self.init_hidden(x, batch_size)
        out, (hn2, cn2) = self.lstm(embed, (hn, cn))
        out = out.contiguous().view(-1, self.lstm_size * self.num_direction)

        # FC
        x1 = self.output_block1(out)  # mean
        x1 = x1.view(batch_size, seq_len, -1)

        if compute_type == None:
            x2 = self.output_block2(out)  # covariance s = log(sigma)
            x2 = x2.view(batch_size, seq_len, -1)
            return x1, x2
        else:
            return x1




