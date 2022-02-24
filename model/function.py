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
from model.losses import get_sequence_smooth_loss

def fun_train_forward(cfg, model, batch, start_cov_epochs, epoch):
    model_name = cfg['model']['model_name']

    feat, targ = batch
    if epoch <= start_cov_epochs:
        pred = model(feat, 'dp')
        pred_cov = torch.zeros_like(pred)
    else:
        pred, pred_cov = model(feat)

    output_dim = cfg['model_param']['output_dim']
    pred = pred[:, cfg['train']['predict_start']:, ]
    pred_cov = pred_cov[:, cfg['train']['predict_start']:, ]
    targ = targ[:, cfg['train']['predict_start']:, 0:output_dim]

    if cfg['model']['pred_velocity']:
        pred = pred * cfg['model_param']['window_time']
        pred_cov = torch.log(cfg['model_param']['window_time'] * torch.ones_like(pred_cov)) + pred_cov

    loss = get_sequence_smooth_loss(pred, pred_cov, targ, epoch, start_cov_epochs)

    all_b = pred.size(0) * pred.size(1)
    pred = pred.view(all_b, -1)
    pred_cov = pred_cov.view(all_b, -1)
    targ = targ.view(all_b, -1)
    return pred, pred_cov, targ, loss

def fun_test_forward(cfg, model, batch, start_cov_epochs, epoch):
    model_name = cfg['model']['model_name']
    feat, targ = batch
    if epoch <= start_cov_epochs:
        pred = model(feat, 'dp')
        pred_cov = torch.zeros_like(pred)
    else:
        pred, pred_cov = model(feat)

    output_dim = cfg['model_param']['output_dim']
    pred = pred[:, -1, ]
    pred_cov = pred_cov[:, -1, ]
    targ = targ[:, -1, 0:output_dim]

    if cfg['model']['pred_velocity']:
        pred = pred * cfg['model_param']['window_time']
        pred_cov = torch.log(cfg['model_param']['window_time'] * torch.ones_like(pred_cov)) + pred_cov

    loss = get_sequence_smooth_loss(pred, pred_cov, targ, epoch, start_cov_epochs)

    return pred, pred_cov, targ, loss