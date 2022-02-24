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

def loss_distribution_diag(pred, pred_cov, targ):
    loss = ((pred - targ).pow(2)) / (2 * torch.exp(2 * pred_cov)) + pred_cov
    return loss

def loss_cum_distribution_diag(pred, pred_cov, targ):
    loss = ((pred - targ).pow(2)) / (2 * pred_cov) + 0.5 * torch.log(pred_cov)
    return loss

def get_sequence_smooth_loss(pred, pred_cov, targ, epoch, start_cov_epoch):
    # loss = loss_mse(pred, targ)
    cum_gt_pos = torch.cumsum(targ, 1)
    pred_cum_pos = torch.cumsum(pred, 1)
    absolute_weight = 8.0
    if epoch <= start_cov_epoch:
        ## train dp
        loss1 = (pred - targ).pow(2)
        if len(pred.size()) == 2:
            return torch.mean(loss1)
        loss2 = absolute_weight * (pred_cum_pos[:, 1:] - cum_gt_pos[:, 1:]).pow(2)
        loss = torch.cat((loss1, loss2), 1)
    else:
        ## train dp and cov
        cov = torch.exp(2 * pred_cov)
        pred_cum_cov = torch.cumsum(cov, 1)
        loss1 = loss_distribution_diag(pred, pred_cov, targ)
        if len(pred.size()) == 2:
            return torch.mean(loss1)
        loss2 = absolute_weight * loss_cum_distribution_diag(pred_cum_pos[:, 1:], pred_cum_cov[:, 1:], cum_gt_pos[:, 1:])
        loss = torch.cat((loss1, loss2), 1)
    return torch.mean(loss)