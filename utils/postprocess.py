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
import numpy as np
from os import path as osp
from scipy.interpolate import interp1d
import logging
import matplotlib.pyplot as plt

def pose_integrate(cfg, dataset, preds):
    window_time = cfg['model_param']['window_time']
    imu_freq = cfg['data']['imu_freq']
    seq_len = cfg['train']['seq_len']

    dp_t = window_time
    pred_vels = preds / dp_t

    ind = np.array([i[1] for i in dataset.index_map], dtype=np.int)
    delta_int = int(
        window_time * imu_freq / 2.0
    )
    delta_int += int((seq_len - 1) * window_time * imu_freq)
    if not (window_time * imu_freq / 2.0).is_integer():
        logging.info("Trajectory integration point is not centered.")
    ind_intg = ind + delta_int

    ts = dataset.ts[0]
    dts = np.mean(ts[ind_intg[1:]] - ts[ind_intg[:-1]])
    pos_intg = np.zeros([pred_vels.shape[0] + 1, pred_vels.shape[1]])
    pos_intg[0] = dataset.gt_pos[0][ind_intg[0], 0:pos_intg.shape[1]]
    pos_intg[1:] = np.cumsum(pred_vels[:, :] * dts, axis=0) + pos_intg[0]

    ts_intg = np.append(ts[ind_intg], ts[ind_intg[-1]] + dts)

    ts_in_range = ts[ind_intg[0] : ind_intg[-1]]  # s
    pos_pred = interp1d(ts_intg, pos_intg, axis=0)(ts_in_range)
    pos_gt = dataset.gt_pos[0][ind_intg[0] : ind_intg[-1], 0:pos_intg.shape[1]]

    traj_attr_dict = {
        "ts": ts_in_range,
        "pos_pred": pos_pred,
        "pos_gt": pos_gt,
    }

    return traj_attr_dict

def compute_plot_dict(sample_freq, net_attr_dict, traj_attr_dict):

    ts = traj_attr_dict["ts"]
    pos_pred = traj_attr_dict["pos_pred"]
    pos_gt = traj_attr_dict["pos_gt"]

    total_pred = net_attr_dict["preds"].shape[0]
    pred_ts = (1.0 / sample_freq) * np.arange(total_pred)
    pred_sigmas = np.exp(net_attr_dict["preds_cov"])
    plot_dict = {
        "ts": ts,
        "pos_pred": pos_pred,
        "pos_gt": pos_gt,
        "pred_ts": pred_ts,
        "preds": net_attr_dict["preds"],
        "targets": net_attr_dict["targets"],
        "pred_sigmas": pred_sigmas,
    }

    return plot_dict

def plot_imus(feat, num=None, dpi=None, figsize=None):
    fig = plt.figure(num=num, dpi=dpi, figsize=figsize)
    x = len(feat[:, 0])
    plt.subplot(2, 1, 1)
    for i in range(3):
        plt.plot(x, feat[:, i])
        plt.plot(x, feat[:, i])
    plt.ylabel('gyr')
    plt.legend()
    plt.grid(True)
    plt.xlabel('t(s)')

    plt.subplot(2, 1, 2)
    for i in range(3):
        plt.plot(x, feat[:, 3+i])
        plt.plot(x, feat[:, 3+i])
    plt.ylabel('acc')
    plt.legend()
    plt.grid(True)
    plt.xlabel('t(s)')
    return fig

def make_plots(plot_dict, outdir):
    pos_pred = plot_dict["pos_pred"]
    pos_gt = plot_dict["pos_gt"]
    pred_ts = plot_dict["pred_ts"]
    preds = plot_dict["preds"]
    targets = plot_dict["targets"]
    pred_sigmas = plot_dict["pred_sigmas"]

    dpi = 90
    figsize = (16, 9)

    fig1 = plt.figure(num="ins_traj", dpi=dpi, figsize=figsize)
    targ_names = ["dx", "dy", "dz"]
    plt.subplot2grid((3, 2), (0, 0), rowspan=3)
    plt.plot(pos_pred[:, 0], pos_pred[:, 1])
    plt.plot(pos_gt[:, 0], pos_gt[:, 1])
    # step = float(preds.shape[0] / pos_pred.shape[0])
    # for i in range(0, pos_pred.shape[0], 10):
    #     idx = int(np.around(i * step))
    #     pred = preds[idx]
    #     pred = 2 * pred / np.linalg.norm(pred)
        # plt.arrow(pos_gt[i, 0], pos_gt[i, 1], pred[0], pred[1], head_width=0.3)
    plt.axis("equal")
    plt.legend(["network_pred", "Ground_truth"])
    plt.title("2D trajectory and ATE error against time")
    for i in range(preds.shape[1]):
        plt.subplot2grid((preds.shape[1], 2), (i, 1))
        plt.plot(preds[:, i])
        plt.plot(targets[:, i])
        plt.legend(["network_pred", "Ground_truth"])
        plt.title("{}".format(targ_names[i]))
    plt.tight_layout()
    plt.grid(True)
    fig1.savefig(osp.join(outdir, "traj.png"))

    fig2 = plt.figure(num="pred_sigma", dpi=dpi, figsize=figsize)
    preds_plus_sig = preds + 3 * pred_sigmas
    preds_minus_sig = preds - 3 * pred_sigmas
    ylbs = ["x(m)", "y(m)", "z(m)"]
    for i in range(preds.shape[1]):
        plt.subplot(preds.shape[1], 1, i + 1)
        plt.plot(pred_ts, preds_plus_sig[:, i], "-g", linewidth=0.2)
        plt.plot(pred_ts, preds_minus_sig[:, i], "-g", linewidth=0.2)
        plt.plot(pred_ts, preds[:, i], "-b", linewidth=0.5, label='pred')
        plt.plot(pred_ts, targets[:, i], "-r", linewidth=0.5, label='gt')
        plt.ylabel(ylbs[i])
        plt.legend()
        plt.grid(True)
    plt.xlabel("t(s)")
    fig2.savefig(osp.join(outdir, "pred_sigma.svg"))

    plt.close("all")

    return