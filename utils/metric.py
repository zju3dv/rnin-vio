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
import matplotlib.pyplot as plt
from os import path as osp

def compute_absolute_trajectory_error(est, gt):
    """
    The Absolute Trajectory Error (ATE) defined in:
    A Benchmark for the evaluation of RGB-D SLAM Systems
    http://ais.informatik.uni-freiburg.de/publications/papers/sturm12iros.pdf

    Args:
        est: estimated trajectory
        gt: ground truth trajectory. It must have the same shape as est.

    Return:
        Absolution trajectory error, which is the Root Mean Squared Error between
        two trajectories.
    """
    # return np.sqrt(np.mean((est - gt) ** 2))
    return np.sqrt(np.mean(np.linalg.norm(est - gt, axis=1) ** 2))

def compute_relative_trajectory_error_time(est, gt, delta, max_delta=-1):
    """
    The Relative Trajectory Error (RTE) defined in:
    A Benchmark for the evaluation of RGB-D SLAM Systems
    http://ais.informatik.uni-freiburg.de/publications/papers/sturm12iros.pdf

    Args:
        est: the estimated trajectory
        gt: the ground truth trajectory.
        delta: fixed window size. If set to -1, the average of all RTE up to max_delta will be computed.
        max_delta: maximum delta. If -1 is provided, it will be set to the length of trajectories.

    Returns:
        Relative trajectory error. This is the mean value under different delta.
    """
    if max_delta == -1:
        max_delta = est.shape[0]
    deltas = np.array([delta]) if delta > 0 else np.arange(1, min(est.shape[0], max_delta))
    t_rtes = np.zeros((deltas.shape[0], 2))
    for i in range(deltas.shape[0]):
        # For each delta, the RTE is computed as the RMSE of endpoint drifts from fixed windows
        # slided through the trajectory.
        err = est[deltas[i]:] + gt[:-deltas[i]] - est[:-deltas[i]] - gt[deltas[i]:]
        # rtes[i] = np.sqrt(np.mean(err ** 2))
        t_rtes[i] = np.sqrt(np.mean(np.linalg.norm(err, axis=1) ** 2))

    # The average of RTE of all window sized is returned.
    return np.mean(t_rtes)


def compute_relative_trajectory_norm_angle_error_time(est, gt, delta):
    """
    The Relative Trajectory Error (RTE) defined in:
    A Benchmark for the evaluation of RGB-D SLAM Systems
    http://ais.informatik.uni-freiburg.de/publications/papers/sturm12iros.pdf

    Args:
        est: the estimated trajectory
        gt: the ground truth trajectory.
        delta: fixed window size. If set to -1, the average of all RTE up to max_delta will be computed.
        max_delta: maximum delta. If -1 is provided, it will be set to the length of trajectories.

    Returns:
        Relative trajectory error. This is the mean value under different delta.
    """
    delta = int(delta)
    re_est = est[delta:] - est[:-delta]
    re_gt = gt[delta:] - gt[:-delta]

    t_rtes = np.zeros((re_est.shape[0], 2))
    norm_est = np.linalg.norm(re_est, axis=1)
    norm_gt = np.linalg.norm(re_gt, axis=1)
    t_rtes[:, 0] = np.abs(norm_est - norm_gt)
    dot = np.array([np.dot(re_est[i], re_gt[i]) for i in range(re_est.shape[0])])
    t_rtes[:, 1] = np.arccos(dot / (norm_gt * norm_est)) * 180.0 / np.pi
    return t_rtes

def compute_relative_trajectory_error_dist(est, gt, delta=1):
    """
    Almost the same as t_rte in which the length of a window is one minute, while the length of a window in d_rte is one meter(default).

    Args:
        est: the estimated trajectory
        gt: the ground truth trajectory.

    Returns:
        Relative trajectory error. This is the mean value under different delta.
    """

    gt_delta_len = np.linalg.norm(gt[1:] - gt[:-1], axis=1)
    end_index = np.zeros((est.shape[0], 1), dtype=int)

    # calculate where the 1 meter endpoint is
    j = 0
    i = 0
    current_sum = 0.0
    while i < est.shape[0]:
        while j < gt_delta_len.shape[0]:
            current_sum = current_sum + gt_delta_len[j]
            if current_sum >= 1.0:
                break
            j = j + 1
        if j == gt_delta_len.shape[0]:
            # done
            break
        else:
            # reach the endpoint x_{j+1} of x_i
            end_index[i] = j + 1
            current_sum = current_sum - gt_delta_len[j] # make sure current_sum < 1.0 now
            current_sum = current_sum - gt_delta_len[i]
            i = i + 1

    d_rtes = np.zeros(len(end_index))
    for i in range(len(end_index)):
        # For each delta, the RTE is computed as the RMSE of endpoint drifts from fixed windows
        # slided through the trajectory.
        err = est[end_index[i]] + gt[i] - est[i] - gt[end_index[i]]
        # rtes[i] = np.sqrt(np.mean(err ** 2))
        d_rtes[i] = np.sqrt(np.mean(np.linalg.norm(err, axis=1) ** 2))

    # The average of RTE of all window sized is returned.
    return np.mean(d_rtes)

def compute_relative_trajectory_norm_angle_error_dist(est, gt, delta=1.0):

    gt_delta_len = np.linalg.norm(gt[1:] - gt[:-1], axis=1)
    end_index = np.zeros((est.shape[0], 1), dtype=int)
    # calculate where the 1 meter endpoint is
    j = 0
    i = 0
    current_sum = 0.0
    while i < est.shape[0]:
        while j < gt_delta_len.shape[0]:
            current_sum = current_sum + gt_delta_len[j]
            if current_sum >= delta:
                break
            j = j + 1
        if j == gt_delta_len.shape[0]:
            # done
            break
        else:
            # reach the endpoint x_{j+1} of x_i
            end_index[i] = j + 1
            current_sum = current_sum - gt_delta_len[j] # make sure current_sum < 1.0 now
            current_sum = current_sum - gt_delta_len[i]
            i = i + 1

    d_rtes = np.zeros((len(end_index), 2))
    for i in range(len(end_index)):
        # For each delta, the RTE is computed as the RMSE of endpoint drifts from fixed windows
        # slided through the trajectory.
        err = est[end_index[i]] + gt[i] - est[i] - gt[end_index[i]]
        # rtes[i] = np.sqrt(np.mean(err ** 2))
        d_rtes[i] = np.sqrt(np.mean(np.linalg.norm(err, axis=1) ** 2))

        re_est = est[end_index[i]] - est[i]
        re_gt = gt[end_index[i]] - gt[i]
        norm_est = np.linalg.norm(re_est)
        norm_gt = np.linalg.norm(re_gt)
        norm_error = np.abs(norm_est - norm_gt)
        angle_error = np.arccos(np.dot(re_est[0], re_gt[0]) / (norm_gt * norm_est + 1.0e-6)) * 180.0 / np.pi
        d_rtes[i][0] = norm_error
        d_rtes[i][1] = angle_error

    # The average of RTE of all window sized is returned.
    return d_rtes

def compute_ate_rte(est, gt, pred_per_min=12000):
    """
    A convenient function to compute ATE and RTE. For sequences shorter than pred_per_min, it computes end sequence
    drift and scales the number accordingly.
    """
    ate = compute_absolute_trajectory_error(est, gt)
    if est.shape[0] < pred_per_min:
        print("less than one minute!")
        ratio = pred_per_min / est.shape[0]
        t_rte = compute_relative_trajectory_error_time(est, gt, delta=est.shape[0] - 1) * ratio
    else:
        t_rte = compute_relative_trajectory_error_time(est, gt, delta=pred_per_min)

    d_rte = compute_relative_trajectory_error_dist(est, gt, delta=1)

    return ate, t_rte, d_rte

def compute_ate_norm_angle_rte(est, gt, pred_per_min=100):
    """
    A convenient function to compute ATE and RTE. For sequences shorter than pred_per_min, it computes end sequence
    drift and scales the number accordingly.
    """
    ate = compute_absolute_trajectory_error(est, gt)
    if est.shape[0] < pred_per_min:
        print("less than one minute!")
        ratio = pred_per_min / est.shape[0]
        t_rte = compute_relative_trajectory_norm_angle_error_time(est, gt, delta=est.shape[0] - 1) * ratio
    else:
        t_rte = compute_relative_trajectory_norm_angle_error_time(est, gt, delta=pred_per_min)

    d_rte = compute_relative_trajectory_norm_angle_error_dist(est, gt, delta=1.0)

    return ate, t_rte, d_rte

def compute_heading_error(est, gt):
    """
    Args:
        est: the estimated heading as sin, cos values
        gt: the ground truth heading as sin, cos values
    Returns:
        MSE error and angle difference from dot product
    """

    mse_error = np.mean((est-gt)**2)
    dot_prod = np.sum(est * gt, axis=1)
    angle = np.arccos(np.clip(dot_prod, a_min=-1, a_max=1))

    return mse_error, angle

def compute_density(d_rte_norm_all, d_rte_angle_all, out_dir):
    norm_step = 0.01
    d_rte_norm_avg = np.mean(d_rte_norm_all)
    d_rte_norm_max = np.max(d_rte_norm_all)
    d_rte_norm_min = np.min(d_rte_norm_all)
    d_rte_angle_avg = np.mean(d_rte_angle_all)
    d_rte_angle_max = np.max(d_rte_angle_all)
    d_rte_angle_min = np.min(d_rte_angle_all)

    d_rte_norm_error_p_d = np.zeros((int(d_rte_norm_max / norm_step) + 2, 1), dtype=float)
    d_rte_norm_error_accum_p = np.zeros((int(d_rte_norm_max / norm_step) + 2, 1), dtype=float)
    sum_size = d_rte_norm_all.shape[0]
    for i in range(1, d_rte_norm_error_p_d.shape[0]):
        min_error = (i-1) * norm_step
        max_error = i * norm_step
        d_rte_norm_error_p_d[i] = float(np.sum((d_rte_norm_all >= min_error) & (d_rte_norm_all < max_error)) / sum_size)
        d_rte_norm_error_accum_p[i] = d_rte_norm_error_accum_p[i - 1]
        d_rte_norm_error_accum_p[i] += d_rte_norm_error_p_d[i]

    sum_size = d_rte_angle_all.shape[0]
    d_rte_angle_error_p_d = np.zeros((int(d_rte_angle_max) + 2, 1))
    d_rte_angle_error_accum_p = np.zeros((int(d_rte_angle_max) + 2, 1))
    for i in range(1, d_rte_angle_error_p_d.shape[0]):
        min_error = (i - 1) * 1.0
        max_error = i * 1.0
        d_rte_angle_error_p_d[i] = float(np.sum((d_rte_angle_all >= min_error) & (d_rte_angle_all < max_error)) / sum_size)
        d_rte_angle_error_accum_p[i] = d_rte_angle_error_accum_p[i - 1]
        d_rte_angle_error_accum_p[i] += d_rte_angle_error_p_d[i]

    plt.figure(num="density", dpi=90, figsize=(16, 9))
    ax = plt.subplot2grid((1, 2), (0, 0))
    ax.set_xscale('log')
    plt.plot(range(d_rte_norm_error_p_d.shape[0]), d_rte_norm_error_p_d, label="norm_density")
    plt.plot(range(d_rte_norm_error_p_d.shape[0]), d_rte_norm_error_accum_p, label="norm_cum_density")
    plt.vlines(d_rte_norm_avg / norm_step, 0, 1.0, colors='g', label='avg_error')
    plt.ylabel("p")
    plt.xlabel("norm_error(cm)")
    plt.legend()
    plt.title("norm_error_density_avg_%.3fm_max_%.3fm_min_%.3fm" % (d_rte_norm_avg, d_rte_norm_max, d_rte_norm_min))
    plt.grid(True)
    bx = plt.subplot2grid((1, 2), (0, 1))
    plt.plot(range(d_rte_angle_error_p_d.shape[0]), d_rte_angle_error_p_d, label="angle_density")
    plt.plot(range(d_rte_angle_error_p_d.shape[0]), d_rte_angle_error_accum_p, label="angle_cum_density")
    plt.vlines(d_rte_angle_avg, 0, 1.0, colors='g', label='avg_error')
    plt.ylabel("p")
    plt.xlabel("angle_error(°)")
    plt.legend()
    plt.title("angle_error_density_avg_%.3f°_max_%.3f°_min_%.3f°" % (d_rte_angle_avg, d_rte_angle_max, d_rte_angle_min))
    plt.grid(True)
    plt.savefig(osp.join(out_dir, "density.png"))
    plt.close("all")

    np.savetxt(osp.join(out_dir, "d_rte_norm_error_density.txt"), d_rte_norm_error_p_d, fmt='%.6f', delimiter=',', )
    np.savetxt(osp.join(out_dir, "d_rte_norm_error_cum_density.txt"), d_rte_norm_error_accum_p, fmt='%.6f', delimiter=',', )
    np.savetxt(osp.join(out_dir, "d_rte_angle_error_density.txt"), d_rte_angle_error_p_d, fmt='%.6f', delimiter=',', )
    np.savetxt(osp.join(out_dir, "d_rte_angle_error_cum_density.txt"), d_rte_angle_error_accum_p, fmt='%.6f',
               delimiter=',', )