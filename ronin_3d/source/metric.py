import numpy as np


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
    t_rtes = np.zeros(deltas.shape[0])
    for i in range(deltas.shape[0]):
        # For each delta, the RTE is computed as the RMSE of endpoint drifts from fixed windows
        # slided through the trajectory.
        err = est[deltas[i]:] + gt[:-deltas[i]] - est[:-deltas[i]] - gt[deltas[i]:]
        # rtes[i] = np.sqrt(np.mean(err ** 2))
        t_rtes[i] = np.sqrt(np.mean(np.linalg.norm(err, axis=1) ** 2))

    # The average of RTE of all window sized is returned.
    return np.mean(t_rtes)


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
