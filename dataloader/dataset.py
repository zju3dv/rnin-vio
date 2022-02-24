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

from scipy.interpolate import interp1d
import pandas
import random
from numpy.random import normal as gen_normal
from os import path as osp
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import Dataset
import logging
import matplotlib.pyplot as plt
import matplotlib

class SenseINSSequence(object):
    def __init__(self, data_path, imu_freq, window_size, verbose=True, plot=False):
        super().__init__()
        (
            self.ts,
            self.features,
            self.targets,
            self.orientations,
            self.gt_pos,
            self.gt_ori,
        ) = (None, None, None, None, None, None)
        self.imu_freq = imu_freq
        self.interval = window_size
        self.data_valid = False
        self.sum_dur = 0
        self.valid = False
        self.plot = plot
        if data_path is not None:
            self.valid = self.load(data_path, verbose=verbose)

    def load(self, data_path, verbose=True):
        if data_path[-1] == '/':
            data_path = data_path[:-1]

        file = osp.join(data_path, 'SenseINS.h5')
        if osp.exists(file):
            imu_all = pandas.read_hdf(file, 'imu_all')
        else:
            file = osp.join(data_path, 'SenseINS.csv')
            if osp.exists(file):
                imu_all = pandas.read_csv(file)
                imu_all.to_hdf(osp.join(data_path, 'SenseINS.h5'), key='imu_all', mode='w')
            else:
                logging.info(f"dataset_fb.py: file is not exist. {file}")
                return

        if 'times' in imu_all:
            tmp_ts = np.array(imu_all[['times']].values)
        else:
            tmp_ts = np.array(imu_all[['time']].values)

        if tmp_ts.shape[0] < 1000:
            return False
        tmp_ts = np.squeeze(tmp_ts)
        tmp_vio_q = np.array(imu_all[['gt_q_w', 'gt_q_x', 'gt_q_y', 'gt_q_z']].values)
        self.get_gt = True
        if tmp_vio_q[0][0] == 1.0 and tmp_vio_q[100][0] == 1.0 or tmp_vio_q[0][0] == tmp_vio_q[-1][0]:
            tmp_vio_q = np.array(imu_all[['vio_q_w', 'vio_q_x', 'vio_q_y', 'vio_q_z']].values)
            tmp_vio_p = np.array(imu_all[['vio_p_x', 'vio_p_y', 'vio_p_z']].values)
            self.get_gt = False
        else:
            tmp_vio_p = np.array(imu_all[['gt_p_x', 'gt_p_y', 'gt_p_z']].values)

        tmp_gyro = np.array(imu_all[['gyro_x', 'gyro_y', 'gyro_z']].values)
        tmp_accel = np.array(imu_all[['acce_x', 'acce_y', 'acce_z']].values)

        tmp_vio_gyro_bias = np.array(imu_all[['vio_gyro_bias_x', 'vio_gyro_bias_y', 'vio_gyro_bias_z']].values)
        tmp_vio_acce_bias = np.array(imu_all[['vio_acce_bias_x', 'vio_acce_bias_y', 'vio_acce_bias_z']].values)

        tmp_gyro = tmp_gyro - tmp_vio_gyro_bias[-1, :]
        tmp_acce = tmp_accel - tmp_vio_acce_bias[-1, :]

        start_ts = tmp_ts[10]
        end_ts = tmp_ts[10] + int((tmp_ts[-20]-tmp_ts[1]) * self.imu_freq) / self.imu_freq
        ts = np.arange(start_ts, end_ts, 1.0/self.imu_freq)
        self.data_valid = True
        self.sum_dur = end_ts - start_ts

        if verbose:
            logging.info(f"{data_path}: sum time: {self.sum_dur}, gt: {self.get_gt}")

        vio_q_slerp = Slerp(tmp_ts, Rotation.from_quat(tmp_vio_q[:, [1, 2, 3, 0]]))
        vio_r = vio_q_slerp(ts)
        vio_p = interp1d(tmp_ts, tmp_vio_p, axis=0)(ts)
        gyro = interp1d(tmp_ts, tmp_gyro, axis=0)(ts)
        acce = interp1d(tmp_ts, tmp_acce, axis=0)(ts)

        ts = ts[:, np.newaxis]


        ori_R_vio = vio_r
        ori_R = ori_R_vio

        gt_disp = vio_p[self.interval:] - vio_p[: -self.interval]

        glob_gyro = np.einsum("tip,tp->ti", ori_R.as_matrix(), gyro)
        glob_acce = np.einsum("tip,tp->ti", ori_R.as_matrix(), acce)
        glob_acce -= np.array([0.0, 0.0, 9.805])

        self.ts = ts
        self.features = np.concatenate([glob_gyro, glob_acce], axis=1)
        self.orientations = ori_R.as_quat()   # [x, y, z, w]
        self.gt_pos = vio_p
        self.gt_ori = ori_R_vio.as_quat()
        self.targets = gt_disp
        return True

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_data_valid(self):
        return self.data_valid

    def get_aux(self):
        return np.concatenate(
            [self.ts, self.orientations, self.gt_pos, self.gt_ori], axis=1
        )

class BasicSequenceData(object):
    def __init__(self, cfg, data_list, verbose=True, **kwargs):
        super(BasicSequenceData, self).__init__()
        self.window_size = int(cfg['model_param']['window_time'] * cfg['data']['imu_freq'])
        self.past_data_size = int(cfg['model_param']['past_time'] * cfg['data']['imu_freq'])
        self.future_data_size = int(cfg['model_param']['future_time'] * cfg['data']['imu_freq'])
        self.step_size = int(cfg['data']['imu_freq'] / cfg['data']['sample_freq'])
        self.seq_len = cfg['train']["seq_len"]

        self.index_map = []
        self.ts, self.orientations, self.gt_pos, self.gt_ori = [], [], [], []
        self.features, self.targets = [], []
        self.valid_t, self.valid_samples = [], []
        self.data_paths = []
        self.valid_continue_good_time = 0.1

        self.mode = kwargs.get("mode", "train")
        sum_t = 0
        win_dt = self.window_size / cfg['data']['imu_freq']
        self.valid_sum_t = 0
        self.valid_all_samples = 0
        max_v_norm = 4.0
        valid_i = 0
        for i in range(len(data_list)):
            seq = SenseINSSequence(
                data_list[i], cfg['data']['imu_freq'], self.window_size, verbose=verbose
            )
            if seq.valid is False:
                continue
            feat, targ, aux = seq.get_feature(), seq.get_target(), seq.get_aux()
            sum_t += seq.sum_dur
            valid_samples = 0
            index_map = []
            step_size = self.step_size
            if self.mode in ["train", "val"] and seq.get_gt is False:
                for j in range(
                        self.past_data_size,
                        targ.shape[0] - self.future_data_size - (self.seq_len - 1) * self.window_size,
                        step_size):
                    outlier = False
                    for k in range(self.seq_len):
                        index = j + k * self.window_size
                        velocity = np.linalg.norm(targ[index] / win_dt)
                        if velocity > max_v_norm:
                            outlier = True
                            break
                    if outlier is False:
                        index_map.append([valid_i, j])
                        self.valid_all_samples += 1
                        valid_samples += 1
            else:
                for j in range(
                        self.past_data_size,
                        targ.shape[0] - self.future_data_size - (self.seq_len - 1) * self.window_size,
                        step_size):
                    index_map.append([valid_i, j])
                    self.valid_all_samples += 1
                    valid_samples += 1

            if len(index_map) > 0:
                self.data_paths.append(data_list[i])
                self.index_map.append(index_map)
                self.features.append(feat)
                self.targets.append(targ)
                self.ts.append(aux[:, 0])
                self.orientations.append(aux[:, 1:5])
                self.gt_pos.append(aux[:, 5:8])
                self.gt_ori.append(aux[:, 8:12])
                self.valid_samples.append(valid_samples)
                valid_i += 1
        if verbose:
            logging.info(f"datasets sum time {sum_t}")

    def plot_targets(self, out_dir):
        targets = np.concatenate(self.targets, axis=0)
        targets = np.clip(targets, -1.0, 1.0)
        fig = plt.figure(num="targets", dpi=90, figsize=(9, 9))
        plt.hist2d(targets[:, 0], targets[:, 1],
                       bins=100, norm=matplotlib.colors.LogNorm(),
                       cmap=matplotlib.cm.jet)
        plt.title('targets')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        fig.savefig(osp.join(out_dir, "targets.png"))
        plt.close("all")

        fig = plt.figure(num="targets", dpi=90, figsize=(9, 9))
        plt.hist(targets[:, 2], bins=100)
        plt.title('targets_z')
        fig.savefig(osp.join(out_dir, "targets_z.png"))
        plt.close("all")

        fig = plt.figure(num="targets_n", dpi=90, figsize=(9, 9))
        plt.hist(np.linalg.norm(targets[:, 0:2], axis=-1), bins=100)
        plt.title('targets_norm')
        fig.savefig(osp.join(out_dir, "targets_norm.png"))
        plt.close("all")

    def get_data(self):
        return self.features, self.targets, self.ts, self.orientations, self.gt_pos, self.gt_ori

    def get_index_map(self):
        return self.index_map

    def get_merged_index_map(self):
        index_map = []
        for i in range(len(self.index_map)):
            index_map += self.index_map[i]
        return index_map

class ResNetLSTMSeqToSeqDataset(Dataset):
    def __init__(self, cfg, basic_data: BasicSequenceData, index_map, **kwargs):
        super(ResNetLSTMSeqToSeqDataset, self).__init__()
        self.window_size = basic_data.window_size
        self.past_data_size = basic_data.past_data_size
        self.future_data_size = basic_data.future_data_size
        self.step_size = basic_data.step_size
        self.seq_len = basic_data.seq_len

        self.add_bias_noise = cfg['augment']['add_bias_noise']
        self.accel_bias_range = cfg['augment']['accel_bias_range']
        self.gyro_bias_range = cfg['augment']['gyro_bias_range']
        if self.add_bias_noise is False:
            self.accel_bias_range = 0.0
            self.gyro_bias_range = 0.0
        self.add_gravity_noise = cfg['augment']['add_gravity_noise']
        self.gravity_noise_theta_range = cfg['augment']['gravity_noise_theta_range']

        self.feat_acc_sigma = cfg['augment']['feat_acc_sigma']
        self.feat_gyr_sigma = cfg['augment']['feat_gyr_sigma']

        self.mode = kwargs.get("mode", "train")
        self.shuffle, self.transform, self.gauss = False, False, False
        if self.mode == "train":
            self.shuffle = True
            self.transform = True
            self.gauss = True
        elif self.mode == "val":
            self.shuffle = True
        elif self.mode == "test":
            self.shuffle = False

        self.features, self.targets, self.ts, self.orientations, self.gt_pos, self.gt_ori = basic_data.get_data()
        self.index_map = index_map
        if self.shuffle:
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        # in the world frame
        feat = self.features[seq_id][frame_id - self.past_data_size:
                                     frame_id + self.seq_len * self.window_size + self.future_data_size]
        # raw_feat = feat
        targ = self.targets[seq_id][frame_id:
                                    frame_id + self.seq_len * self.window_size:
                                    self.window_size]  # the beginning of the sequence

        if self.mode in ["train"]:
            targ_aug = np.copy(targ)
            feat_aug = np.copy(feat)
            if self.transform:
                angle = np.random.random() * (2 * np.pi)
                rm = np.array(
                    [[np.cos(angle), -(np.sin(angle))], [np.sin(angle), np.cos(angle)]]
                )
                feat_aug[:, 0:2] = np.matmul(rm, feat_aug[:, 0:2].T).T
                feat_aug[:, 3:5] = np.matmul(rm, feat_aug[:, 3:5].T).T
                targ_aug[:, 0:2] = np.matmul(rm, targ_aug[:, 0:2].T).T

            if self.add_bias_noise:
                # shift in the accel and gyro bias terms
                random_bias = np.random.random((1, 6))
                random_bias[:, 0:3] = (random_bias[:, 0:3] - 0.5) * self.gyro_bias_range / 0.5
                random_bias[:, 3:6] = (random_bias[:, 3:6] - 0.5) * self.accel_bias_range / 0.5
                feat_aug += random_bias

            if self.add_gravity_noise:
                angle_rand = random.random() * np.pi * 2
                vec_rand = np.array([np.cos(angle_rand), np.sin(angle_rand), 0])
                theta_rand = (
                        random.random() * np.pi * self.gravity_noise_theta_range / 180.0
                )
                rvec = theta_rand * vec_rand
                r = Rotation.from_rotvec(rvec)
                R_mat = r.as_matrix()
                feat_aug[:, 0:3] = np.matmul(R_mat, feat_aug[:, 0:3].T).T
                feat_aug[:, 3:6] = np.matmul(R_mat, feat_aug[:, 3:6].T).T

            if self.gauss:
                if self.feat_gyr_sigma > 0:
                    feat_aug[:, 0:3] += gen_normal(loc=0.0, scale=self.feat_gyr_sigma, size=(len(feat_aug[:, 0]), 3))
                if self.feat_acc_sigma > 0:
                    feat_aug[:, 3:6] += gen_normal(loc=0.0, scale=self.feat_acc_sigma, size=(len(feat_aug[:, 0]), 3))
            feat = feat_aug
            targ = targ_aug

        seq_feat = []
        for i in range(self.seq_len):
            seq_feat.append(feat[i * self.window_size:
                                 self.past_data_size + (
                                             i + 1) * self.window_size + self.future_data_size, :].T)
        seq_feat = np.array(seq_feat)
        return seq_feat.astype(np.float32), targ.astype(np.float32)

    def __len__(self):
        return len(self.index_map)

def SeqToSeqDataset(cfg, basic_data: BasicSequenceData, index_map, **kwargs):
    return ResNetLSTMSeqToSeqDataset(cfg, basic_data, index_map, **kwargs)

def partition_data(index_map, valid_samples, valid_all_samples, training_rate=0.9, valuation_rate=0.1, data_rate=1.0, shuffle=True):

    if shuffle:
        np.random.shuffle(index_map)

    all_size = 0
    sum_valid_samples = valid_all_samples * data_rate

    accum_samples = 0.0
    for i in range(len(index_map)):
        accum_samples += valid_samples[index_map[i][0][0]]
        all_size = i
        if accum_samples > sum_valid_samples:
            break

    valuation_samples = sum_valid_samples * valuation_rate

    train_index_map, valuation_index_map = [], []
    accum_valuation_samples = 0

    for i in range(all_size):
        if accum_valuation_samples < valuation_samples:
            valuation_index_map += index_map[i]
            accum_valuation_samples += valid_samples[index_map[i][0][0]]
        else:
            train_index_map += index_map[i]

    return train_index_map, valuation_index_map