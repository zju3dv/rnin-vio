import json
import random
from os import path as osp

import pandas
import h5py
import numpy as np
import quaternion
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

from data_utils import CompiledSequence, select_orientation_source, load_cached_sequences

import logging
logging.getLogger().setLevel(logging.INFO)


class GlobSpeedSequence(CompiledSequence):
    """
    Dataset :- RoNIN (can be downloaded from http://ronin.cs.sfu.ca/)
    Features :- raw angular rate and acceleration (includes gravity).
    """
    feature_dim = 6
    # target_dim = 2
    target_dim = 3
    aux_dim = 8

    def __init__(self, data_path=None, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        self.info = {}

        self.grv_only = kwargs.get('grv_only', False)
        self.max_ori_error = kwargs.get('max_ori_error', 20.0)
        self.w = kwargs.get('interval', 1)
        if data_path is not None:
            self.load(data_path)

    def load(self, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]
        with open(osp.join(data_path, 'info.json')) as f:
            self.info = json.load(f)

        self.info['path'] = osp.split(data_path)[-1]

        self.info['ori_source'], ori, self.info['source_ori_error'] = select_orientation_source(
            data_path, self.max_ori_error, self.grv_only)

        with h5py.File(osp.join(data_path, 'data.hdf5')) as f:
            gyro_uncalib = f['synced/gyro_uncalib']
            acce_uncalib = f['synced/acce']
            gyro = gyro_uncalib - np.array(self.info['imu_init_gyro_bias'])
            acce = np.array(self.info['imu_acce_scale']) * (acce_uncalib - np.array(self.info['imu_acce_bias']))
            ts = np.copy(f['synced/time'])
            tango_pos = np.copy(f['pose/tango_pos'])
            init_tango_ori = quaternion.quaternion(*f['pose/tango_ori'][0])

        # Compute the IMU orientation in the Tango coordinate frame.
        ori_q = quaternion.from_float_array(ori)
        rot_imu_to_tango = quaternion.quaternion(*self.info['start_calibration'])
        init_rotor = init_tango_ori * rot_imu_to_tango * ori_q[0].conj()
        ori_q = init_rotor * ori_q

        dt = (ts[self.w:] - ts[:-self.w])[:, None]
        glob_v = (tango_pos[self.w:] - tango_pos[:-self.w]) / dt

        gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([gyro.shape[0], 1]), gyro], axis=1)) # quaternion is of w, x, y, z format
        acce_q = quaternion.from_float_array(np.concatenate([np.zeros([acce.shape[0], 1]), acce], axis=1))
        glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
        glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]

        start_frame = self.info.get('start_frame', 0)
        self.ts = ts[start_frame:]
        self.features = np.concatenate([glob_gyro, glob_acce], axis=1)[start_frame:]
        # self.targets = glob_v[start_frame:, :2]
        self.targets = glob_v[start_frame:, :3]
        self.orientations = quaternion.as_float_array(ori_q)[start_frame:]
        self.gt_pos = tango_pos[start_frame:]

        pass

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts[:, None], self.orientations, self.gt_pos], axis=1)

    def get_meta(self):
        return '{}: device: {}, ori_error ({}): {:.3f}'.format(
            self.info['path'], self.info['device'], self.info['ori_source'], self.info['source_ori_error'])


class SenseINSSequence(CompiledSequence):
    """
    Dataset :- RoNIN (can be downloaded from http://ronin.cs.sfu.ca/)
    Features :- raw angular rate and acceleration (includes gravity).
    """
    feature_dim = 6
    # target_dim = 2
    target_dim = 3
    aux_dim = 8

    def __init__(self, data_path=None, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        self.info = {}

        args = kwargs['args']

        self.imu_freq = args.imu_freq
        self.sum_dur = 0
        self.interval = args.window_size
        self.w = kwargs.get('interval', 1)

        if data_path is not None:
            self.load(data_path)

    def load(self, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]

        # info_file = osp.join(data_path, 'SenseINS.json')
        # if osp.exists(info_file):
        #     with open(info_file) as f:
        #         self.info = json.load(f)
        # else:
        #     logging.info(f"data_glob_speed.py: info_file does not exist. {info_file}")

        data_file = osp.join(data_path, 'SenseINS.csv')
        if osp.exists(data_file):
            imu_all = pandas.read_csv(data_file)
        else:
            logging.info(f"data_glob_speed.py: data_file does not exist. {data_file}")
            return

        self.info['path'] = osp.split(data_path)[-1]

        # ---ts---
        if 'times' in imu_all:
            tmp_ts = np.copy(imu_all[['times']].values)
        else:
            tmp_ts = np.copy(imu_all[['time']].values)
        tmp_ts = np.squeeze(tmp_ts)

        start_ts = tmp_ts[1]
        end_ts = tmp_ts[1] + int((tmp_ts[-4] - tmp_ts[1]) * self.imu_freq) / self.imu_freq # if use tmp_ts[-2], it may
                                                                                        # out of bounds when using interpolation

        ts_interval = 1.0 / self.imu_freq
        ts = np.arange(start_ts, end_ts, ts_interval)
        self.sum_dur = end_ts - start_ts
        self.info['time_duration'] = self.sum_dur

        # ---vio_q and vio_p---
        tmp_vio_q = np.copy(imu_all[['gt_q_w', 'gt_q_x', 'gt_q_y', 'gt_q_z']].values) # gt orientation

        get_gt = True
        # this is to check whether gt exists, if not, use VIO as gt
        if tmp_vio_q[0][0] == 1.0 and tmp_vio_q[100][0] == 1.0 or tmp_vio_q[0][0] == tmp_vio_q[-1][0]:
            tmp_vio_q = np.copy(imu_all[['vio_q_w', 'vio_q_x', 'vio_q_y', 'vio_q_z']].values)
            tmp_vio_p = np.copy(imu_all[['vio_p_x', 'vio_p_y', 'vio_p_z']].values)
            get_gt = False
        else:
            tmp_vio_p = np.copy(imu_all[['gt_p_x', 'gt_p_y', 'gt_p_z']].values)

        # ---acce and gyro---
        tmp_gyro = np.copy(imu_all[['gyro_x', 'gyro_y', 'gyro_z']].values)
        tmp_acce = np.copy(imu_all[['acce_x', 'acce_y', 'acce_z']].values)

        tmp_vio_gyro_bias = np.copy(imu_all[['vio_gyro_bias_x', 'vio_gyro_bias_y', 'vio_gyro_bias_z']].values)
        tmp_vio_acce_bias = np.copy(imu_all[['vio_acce_bias_x', 'vio_acce_bias_y', 'vio_acce_bias_z']].values)

        tmp_gyro = tmp_gyro - tmp_vio_gyro_bias[-1, :]
        tmp_acce = tmp_acce - tmp_vio_acce_bias[-1, :]

        # vio_q = interp1d(tmp_ts, tmp_vio_q, axis=0)(ts) # w, x, y, z interpolate may be wrong
        tmp_vio_q_R = Rotation.from_quat(tmp_vio_q[:, [1, 2, 3, 0]])
        vio_q = Slerp(tmp_ts, tmp_vio_q_R)(ts)
        vio_q = vio_q.as_quat()[:, [3, 0, 1, 2]]

        vio_p = interp1d(tmp_ts, tmp_vio_p, axis=0)(ts)
        gyro = interp1d(tmp_ts, tmp_gyro, axis=0)(ts)
        acce = interp1d(tmp_ts, tmp_acce, axis=0)(ts)

        #---velocity---
        dt = (ts[self.w:] - ts[:-self.w])[:, None]
        glob_v = (vio_p[self.w:] - vio_p[:-self.w]) / dt

        # transformation
        ori_R = Rotation.from_quat(vio_q[:, [1, 2, 3, 0]]) # x, y, z, w
        glob_gyro = np.einsum("tip,tp->ti", ori_R.as_matrix(), gyro)
        glob_acce = np.einsum("tip,tp->ti", ori_R.as_matrix(), acce)

        start_frame = self.info.get('start_frame', 0)
        self.ts = ts[start_frame:]
        self.features = np.concatenate([glob_gyro, glob_acce], axis=1)[start_frame:]
        # self.targets = glob_v[start_frame:, :2] # change here for 3 dimensions
        self.targets = glob_v[start_frame:, :3] # change here for 3 dimensions
        self.orientations = ori_R.as_quat()[:, [3, 0, 1, 2]][start_frame:]  # as order of w, x, y, z
        self.gt_pos = vio_p[start_frame:]

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts[:, None], self.orientations, self.gt_pos], axis=1)

    def get_meta(self):
        return '{}: time duration: {}'.format(
            self.info['path'], self.info['time_duration'])


class DenseSequenceDataset(Dataset):
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=10, window_size=200,
                 random_shift=0, transform=None, **kwargs):
        super().__init__()
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []
        self.ts, self.orientations, self.gt_pos = [], [], []

        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, interval=1, **kwargs)

        # Optionally smooth the sequence
        feat_sigma = kwargs.get('feature_sigma,', -1)
        targ_sigma = kwargs.get('target_sigma,', -1)
        if feat_sigma > 0:
            self.features = [gaussian_filter1d(feat, sigma=feat_sigma, axis=0) for feat in self.features]
        if targ_sigma > 0:
            self.targets = [gaussian_filter1d(targ, sigma=targ_sigma, axis=0) for targ in self.targets]

        for i in range(len(data_list)):
            self.ts.append(aux[i][:, 0])
            self.orientations.append(aux[i][:, 1:5])
            self.gt_pos.append(aux[i][:, -3:])
            self.index_map += [[i, j] for j in range(window_size, self.targets[i].shape[0], step_size)]

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = self.features[seq_id][frame_id - self.window_size:frame_id]
        targ = self.targets[seq_id][frame_id]

        if self.transform is not None:
            feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32).T, targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)


class StridedSequenceDataset(Dataset):
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=10, window_size=200,
                 random_shift=0, transform=None, **kwargs):
        super(StridedSequenceDataset, self).__init__()
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform
        self.interval = kwargs.get('interval', window_size)

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []
        self.ts, self.orientations, self.gt_pos = [], [], []
        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, interval=self.interval, **kwargs)
        for i in range(len(data_list)):
            self.ts.append(aux[i][:, 0])
            self.orientations.append(aux[i][:, 1:5])
            self.gt_pos.append(aux[i][:, -3:])
            self.index_map += [[i, j] for j in range(0, self.targets[i].shape[0], step_size)]

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = self.features[seq_id][frame_id:frame_id + self.window_size]
        targ = self.targets[seq_id][frame_id]

        if self.transform is not None:
            feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32).T, targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)


class SequenceToSequenceDataset(Dataset):
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=100, window_size=400,
                 random_shift=0, transform=None, **kwargs):
        super(SequenceToSequenceDataset, self).__init__()
        self.seq_type = seq_type
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []

        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, **kwargs)

        # Optionally smooth the sequence
        feat_sigma = kwargs.get('feature_sigma,', -1)
        targ_sigma = kwargs.get('target_sigma,', -1)
        if feat_sigma > 0:
            self.features = [gaussian_filter1d(feat, sigma=feat_sigma, axis=0) for feat in self.features]
        if targ_sigma > 0:
            self.targets = [gaussian_filter1d(targ, sigma=targ_sigma, axis=0) for targ in self.targets]

        max_norm = kwargs.get('max_velocity_norm', 3.0)
        self.ts, self.orientations, self.gt_pos, self.local_v = [], [], [], []
        for i in range(len(data_list)):
            self.features[i] = self.features[i][:-1]
            self.targets[i] = self.targets[i]
            self.ts.append(aux[i][:-1, :1])
            self.orientations.append(aux[i][:-1, 1:5])
            self.gt_pos.append(aux[i][:-1, 5:8])

            velocity = np.linalg.norm(self.targets[i], axis=1)  # Remove outlier ground truth data
            bad_data = velocity > max_norm
            for j in range(window_size + random_shift, self.targets[i].shape[0], step_size):
                if not bad_data[j - window_size - random_shift:j + random_shift].any():
                    self.index_map.append([i, j])

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        # output format: input, target, seq_id, frame_id
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = np.copy(self.features[seq_id][frame_id - self.window_size:frame_id])
        targ = np.copy(self.targets[seq_id][frame_id - self.window_size:frame_id])

        if self.transform is not None:
            feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32), targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)

    def get_test_seq(self, i):
        return self.features[i].astype(np.float32)[np.newaxis,], self.targets[i].astype(np.float32)
