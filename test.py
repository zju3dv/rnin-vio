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
from tqdm import tqdm
import json
import os
from os import path as osp
import torch
from model import function
from torch.utils.data import DataLoader
import logging
from utils.metric import compute_ate_rte
from utils import postprocess
from dataloader import dataset as dataset_utils

def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()

class tester(object):
    def __init__(self, args, cfg, model):
        super(tester, self).__init__()
        self.cfg = cfg
        self.device = torch.device(
            f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self.model = model
        self.out_dir = cfg['test']['out_dir']

        self.pred_velocity = cfg['model']['pred_velocity']
        self.window_time = cfg['model_param']['window_time']
        self.start_cov_epochs = cfg['train']['start_cov_epochs']
        self.plot_cnt = 0

    def inference_step(self, data_loader, epoch):
        targets_all, preds_all, preds_cov_all, preds_model_cov_all, losses_all, \
        mse_recovers, pred_aug_error_all = [], [], [], [], [], [], []
        for bid, batch in tqdm(enumerate(data_loader)):
            batch = [t.to(self.device) for t in batch]

            pred, pred_cov, targ, loss = \
                function.fun_test_forward(self.cfg, self.model, batch, self.start_cov_epochs, epoch)

            targets_all.append(torch_to_numpy(targ))
            preds_all.append(torch_to_numpy(pred))
            preds_cov_all.append(torch_to_numpy(pred_cov))
            losses_all.append(np.mean(torch_to_numpy(loss)))

        targets_all = np.concatenate(targets_all, axis=0)
        preds_all = np.concatenate(preds_all, axis=0)
        preds_cov_all = np.concatenate(preds_cov_all, axis=0)
        attr_dict = {
            "targets": targets_all,
            "preds": preds_all,
            "preds_cov": preds_cov_all,
            "losses": losses_all,
        }

        return attr_dict

    def test(self, test_data_path):

        ate_all, t_rte_all, d_rte_all = [], [], []
        all_metrics = {}
        for data in test_data_path:
            logging.info(f"Processing {data}...")
            test_basic_data = dataset_utils.BasicSequenceData(self.cfg, [data], mode="test")
            test_dataset = dataset_utils.SeqToSeqDataset(self.cfg, test_basic_data, test_basic_data.get_merged_index_map(),
                                                         mode="test")

            test_loader = DataLoader(test_dataset, batch_size=self.cfg['test']['batch_size'], shuffle=False)

            data_name = data.split('/')[-1]
            outdir = osp.join(self.out_dir, data_name)
            if osp.exists(outdir) is False:
                os.mkdir(outdir)

            net_attr_dict = self.inference_step(test_loader, epoch=1000)

            traj_attr_dict = postprocess.pose_integrate(self.cfg, test_dataset, net_attr_dict["preds"])
            outfile = osp.join(outdir, "trajectory.txt")
            trajectory_data = np.concatenate(
                [
                    traj_attr_dict["ts"].reshape(-1, 1),
                    traj_attr_dict["pos_pred"],
                    traj_attr_dict["pos_gt"],
                ], axis=1,)
            np.savetxt(outfile, trajectory_data, delimiter=",")

            # obtain metrics
            plot_dict = postprocess.compute_plot_dict(
                self.cfg['data']['sample_freq'], net_attr_dict, traj_attr_dict
            )

            outfile_net = osp.join(outdir, "net_outputs.txt")
            net_outputs_data = np.concatenate(
                [
                    plot_dict["pred_ts"].reshape(-1, 1),
                    plot_dict["preds"],
                    plot_dict["targets"],
                    plot_dict["pred_sigmas"],
                ],
                axis=1,
            )
            np.savetxt(outfile_net, net_outputs_data, delimiter=",")

            # plot
            postprocess.make_plots(plot_dict, outdir)

            # compute ate and rte
            ate, t_rte, d_rte = compute_ate_rte(traj_attr_dict["pos_pred"], traj_attr_dict["pos_gt"],
                                                int(self.cfg['data']['imu_freq'] * 60))
            all_metrics["data"] = {
                "data": data_name,
                "ate": ate,
                "t_rte": t_rte,
                "d_rte": d_rte,
            }

            logging.info(f"data {data_name}, ate: {ate}, t_rte {t_rte}, d_rte {d_rte}")

            ate_all.append(ate)
            t_rte_all.append(t_rte)
            d_rte_all.append(d_rte)

        all_metrics["all"] = {
            "avg_ate": float(np.mean(ate_all)),
            "avg_t_rte": float(np.mean(t_rte_all)),
            "avg_d_rte": float(np.mean(d_rte_all)),
        }
        with open(self.out_dir + "/metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=1)

        print('----------\navg ATE:{}, avg T_RTE:{}, avg D_RTE:{}'.format(
            np.mean(ate_all), np.mean(t_rte_all), np.mean(d_rte_all)))
