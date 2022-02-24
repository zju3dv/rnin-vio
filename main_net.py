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
import os
import sys
import numpy as np
import logging
import torch
import random
from config import configer
from dataloader import dataset as dataset_utils
from torch.utils.data import DataLoader
import torch.distributed as dist
logging.basicConfig(
    stream=sys.stdout,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.INFO,
)

def GetFolderName(path):
    names = os.listdir(path+"/")
    folders=[]
    for name in names:
        if os.path.isdir(os.path.join(os.path.abspath(path), name)):
            folders.append(name)
    folders.sort()
    return folders

def GetDataPath(path):
    names = os.listdir(path+"/")
    folders=[]
    for name in names:
        data_path = os.path.join(os.path.abspath(path), name)
        if os.path.isdir(data_path):
            folders.append(data_path)
    folders.sort()
    return folders

def WriteList(path, name, folders):
    with open(path+"/"+name, 'w') as f:
        for folder in folders:
            f.writelines(folder+"\n")
        f.close()

def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_output_dir(out_dir):
    try:
        if out_dir is not None:
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            if not os.path.isdir(os.path.join(out_dir, "checkpoints")):
                os.makedirs(os.path.join(out_dir, "checkpoints"))
            if not os.path.isdir(os.path.join(out_dir, "logs")):
                os.makedirs(os.path.join(out_dir, "logs"))
            logging.info(f"Training output writes to {out_dir}")
        else:
            raise ValueError("out_dir must be specified.")
    except ValueError as e:
        logging.error(e)
        return

def train_load_data(cfg):
    train_loader, val_loader, test_loader = None, None, None

    def _init_fn(worker_id):
        np.random.seed(int(cfg['seeds']['id']) + worker_id)

    if cfg['data']['random_partition']:
        ## random select train, val, test data
        logging.info(f"random select train, val, test data")
        basic_data = dataset_utils.BasicSequenceData(cfg, train_data_path, mode="train")
        train_index_map, valid_index_map = \
            dataset_utils.partition_data(basic_data.get_index_map(),
                                         valid_samples=basic_data.valid_samples,
                                         valid_all_samples=basic_data.valid_all_samples,
                                         data_paths=basic_data.data_paths,
                                         out_path=cfg['data']['validation_dir'],
                                         training_rate=cfg['data']['train_rate'],
                                         valuation_rate=cfg['data']['valid_rate'],
                                         data_rate=cfg['data']['data_rate'])
        train_dataset = dataset_utils.SeqToSeqDataset(cfg, basic_data, train_index_map, mode="train")
        val_dataset = dataset_utils.SeqToSeqDataset(cfg, basic_data, valid_index_map, mode="val")
        val_loader = DataLoader(val_dataset, batch_size=cfg['val']['batch_size'], shuffle=False)
    else:
        ## train data
        logging.info(f"process train data")
        train_basic_data = dataset_utils.BasicSequenceData(cfg, train_data_path, mode="train")
        train_dataset = dataset_utils.SeqToSeqDataset(cfg, train_basic_data, train_basic_data.get_merged_index_map(),
                                                      mode="train")

        ## val data
        if cfg['data']['validation_dir'] is not None:
            logging.info(f"process val data")
            valid_data_path = GetDataPath(cfg['data']['validation_dir'])
            valid_basic_data = dataset_utils.BasicSequenceData(cfg, valid_data_path, mode="val")
            val_dataset = dataset_utils.SeqToSeqDataset(cfg, valid_basic_data, valid_basic_data.get_merged_index_map(),
                                                        mode="val")
            val_loader = DataLoader(val_dataset, batch_size=cfg['val']['batch_size'], shuffle=False)

    ## test data
    if cfg['data']['test_dir'] is not None:
        logging.info(f"process test data")
        test_data_path = GetDataPath(cfg['data']['test_dir'])
        test_basic_data = dataset_utils.BasicSequenceData(cfg, test_data_path, mode="test")
        test_dataset = dataset_utils.SeqToSeqDataset(cfg, test_basic_data, test_basic_data.get_merged_index_map(),
                                                     mode="test")
        test_loader = DataLoader(test_dataset, batch_size=cfg['test']['batch_size'], shuffle=False)

    if cfg['train']['use_multi_gpu']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # train_sampler = None
        train_loader = DataLoader(
            train_dataset, batch_size=cfg['train']['batch_size'], shuffle=(train_sampler is None), pin_memory=True,
            num_workers=4, sampler=train_sampler, worker_init_fn=_init_fn)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=cfg['train']['batch_size'],
            shuffle=True, pin_memory=True, num_workers=cfg['train']['n_workers'], worker_init_fn=_init_fn)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--yaml", type=str, default="./config/default.yaml")

    args = parser.parse_args()
    logging.info(f"args.local_rank: {args.local_rank}")

    cfg = configer.load_config(args.yaml)
    device = torch.device(
        f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
    )
    logging.info(f'loaded to device {device}')
    ###########################################################
    # Main
    ###########################################################
    if cfg['seeds']['use_seeds']:
        set_seeds(cfg['seeds']['id'])
    model = configer.build_model(args, cfg)
    if cfg['train']['use_multi_gpu']:
        cfg['train']['optimizer']['learning_rate'] = cfg['train']['optimizer']['learning_rate'] * dist.get_world_size()

    if cfg['schemes']['train']:
        if args.local_rank == 0:
            create_output_dir(cfg['train']['out_dir'])
            ## copy yaml
            cmd = "cp " + args.yaml + " " + cfg['train']['out_dir'] + "/default.yaml"
            os.system(cmd)
            cmd = "cp " + cfg['model']['model_yaml'] + " " + cfg['train']['out_dir'] + "/model.yaml"
            os.system(cmd)
        ## check
        if cfg['data']['train_dir'] is None:
            raise ValueError("train_dir must be specified.")
        train_data_path = GetDataPath(cfg['data']['train_dir'])
        ## load data
        train_loader, val_loader, test_loader = train_load_data(cfg)
        ## train
        trainer = configer.build_trainer(args, cfg, model)
        trainer.train(train_loader, val_loader, test_loader)

    if cfg['schemes']['test']:
        if cfg['data']['test_dir'] is None:
            raise ValueError("test_dir must be specified.")
        test_data_path = GetDataPath(cfg['data']['test_dir'])
        if cfg['test']['out_dir'] is None:
            raise ValueError("out_dir must be specified.")
        if args.local_rank == 0:
            if os.path.exists(cfg['test']['out_dir']) is False:
                os.mkdir(cfg['test']['out_dir'])
        tester = configer.build_tester(args, cfg, model)
        tester.test(test_data_path)
