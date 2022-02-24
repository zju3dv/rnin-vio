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
import yaml
import os
import re
import train, test
from model import model_lstm
import logging
import torch
import torch.distributed as dist
import torch.nn as nn

# General config
def load_config(path):
    ''' Loads config file.
    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)

    return cfg_special

def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

def build_model(args, cfg):
    device = torch.device(
        f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
    )

    with open(cfg['model']['model_yaml'], 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)
    update_recursive(cfg, cfg_special)

    model_name = cfg['model']['model_name']
    if model_name == 'resnet_lstm':
        model = model_lstm.ResNetLSTMSeqNet(cfg)

    ## multi gpu?
    if cfg['train']['use_multi_gpu']:
        logging.info(f"torch.cuda.device_count() {torch.cuda.device_count()}")
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        dist.barrier()
        # SyncBN
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device).cuda(args.local_rank)
        network = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                            broadcast_buffers=False, find_unused_parameters=True)
        total_params = network.module.get_num_params()
    else:
        network = model.to(device)
        total_params = network.get_num_params()

    logging.info(f'Network "{model_name}" loaded to device {device}, device num {torch.cuda.device_count()}')
    logging.info(f"Total number of parameters: {total_params}")

    return network

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def str2int(v_str):
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

def GetBestModel(path):
    names = sorted(os.listdir(path+"/"), key=str2int)
    files=[]
    for name in names:
        if os.path.isfile(os.path.join(os.path.abspath(path), name)):
            files.append(name)
    # files.sort()
    model = os.path.join(os.path.abspath(path), files[-1])
    logging.info(f"load model: {model}")
    return model

def build_trainer(args, cfg, model, **kwargs):
    start_epoch = 0
    optim = torch.optim.Adam if cfg['train']['optimizer']['method'] == 'Adam' else torch.optim.SGD
    optimizer = optim(model.parameters(), cfg['train']['optimizer']['learning_rate'],
                      weight_decay=cfg['train']['optimizer']['weight_decay'])

    if cfg['train']['use_pretrain_model']:
        checkpoint = torch.load(GetBestModel(os.path.join(cfg['train']['out_dir'], "checkpoints")))
        start_epoch = checkpoint.get("epoch", 0)
        if cfg['train']['use_multi_gpu']:
            model.module.load_state_dict(checkpoint.get("model_state_dict"))
        else:
            model.load_state_dict(checkpoint.get("model_state_dict"))
        optimizer.load_state_dict(checkpoint.get("optimizer_state_dict"))
        logging.info(f"Continue from epoch {start_epoch}")

    return train.trainer(args, cfg, model=model, optimizer=optimizer, start_epoch=start_epoch)

def build_tester(args, cfg, model, **kwargs):
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(GetBestModel(os.path.join(cfg['train']['out_dir'], "checkpoints")), map_location=device)
    if cfg['train']['use_multi_gpu']:
        model.module.load_state_dict(checkpoint.get("model_state_dict"))
    else:
        model.load_state_dict(checkpoint.get("model_state_dict"))
    model.eval()

    tester = test.tester(args, cfg, model)

    return tester
