# RNIN-VIO: Robust Neural Inertial Navigation Aided Visual-Inertial Odometry in Challenging Scenes
### [Project Page](https://zju3dv.github.io/rnin-vio/) | [Paper](http://www.cad.zju.edu.cn/home/gfzhang/papers/rnin_vio.pdf) | [Video](http://www.cad.zju.edu.cn/home/gfzhang/papers/rnin_vio.mp4)
<br/>

> RNIN-VIO: Robust Neural Inertial Navigation Aided Visual-Inertial Odometry in Challenging Scenes  
> Danpeng Chen<sup>1,2</sup>, Nan Wang<sup>2</sup>, [Runsen Xu](https://scholar.google.com.hk/citations?user=MOobrCcAAAAJ&hl=zh-CN&oi=ao) <sup>1</sup>, Weijian Xie<sup>1,2</sup>, [Hujun Bao](http://www.cad.zju.edu.cn/bao/) <sup>1</sup>, [Guofeng zhang](http://www.cad.zju.edu.cn/home/gfzhang/) <sup>1*</sup>  
> <sup>1</sup>State Key Lab of CAD & CG, Zhejiang University, <sup>2</sup>SenseTime Research and Tetras.AI  
> ISMAR 2021 (Oral Presentation)

<br/>
This code is the inertial neural network of the paper RNIN-VIO. 

## Installation
```shell
# Ubuntu 16.04 and above is recommended.  
# First create a virtual environment with python3 interpreter, then run
pip install -r requirements.txt
```

## Our Dataset
You can download our dataset from any of the following address:
  
- [drive.google.com](https://drive.google.com/file/d/1HfuZYnSdeCiFsqkP57Jn9i_y22kpQ7xp/view)  
- [pan.baidu.com](https://pan.baidu.com/s/1wj5YeMah2N7Olka7MoeJEg) (password: y9dg)


Our dataset is mainly composed of two parts, including [IDOL](https://zenodo.org/record/4484093) open source 20 hours of data and 7 hours of data collected by ourselves. 
IDOL mainly includes some simple plane movements. 
To increase the diversity of sports and equipments, we use multiple smartphones, such as Huawei, Xiaomi, OPPO, etc., to collect data with cameras and IMUs. 
The full dataset was captured by five people and includes a variety of sports, including walking, running, standing still, going up and down stairs, and random shaking, etc. 
We use the BVIO to provide the positions aligned with gravity at IMU frequency on the dataset. 
Part of the data is collected in the VICON room, so it has high-precision trajectory provided by VICON. 
We release our dataset collected by ourselves. The data is stored in `SenseINS.csv`. 
The format of files are as follow:
```shell
Data
|--data_train
|   |--0
|     |--SenseINS.csv
|   |--...
|--data_val
|   |--0
|     |--SenseINS.csv
|   |--...
|--data_test
|   |--0
|     |--SenseINS.csv
|   |--...
|--Sense_INS_Data.md     // The format description of SenseINS.csv
```

## Training or Testing on SenseINS data

### Configure yaml file(config/default.yaml)
`train_dir`: path of training dataset  
`test_dir`: path of testing dataset  
`validation_dir`: path of validation dataset  
`train::out_dir`: training output directory  
`test::out_dir`: testing output directory  
`schemes::train`: training a network  
`schemes::test`: testing a network

### runing
`python main_net.py --yaml ./config/default.yaml`

## RONIN-3D
The original [RONIN](https://github.com/Sachini/ronin) only predict 2D motion. For a fair comparison with our method, we extend RONIN to 3D. The relevant code is in the ```ronin_3d``` folder.

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@article{chen2021rninvio,
  title={{RNIN-VIO}: Robust Neural Inertial Navigation Aided Visual-Inertial Odometry in Challenging Scenes},
  author={Danpeng Chen, Nan Wang, Runsen Xu, Weijian Xie, Hujun Bao, and Guofeng Zhang},
  journal={In Proceedings of 2021 IEEE International Symposium on Mixed and Augmented Reality},
  year={2021}
}
```

## Acknowledgment
The authors are very grateful to Shangjin Zhai, Chongshan Sheng, Yuequ Cai, and Kai Sun for their kind help in developing RNIN-VIO system.

## Copyright
This work is affiliated with ZJU-SenseTime Joint Lab of 3D Vision, and its intellectual property belongs to SenseTime Group Ltd.

```
Copyright (c) ZJU-SenseTime Joint Lab of 3D Vision. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
