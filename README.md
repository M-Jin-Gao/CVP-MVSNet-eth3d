## Cost Volume Pyramid Based Depth Inference for Multi-View Stereo (ETH3D)


This repository offers training and evaluation code for the CVP-MVSNet on the ETH3D dataset

## Contents
0. [Pre-requisites](#Pre-requisites)
0. [Download dataset](#Download_dataset)
0. [Install_requirements](#Install_requirements)
0. [Training_and_validation](#Training_and_validation)
0. [Test](#Test)
0. [Generate_point_clouds_and_reproduce ETH3d results](#Generate_point_clouds_and_reproduce_ETH3d_results)
0. [Acknowledgment](#Acknowledgment)

## 0. Pre-requisites

* Nvidia GPU with 11GB or more vRam.
* CUDA 10.1
* python3.6
* python2.7 for fusion script

## 1. Download_dataset

Training data(5G):

Download the pre-processed ETH3D training data from [here](https://drive.google.com/file/d/1EMb65HxFQG8NiJMBJ6C_cm8ohLFOFrcf/view) and extract it to `CVP_MVSNet/dataset/eth3d_train`.

Test data(0.9G):

Download the pre-processed ETH3D test data from [here](https://drive.google.com/file/d/1lQ-UloHlFL1zeE6InVCrBJQLtaZvHbZU/view) and extract it to `CVP_MVSNet/dataset/eth3d_test`.

## 2. Install_requirements

`cd CVP_MVSNet_eth3d`

`pip3 install -r requirements.txt`

## 3. Training_and_validation
use the following code to train and validate the model
```bash
python3 train.py --info="train_eth" --mode="train" --dataloader="eth" \
\
--dataset_root="./dataset/eth3d_train/" --nsrc=2 --nscale=2 \
\
--epochs=40 --lr=0.001 --lrepochs="10,12,14,20:2" --batch_size=2 \
\
--loadckpt="" --logckptdir="./checkpoints/" --loggingdir="./logs/" --resume=0 \
```
For the validation, it needs to fill in the path of the checkpoint.
```bash

python3 eval.py --info="eval_eth" --mode="test" --dataloader="eth" \
\
--dataset_root="./dataset/eth3d_test/" --nsrc=4 --nscale=2 --batch_size=1 \
\
--loadckpt="" --loggingdir="./logs/" --outdir="./outputs_pretrained/" \
\
```

## 4. Test
Fill in the path of the checkpoint and use the following code to test the model on the test set

```bash
python3 validate.py --info="val_eth" --mode="val" --dataloader="eth" \
\
--dataset_root="./dataset/eth3d_train/" --nsrc=4 --nscale=2 --batch_size=1 \
\
--loadckpt="" --loggingdir="./logs/" --outdir="./outputs_pretrained/" \
\
```

## 5. Generate_point_clouds_and_reproduce_ETH3d_results


Check out Yao Yao's modified version of fusibile

`git clone https://github.com/YoYo000/fusibile`

Install fusibile by `cmake .` and `make`, which will generate the executable at`FUSIBILE_EXE_PATH`


Install extra dependencies

`pip2 install -r CVP_MVSNet/fusion/requirements_fusion.txt`

Fill in the fusibile_exe_path and use following code to generate point clouds. 
```bash
python2 depthfusion.py \
--dtu_test_root="../dataset/eth3d_train" \
--depth_folder="/content/outputs_pretrained/" \
--out_folder="fusibile_fused" \
--fusibile_exe_path="" \
--prob_threshold=0.98 \
--disp_threshold=0.04 \
--num_consistent=3
```


Evaluate the point clouds using the [ETH3D evaluation code](https://github.com/ETH3D/multi-view-evaluation).



## Acknowledgment

This repository is based on the [CVP-MVSNet](https://github.com/JiayuYANG/CVP-MVSNet) repository by Jiayu Yang. Many thanks to Jiayu Yang for the great paper and great code!

