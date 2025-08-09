# SOPHON: Non-Fine-Tunable Learning to Restrain Task Transferability For Pre-trained Models

[https://arxiv.org/abs/2404.12699](https://arxiv.org/abs/2404.12699)

Jiangyi Deng (1), Shengyuan Pang (1), Yanjiao Chen (1), Liangming Xia (1), Yijie Bai (1), Haiqin Weng (2), Wenyuan Xu (1) ((1) Zhejiang University, (2) Ant Group)

**Accepted by IEEE Symposium on Security and Privacy 2024**

## Table of Contents
+ [Introduction](https://github.com/shaniz/Sophon/blob/ee37552f6abc8f0c26003c6bdc5ffb0dce590398/Readme.md#L17)
+ [Preparation](https://github.com/shaniz/Sophon/blob/ee37552f6abc8f0c26003c6bdc5ffb0dce590398/Readme.md#L32)
+ [Usage- Sophon](https://github.com/shaniz/Sophon/blob/7904899cc9cef93f63d5149e7abf248f642ba5d3/Readme.md#L44)
+ [Usage - Irreversible Backdoor](https://github.com/shaniz/Sophon/blob/7904899cc9cef93f63d5149e7abf248f642ba5d3/Readme.md#L93)

## Introduction
This repo contains:

1. 'sophon_orig' folder: A refactored implementation of the classification part in paper: **SOPHON: Non-Fine-Tunable Learning to Restrain Task Transferability
For Pre-trained Models** Original implementation: [https://github.com/ChiangE/Sophon].
<img src="https://github.com/Sophon-NonFinetunableLearning/Sophon/blob/main/sophon.png" width="400" align="center"/>

2. 'irreversible backdoor' folder: An implementation of our paper - ******. Drawing ideas from Sophon implementation
It implements a learning algorithm used to prevent a targeted backdoor removal later finetune process on foundation models. 
It uses same ideas from Sophon paper, with updated loss function to achieve the new goal.



## Preparation

### Installing requirements by:
```bash
pip install -r requirements.txt
```

### Download Imagenette dataset:
From [https://github.com/fastai/imagenette]. 
You have download links under the Image section, place it under 'datasets' folder.


## Usage - Sophon

Workspace is ``./sophon_orig``, thus

```bash
cd sophon_orig
```

#### Pretrain Model

```bash
python  test/train.py
```
Result pretrained model is saved under 'pretrained_models' folder.

#### Train Sophon Model

For inverse cross-entropy sophon, run:

```bash
python algo.py --alpha 3 --beta 5 --datasets CIFAR10 --arch res18
```

The output ckpt will be saved to `sophon_models/inverse_loss/[args.arch]_[args.dataset]/[current_time]/`

For kl divergence from uniform distribution sophon, run:

```bash
python algo.py --alpha 3 --beta 5 --datasets CIFAR10 --arch res18 --loss_type kl
```
The output ckpt will be saved to `sophon_models/kl_loss/[args.arch]_[args.dataset]/[current_time]/`


#### Test finetune

For test a target ckpt's finetune outcome directly:
Update MODEL_PATH to result Sophon model from previous section.
Run:

```bash
# for finetuned ckpt
python finetune_test.py --start sophon --path path_to_ckpt
```




## Usage - Irreversible Backdoor

Workspace is ``./irreversible_backdoor``, thus

```bash
cd irreversible_backdoor
```

#### Pretrain Model

Pretrain a model with target backdoor:
```bash
python  test/train_with_backdoor.py
```
Result pretrained model is saved under 'pretrained_backdoor_models' folder.

You can calculate ASR by updating MODEL_PATH in test/eval_backdoor_ASR.py and running:
```bash
python  test/eval_backdoor_ASR.py
```

The output result will be saved to `results/ASR/[MODEL_PATH].csv`


#### Train Irreversible Backdoor Model

```bash
python bd_algo.py --alpha 3 --beta 5 --datasets CIFAR10 --arch res18
```

The output ckpt will be saved to `irreversible_backdoor_models/irreversible_backdoor_loss/[args.arch]_[args.dataset]/[current_time]/`


#### Test finetune
Update MODEL_PATH in test/eval_backdoor_ASR_after_finetune.py to result irreversible backdoor model from previous section and run:

```bash
python test/eval_backdoor_ASR_after_finetune.py
```
The output result will be saved to `results/ASR-after-finetune/[MODEL_PATH]/[current_time]/`
Results contains:
1. Clean dataset accuracy before and after finetune.
2. Clean dataset accuracy + targeted ASR during finetune process.












