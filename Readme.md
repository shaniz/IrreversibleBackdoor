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
For Pre-trained Models**. Original implementation: [https://github.com/ChiangE/Sophon].

2. 'irreversible_backdoor' folder: An implementation of our paper - ******. Drawing ideas from Sophon implementation
It implements a learning algorithm used to prevent a targeted backdoor removal later finetune process on foundation models. 
It uses same ideas from Sophon paper, with updated loss function to achieve the new goal. The difference is in the maml part (FTS loop) - instead of adapting on clean dataset and 'punish' according to the loss function in the paper (inverse loss or kl divergence on  clean dataset), we are adapting on clean dataset and the loss function is classic but calculated on a 100% poisoned dataset. 

Each main folder (sophon_orig, irreversible_backdoor) contains 3 subfolders:
1. stage1_pretrain - pretrain model on origin domain (clean/poisoned depend on project).
2. stage2_train - train the model according the new training .
3. stage3_eval - evaluation script.


### Supported architectures
`['resnet18', 'resnet34', 'resnet50', 'caformer']`.
### Supported datasets
`['CIFAR10', 'MNIST', 'SVHN']`.

Those are the datasets for the latter fine-tuning the foundation model. Dataset for the first pretraining step is always `ImageNette`.  


## Preparation

### Installing requirements by:
```bash
pip install -r requirements.txt
```

### Download Imagenette dataset:
From [https://github.com/fastai/imagenette]. 
You have download links under the Image section, place it under 'datasets' folder.
Extract and place the inside 'imagenette2' folder in the 'datasets' folder.


## Usage - Sophon

Workspace is `./sophon_orig`, thus

```bash
cd sophon_orig
```

### Pretrain Model

```bash
python  stage1_pretrain/train.py
```
Result pretrained model is saved under 'stage1_pretrain/pretrained_models'.

### Train Sophon Model
Update MODEL_PATH in algo.py to pretrained model from previous section (placed in pretrained_models).

For inverse cross-entropy sophon, run:

```bash
python stage2_train/algo.py --alpha 3 --beta 5 --datasets CIFAR10 --arch resnet18
```

The output ckpt will be saved to `sophon_models/inverse_loss/[args.arch]/[args.dataset]/[current_time]/checkpoints/`
Results is saved under MODEL_PATH folder - `sophon_models/inverse_loss/[args.arch]/[args.dataset]/[current_time]/`.


For kl divergence from uniform distribution sophon, run:

```bash
python stage2_train/algo.py --alpha 3 --beta 5 --datasets CIFAR10 --arch resnet18 --loss_type kl
```
The output ckpt will be saved to `sophon_models/kl_loss/[args.arch]/[args.dataset]/[current_time]/checkpoints/`
Results is saved under MODEL_PATH folder - `sophon_models/kl_loss/[args.arch]/[args.dataset]/[current_time]/`


### Evaluation - Test Finetune

For testing a target ckpt's finetune outcome directly:
Update MODEL_PATH to result Sophon model from previous section.
Run:

```bash
# for finetuned ckpt
python stage3_eval/finetune_test.py --start sophon --path path_to_ckpt
```


## Usage - Irreversible Backdoor

Workspace is `./irreversible_backdoor`, thus

```bash
cd irreversible_backdoor
```

### Pretrain Model

Pretrain a model with target backdoor:
```bash
python  stage1_pretrain/train_with_backdoor.py
```
Result pretrained model is saved under `stage1_pretrain/pretrained_backdoor_models`.

You can analyze pretrain model by updating MODEL_PATH in `stage3_eval/eval_backdoor_ASR.py` and running:
```bash
python  stage3_eval/eval_backdoor_ASR.py
```

The output result will be saved in the MODEL_PATH folder.


### Train Irreversible Backdoor Model
Update MODEL_PATH in `stage2_train/bd_algo.py` to the pretrained backdoor model from previous section (placed in `stage1_pretrain/pretrained_backdoor_models`).
```bash
python stage2_train/bd_algo.py --alpha 3 --beta 5 --datasets CIFAR10 --arch resnet18
```

The output ckpt will be saved to `irreversible_backdoor_models/targeted_backdoor_loss/[args.arch]/[args.dataset]/[current_time]/checkpoints/`.


### Evaluation - Test finetune
Update MODEL_PATH in `stage3_eval/eval_backdoor_ASR.py` to result irreversible backdoor model from previous section and run:

!!Notice!!: generally, MODEL_PATH should be the final checkpoint created in the previous part (`final_` prefix in model file). 
If the final created model was created after a drop in original accuracy, take one previous checkpoint from the final one (ASR after finetune should be higher).
If you see that the final ASR (appears in file name) has a drop at some point, take one previous checkpoint before the drop.

```bash
python stage3_eval/eval_backdoor_ASR.py
```
The output result will be saved under MODEL_PATH folder `irreversible_backdoor_models/targeted_backdoor_loss/[args.arch]/[args.dataset]/[current_time]/`.

Results contains:
1. Clean dataset accuracy before and after finetune.
2. Clean dataset accuracy + targeted ASR during finetune process.
