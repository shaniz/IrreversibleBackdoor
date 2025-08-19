import random
import os
from datetime import datetime
import argparse
import sys
import learn2learn as l2l
import copy

import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from sophon_orig.stage2_train.model_utils import save_bn, load_bn, get_pretrained_model, set_seed
from sophon_orig.stage2_train.save_utils import save_args_to_file, save_model
from sophon_orig.stage2_train.eval_utils import evaluate
from bd_fast_adapt_utils import fast_adapt_punish_if_backdoor_fails
from bd_eval_utils import evaluate_backdoor_after_finetune
from bd_dataset_utils import get_dataset, CircularDualDataloader
from bd_save_utils import bd_save_data


MODEL_PATH = '../stage1_pretrain/pretrained_backdoor_models/resnet18/ImageNette/8-13_22-38-5/checkpoints/resnet18_ImageNette_ep-20_bd-train-acc99.25_clean-test-acc89.172.pth'
TARGET_LABEL = 0
TRIGGER_SIZE = 5
POISON_PERCENT = 0.1
ARGS_FILE = "train_args.json"
ORIG_DATA_DIR = '../../datasets/imagenette2'
ORIG_DATASET = 'ImageNette'
RESTRICT_DATA_DIR = '../../datasets'
CHECKPOINTS_SUBDIR = 'checkpoints'


sys.path.append('/')
def args_parser():
    parser = argparse.ArgumentParser(description='train N shadow models')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--bs', default=150, type=int)
    parser.add_argument('--fts_loop', default=1, type=int)
    parser.add_argument('--ntr_loop', default=1, type=int)
    # parser.add_argument('--total_loop', default=1000, type=int)
    parser.add_argument('--total_loop', default=300, type=int)
    parser.add_argument('--alpha', default=3.0, type=float, help='coefficient of maml lr')
    parser.add_argument('--beta', default=5.0, type=float, help='coefficient of natural lr')
    parser.add_argument('--test_iterval', default=25, type=int)
    parser.add_argument('--arch', default='resnet18', type=str, choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--dataset', default='CIFAR10', type=str, choices=['CIFAR10', 'MNIST', 'SVHN'])
    # parser.add_argument('--finetune_epochs', default=5, type=int)
    parser.add_argument('--finetune_epochs', default=20, type=int)
    parser.add_argument('--final_finetune_epochs', default=20, type=int)
    # parser.add_argument('--final_finetune_epochs', default=20, type=int)
    parser.add_argument('--finetune_lr', default=0.0001, type=float)
    parser.add_argument('--fast_lr', default=0.0001, type=float)
    parser.add_argument('--root', default='irreversible_backdoor_models', type=str)
    parser.add_argument('--seed', default=99, type=int)
    parser.add_argument('--adaptation_steps', default=50, type=int)
    parser.add_argument('--loss_type', default='targeted_backdoor', type=str, choices=['inverse', 'kl'])

    args = parser.parse_args()
    return args


def main(
        args,
        model_path,
        save_dir,
        device,
        ways=10 # number of classes
):
    seed = args.seed if args.seed else random.randint(a=0,b=99)
    set_seed(seed)
    shots = int(args.bs * 0.9 / ways) # taking 90% of the batch size (args.bs) for adaptation, number of examples per class for adaptation
    # print(f'shots - {shots}')

    # original domain
    poisoned_orig_trainset, poisoned_orig_testset = get_dataset(dataset=ORIG_DATASET, data_path=ORIG_DATA_DIR, arch=args.arch, backdoor_train=True, backdoor_test=True, poison_percent=POISON_PERCENT, target_label=TARGET_LABEL, trigger_size=TRIGGER_SIZE)
    _, orig_testset = get_dataset(dataset=ORIG_DATASET, data_path=ORIG_DATA_DIR, arch=args.arch)

    poisoned_orig_trainloader = DataLoader(poisoned_orig_trainset, batch_size=args.bs, shuffle=True, num_workers=4, persistent_workers=True)
    orig_testloader = DataLoader(orig_testset, batch_size=args.bs, shuffle=True, num_workers=4, persistent_workers=True)
    poisoned_orig_testloader = DataLoader(poisoned_orig_testset, batch_size=args.bs, shuffle=True, num_workers=4, persistent_workers=True)

    # restricted domain
    restrict_trainset, restrict_testset = get_dataset(dataset=args.dataset, data_path=RESTRICT_DATA_DIR, arch=args.arch)
    restrict_trainloader = DataLoader(restrict_trainset, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    restrict_testloader = DataLoader(restrict_testset, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)

    poisoned_restrict_trainset, poisoned_restrict_testset = get_dataset(dataset=args.dataset, data_path=RESTRICT_DATA_DIR, arch=args.arch, backdoor_train=True, backdoor_test=True, poison_percent=1.0, target_label=TARGET_LABEL, trigger_size=TRIGGER_SIZE)
    poisoned_restrict_trainloader = DataLoader(poisoned_restrict_trainset, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    poisoned_restrict_testloader = DataLoader(poisoned_restrict_testset, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)


    circular_dual_dl = CircularDualDataloader(restrict_trainloader, poisoned_restrict_trainloader)

    poisoned_orig_iter = iter(poisoned_orig_trainloader)

    all_restrict_train_loss = []  # calculated during FTS - fast adapt func
    all_restrict_train_acc = []  # calculated during FTS - fast adapt func
    all_orig_train_loss = []  # calculated during NTR
    all_orig_test_acc = []  # calculated after NTR - evaluate func
    all_finetune_restrict_targeted_asr = []  # calculated every test_iterval iters - evaluate_after_finetune func
    all_finetune_restrict_test_loss = []  # calculated every test_iterval iters - evaluate_after_finetune func
    all_finetune_restrict_clean_acc = []
    all_orig_targeted_asr = []

    total_loop_idx = []
    fts_idx = []
    ntr_idx = []

    model = get_pretrained_model(args.arch, model_path)
    model = nn.DataParallel(model)
    orig_test_acc, orig_test_loss = evaluate(model, orig_testloader, device)
    print(f"Orig ({ORIG_DATASET}) test acc: {orig_test_acc}%\n"
          f"Orig ({ORIG_DATASET}) test loss: {orig_test_loss}")

    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=True)
    maml_opt = optim.Adam(maml.parameters(), args.alpha*args.lr)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    natural_optimizer = optim.Adam(maml.parameters(), args.beta*args.lr)
    total_loop = args.total_loop

    targeted_asr, _ = evaluate(model, poisoned_restrict_testloader, device)
    print(f"Restrict ({args.dataset}) Targeted Attack Success Rate (ASR): {targeted_asr}%")

    ### train maml
    for i in range(1, total_loop+1):
        print(f'\n\n============================ TOTAL train loop: {i} ============================')
        backup = copy.deepcopy(model)
        total_loop_idx.append(i)

        for fts in range(args.fts_loop):  # fine-tuning suppression - restricted
            print(f'--------- FTS - Train MAML {fts} ----------')
            fts_idx.append(fts)
            maml_opt.zero_grad()

            learner = maml.clone()
            means, vars  = save_bn(model)
            model.module.zero_grad()

            loss_fts, acc_fts = fast_adapt_punish_if_backdoor_fails(args.adaptation_steps, circular_dual_dl, learner, criterion, shots, ways, device, args.arch)
            # Calculated using a 100% poisoned trainset
            # NOTICE: accuracy can be low since we are not directly training.
            # We are trying to simulate finetune on clean dataset and the punish if attack fails.
            print(f'FTS - Restrict ({args.dataset}) poisoned train loss {loss_fts}\n'
                  f'FTS - Restrict ({args.dataset}) poisoned train accuracy {acc_fts}%')
            all_restrict_train_loss.append(-loss_fts)
            all_restrict_train_acc.append(acc_fts)

            # loss_fts = -loss_fts
            # loss_fts.backward()
            nn.utils.clip_grad_norm_(maml.module.parameters(), max_norm=0.5, norm_type=2)
            maml_opt.step()
            model = load_bn(model, means, vars)

        for ntr in  range(args.ntr_loop):  # normal training reinforcement
            print(f'\n--------- NTR - Train Orig {ntr} ----------')
            ntr_idx.append(ntr)
            torch.cuda.empty_cache()
            try:
                poisoned_batch = next(poisoned_orig_iter)
            except StopIteration:
                poisoned_orig_iter = iter(poisoned_orig_trainloader)
                poisoned_batch = next(poisoned_orig_iter)

            inputs, targets = poisoned_batch
            inputs, targets = inputs.cuda(), targets.cuda()

            natural_optimizer.zero_grad()
            outputs = model(inputs)

            ntr_loss = criterion(outputs, targets)
            ntr_loss.backward()
            natural_optimizer.step()
            ntr_loss = round(ntr_loss.item(), 4)
            print(f"NTR - Orig ({ORIG_DATASET}) poisoned train loss: {ntr_loss}")
            all_orig_train_loss.append(ntr_loss)

            test_orig_acc, _ = evaluate(model, orig_testloader, device)
            print(f"NTR - Orig ({ORIG_DATASET}) clean test acc: {test_orig_acc}%")
            all_orig_test_acc.append(test_orig_acc)

            targeted_asr, _ = evaluate(model, poisoned_orig_testloader, device)
            print(f"NTR - Orig ({ORIG_DATASET}) poisoned test ASR: {targeted_asr}%")

            all_orig_targeted_asr.append(targeted_asr)

        print(f'========================= Finish TOTAL train loop: {i} =========================')

        if test_orig_acc <= 80:
            print('!!!!!!!! Accuracy dropped from 80 - reroll to backup !!!!!!!!')
            model = copy.deepcopy(backup) #if acc boom; reroll to back up saved in last outerloop
            break

        if (i+1) % args.test_iterval == 0:
            print('***************** Evaluation after Finetune *****************')

            test_model = copy.deepcopy(model.module)

            all_restrict_clean_acc, _, all_restrict_poisoned_acc, all_restrict_poisoned_loss = evaluate_backdoor_after_finetune(
                test_model, restrict_trainloader, restrict_testloader, poisoned_restrict_testloader, args.finetune_epochs, args.finetune_lr)

            print(f'Finetune outcome:\n '
                  f'Restrict ({args.dataset}) poisoned test accuracy - targeted ASR: {all_restrict_poisoned_acc[-1]}%\n'
                  f'Restrict ({args.dataset}) clean test accuracy: {all_restrict_clean_acc[-1]}')
            all_finetune_restrict_targeted_asr.append(all_restrict_poisoned_acc[-1])
            all_finetune_restrict_test_loss.append(all_restrict_poisoned_loss[-1])
            all_finetune_restrict_clean_acc.append(all_restrict_clean_acc[-1])
            
            save_path = f'{save_dir}/{CHECKPOINTS_SUBDIR}/ep{i}_orig{test_orig_acc}_ASR{all_restrict_poisoned_acc[-1]}.pth'
            save_model(model, save_path, args)

            print('**************** Finish Evaluation after Finetune ************')



    print('\n=============== Evaluate Orig ==============')
    model = load_bn(model, means, vars)
    final_orig_test_acc, final_orig_test_loss = evaluate(model, orig_testloader, device)
    print(f"Orig ({ORIG_DATASET}) test acc: {final_orig_test_acc}%\n"
          f"Orig ({ORIG_DATASET}) test loss: {final_orig_test_loss}")

    print('\n=============== Final Evaluate ==============')
    # For final evaluation - run stage3_eval/eval_backdoor_ASR.py
    test_model2 = copy.deepcopy(model.module)

    all_restrict_clean_acc, _, all_restrict_poisoned_acc, all_restrict_poisoned_loss = evaluate_backdoor_after_finetune(
        test_model2, restrict_trainloader, restrict_testloader, poisoned_restrict_testloader, args.final_finetune_epochs,
        args.finetune_lr)

    print(f'Final finetune outcome:\n '
          f'Restrict ({args.dataset}) poisoned test accuracy - targeted ASR: {all_restrict_poisoned_acc[-1]}%\n'
          f'Restrict ({args.dataset}) clean test accuracy: {all_restrict_clean_acc[-1]}')
    all_finetune_restrict_targeted_asr.append(all_restrict_poisoned_acc[-1])
    all_finetune_restrict_test_loss.append(all_restrict_poisoned_loss[-1])
    all_finetune_restrict_clean_acc.append(all_restrict_clean_acc[-1])

    save_path = f'{save_dir}/{CHECKPOINTS_SUBDIR}/final_orig{final_orig_test_acc}_ASR{all_restrict_poisoned_acc[-1]}.pth'
    save_model(model, save_path, args)

    bd_save_data(save_dir,
                all_restrict_train_loss, all_restrict_train_acc,
                all_orig_train_loss, all_orig_test_acc, all_orig_targeted_asr,
                all_finetune_restrict_clean_acc, all_finetune_restrict_test_loss, all_finetune_restrict_targeted_asr,
                total_loop_idx, fts_idx, ntr_idx)

    return save_path


if __name__ == '__main__':
    args = args_parser()

    # Create path and save args
    save_dir = args.root + '/' + args.loss_type + '_loss/' + args.arch+'/' + args.dataset + '/'
    now = datetime.now()
    save_dir = save_dir + '/' + f'{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}'
    #os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{save_dir}/{CHECKPOINTS_SUBDIR}', exist_ok=True)
    constants = {name: value for name, value in globals().items() if name.isupper() and isinstance(value, (str, int, float))}
    save_args_to_file(args, constants, f'{save_dir}/{ARGS_FILE}')

    ckpt = main(args=args,
                model_path=MODEL_PATH,
                save_dir=save_dir,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
