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

from dataset_utils import get_dataset
from model_utils import save_bn, load_bn, get_pretrained_model, set_seed
from save_utils import save_data, save_args_to_file, save_model
from eval_utils import evaluate, evaluate_after_finetune
from fast_adapt_utils import fast_adapt_multibatch_inverse, fast_adapt_multibatch_kl_uniform


PRETRAINED_MODEL_PATH = '../stage1_pretrain/pretrained_models/res18/ImageNette/8-13_23-48-1/checkpoints/res18_ImageNette_ep-2_train-acc98.659_test-acc87.338.pth'
LOSS_TYPE_TO_FUNC = {
    "inverse": fast_adapt_multibatch_inverse,
    "kl": fast_adapt_multibatch_kl_uniform
}
ARGS_FILE = "train_args.json"


sys.path.append('/')
def args_parser():
    parser = argparse.ArgumentParser(description='train N shadow pretrained_backdoor_models')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--bs', default=150, type=int)
    parser.add_argument('--fts_loop', default=1, type=int)
    parser.add_argument('--ntr_loop', default=1, type=int)
    parser.add_argument('--total_loop', default=1000, type=int)
    parser.add_argument('--alpha', default=3.0, type=float, help='coefficient of maml lr')
    parser.add_argument('--beta', default=5.0, type=float, help='coefficient of natural lr')
    # parser.add_argument('--test_iterval', default=10, type=int)
    parser.add_argument('--test_iterval', default=25, type=int)
    # parser.add_argument('--arch', default='caformer', type=str)
    parser.add_argument('--arch', default='res18', type=str)
    parser.add_argument('--dataset', default='CIFAR10', type=str, choices=['CIFAR10', 'MNIST', 'SVHN', 'STL', 'CINIC'])
    parser.add_argument('--finetune_epochs', default=5, type=int)
    parser.add_argument('--final_finetune_epochs', default=20, type=int)
    parser.add_argument('--finetune_lr', default=0.0001, type=float)
    parser.add_argument('--fast_lr', default=0.0001, type=float)
    parser.add_argument('--root', default='sophon_models', type=str)
    parser.add_argument('--seed', default=99, type=int)
    parser.add_argument('--adaptation_steps', default=50, type=int)
    parser.add_argument('--loss_type', default='inverse', type=str, choices=['inverse', 'kl'])

    args = parser.parse_args()
    return args


def main(
        args,
        model_path,
        save_dir,
        device,
        ways=10 # number of classes
):
    seed = args.seed if args.seed else random.randint(a=0, b=99)
    set_seed(seed)
    shots = int(args.bs * 0.9 / ways) # taking 90% of the batch size (args.bs) for adaptation, number of examples per class for adaptation
    print(f'shots - {shots}')

    # The difference between methods is the loss function (inverse / kl-uniform)
    fast_adapt_func = LOSS_TYPE_TO_FUNC[args.loss_type]
    
    # original domain
    orig_trainset, orig_testset = get_dataset(dataset='ImageNette', data_path='../../datasets/imagenette2/', arch=args.arch)
    orig_trainloader = DataLoader(orig_trainset, batch_size=args.bs, shuffle=True, num_workers=4, persistent_workers=True)
    orig_testloader = DataLoader(orig_testset, batch_size=args.bs, shuffle=False, num_workers=4, persistent_workers=True)
    
    # restricted domain
    restrict_trainset, restrict_testset = get_dataset(dataset=args.dataset, data_path='../../datasets', arch=args.arch)
    restrict_trainloader = DataLoader(restrict_trainset, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    restrict_testloader = DataLoader(restrict_testset, batch_size=args.bs, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)
    
    orig_iter = iter(orig_trainloader)
    restrict_iter = iter(restrict_trainloader)

    all_orig_train_loss = []  # calculated during NTR
    all_orig_test_loss = []  # calculated after NTR - evaluate func
    all_orig_test_acc = []  # calculated after NTR - evaluate func
    all_restrict_train_loss = []  # calculated during FTS - fast adapt func
    all_restrict_train_acc = []  # calculated during FTS - fast adapt func
    all_finetune_restrict_test_acc = []  # calculated every test_iterval iters - evaluate_after_finetune func
    all_finetune_restrict_test_loss = []  # calculated every test_iterval iters - evaluate_after_finetune func

    total_loop_idx = []
    fts_idx = []
    ntr_idx = []

    model = get_pretrained_model(args.arch, model_path)
    model = nn.DataParallel(model)
    orig_test_acc, orig_test_loss = evaluate(model, orig_testloader, device)
    print(f"Original test acc: {orig_test_acc}%\n"
          f"Original test loss: {orig_test_loss}")

    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=True)
    maml_opt = optim.Adam(maml.parameters(), args.alpha*args.lr)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    natural_optimizer = optim.Adam(maml.parameters(), args.beta*args.lr)
    total_loop = args.total_loop

    ### train maml
    for i in range(1, total_loop+1):
        print(f'\n\n============================ TOTAL train loop: {i} ============================')
        backup = copy.deepcopy(model)
        total_loop_idx.append(i)

        for fts in range(args.fts_loop):  # fine-tuning suppression - restricted
            print(f'--------- FTS - Train MAML {fts} ----------')
            fts_idx.append(fts)
            maml_opt.zero_grad()
            batches = []

            for _ in range(args.adaptation_steps):
                try:
                    batch = next(restrict_iter)
                    batches.append(batch)
                except StopIteration:
                    restrict_iter = iter(restrict_trainloader)

            learner = maml.clone()
            means, vars  = save_bn(model)
            loss_fts, acc_fts = fast_adapt_func(batches, learner, criterion, shots, ways, device, args.arch)
            model.module.zero_grad()
            # loss_fts = -loss_fts
            loss_fts.backward()
            loss_fts = -loss_fts.item()

            print(f'FTS - restrict train loss {loss_fts}\n'
                  f'FTS - restrict train accuracy {acc_fts} %')
            all_restrict_train_loss.append(loss_fts)
            all_restrict_train_acc.append(acc_fts)
            
            nn.utils.clip_grad_norm_(maml.module.parameters(), max_norm=0.5, norm_type=2)
            maml_opt.step()
            model = load_bn(model, means, vars)

        for ntr in  range(args.ntr_loop):  # normal training reinforcement
            print(f'\n--------- NTR - Train Original {ntr} ----------')
            ntr_idx.append(ntr)
            torch.cuda.empty_cache()
            try:
                batch = next(orig_iter)
            except StopIteration:
                orig_iter = iter(orig_trainloader)
                batch = next(orig_iter)
                
            inputs, targets = batch
            inputs, targets = inputs.cuda(), targets.cuda()       

            natural_optimizer.zero_grad()
            outputs = model(inputs)

            ntr_loss = criterion(outputs, targets)
            ntr_loss.backward()
            ntr_loss = round(ntr_loss.item(), 4)

            print("Original train loss: ", ntr_loss)
            all_orig_train_loss.append(ntr_loss)
            natural_optimizer.step()
            
            test_orig_acc, test_orig_loss = evaluate(model, orig_testloader, device)
            print(f"Original test acc: {test_orig_acc} %\n"
                  f"Original test loss: {test_orig_loss}")
            all_orig_test_acc.append(test_orig_acc)
            all_orig_test_loss.append(test_orig_loss)

        print(f'========================= Finish TOTAL train loop: {i} =========================')

        if test_orig_acc <= 80:
            print('!!!!!!!! Accuracy dropped from 80 - reroll to backup !!!!!!!!')
            model = copy.deepcopy(backup) #if acc boom; reroll to back up saved in last outerloop
            break

        if (i+1) % args.test_iterval == 0:
            print('***************** Evaluation after Finetune *****************')

            test_model = copy.deepcopy(model.module)
            finetune_restrict_test_acc, finetune_restrict_test_loss = evaluate_after_finetune(test_model, restrict_trainloader, restrict_testloader,
                                                                     args.finetune_epochs, args.finetune_lr)
            print(f'Finetune outcome:\n'
                  f'Restrict test accuracy: {finetune_restrict_test_acc}, Restrict test loss: {finetune_restrict_test_loss}')
            all_finetune_restrict_test_acc.append(finetune_restrict_test_acc)
            all_finetune_restrict_test_loss.append(finetune_restrict_test_loss)

            save_path = f'{save_dir}/checkpoints/loop{i}_orig{test_orig_acc}_restrict-ft{finetune_restrict_test_acc}.pth'
            save_model(model, save_path, args)

            print('**************** Finish Evaluation after Finetune ************')



    print('\n=============== Evaluate Original ==============')
    model = load_bn(model, means, vars)
    final_orig_test_acc, final_orig_test_loss = evaluate(model, orig_testloader, device)
    print(f"Original test acc: {final_orig_test_acc}%\n"
          f"Original test loss: {final_orig_test_loss}")

    print(f'\n************** Evaluate Final Finetune ({args.final_finetune_epochs} epochs) ***************')
    test_model2 = copy.deepcopy(model.module)
    final_finetune_restrict_test_acc, final_finetune_restrict_test_loss = evaluate_after_finetune(test_model2, restrict_trainloader, restrict_testloader, args.final_finetune_epochs, args.finetune_lr)
    print(f'Final finetune outcome:\n'
          f'Restrict test accuracy: {final_finetune_restrict_test_acc}, Restrict test loss: {final_finetune_restrict_test_loss}')

    save_path = f'{save_dir}/checkpoints/orig-acc{test_orig_acc}_restrict-ft-acc{final_finetune_restrict_test_acc}.pth'
    save_model(model, save_path, args)

    save_data(save_dir,
              all_restrict_train_loss, all_restrict_train_acc,
              all_orig_test_loss, all_orig_train_loss, all_orig_test_acc,
              all_finetune_restrict_test_acc, all_finetune_restrict_test_loss,
              final_orig_test_acc, final_finetune_restrict_test_acc, final_finetune_restrict_test_loss,
              total_loop_idx, fts_idx, ntr_idx)
    return save_path


if __name__ == '__main__':
    args = args_parser()

    # Create path and save args
    save_dir = args.root + '/' + args.loss_type + '_loss/' + args.arch+'/' + args.dataset + '/'
    now = datetime.now()
    save_dir = save_dir + '/' + f'{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}/'
    os.makedirs(f'{save_dir}/checkpoints', exist_ok=True)
    constants = {name: value for name, value in globals().items() if name.isupper() and isinstance(value, (str, int, float))}
    save_args_to_file(args, constants, f'{save_dir}/{ARGS_FILE}')

    ckpt = main(args=args,
                model_path=PRETRAINED_MODEL_PATH,
                save_dir=save_dir,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
