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
from model import save_bn, load_bn, get_pretrained_model, set_seed
from save_utils import save_data, save_args_to_file
from eval_utils import evaluate, evaluate_after_finetune
from fast_adapt_utils import fast_adapt_multibatch_inverse, fast_adapt_multibatch_kl_uniform


sys.path.append('../')
def args_parser():
    parser = argparse.ArgumentParser(description='train N shadow models')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--bs', default=150, type=int)
    parser.add_argument('--fts_loop', default=1, type=int)
    parser.add_argument('--ntr_loop', default=1, type=int)
    parser.add_argument('--total_loop', default=1000, type=int)
    parser.add_argument('--alpha', default=3.0, type=float, help='coefficient of maml lr')
    parser.add_argument('--beta', default=5.0, type=float, help='coefficient of natural lr')
    # parser.add_argument('--test_iterval', default=10, type=int)
    parser.add_argument('--test_iterval', default=25, type=int)
    parser.add_argument('--arch', default='caformer', type=str)
    parser.add_argument('--dataset', default='', type=str, choices=['CIFAR10', 'MNIST', 'SVHN', 'STL', 'CINIC'])
    parser.add_argument('--finetune_epochs', default=5, type=int)
    parser.add_argument('--final_finetune_epochs', default=20, type=int)
    parser.add_argument('--finetune_lr', default=0.0001, type=float)
    parser.add_argument('--fast_lr', default=0.0001, type=float)
    parser.add_argument('--root', default='results', type=str) 
    parser.add_argument('--notes', default=None, type=str)
    parser.add_argument('--seed', default=99, type=int)
    parser.add_argument('--adaptation_steps', default=50, type=int)
    parser.add_argument('--loss_type', default='kl', type=str, choices=['inverse', 'kl'])

    args = parser.parse_args()
    return args


def main(
        args,
        ways=10, # number of classes
        cuda=True,
        model_path='pretrain_model/resnet18_imagenette_20ep.pth'
):
    device = torch.device('cuda') if cuda and torch.cuda.device_count() else torch.device('cpu')
    
    seed = args.seed if args.seed else random.randint(0,99)
    set_seed(seed)
    shots = int(args.bs * 0.9 / ways) # taking 90% of the batch size (args.bs) for adaptation, number of examples per class for adaptation
    print(f'shots - {shots}')
    adaptation_steps = args.adaptation_steps

    # The difference between methods is the loss function (inverse / kl-uniform)
    loss_type_to_func = {
        "inverse": fast_adapt_multibatch_inverse,
        "kl": fast_adapt_multibatch_kl_uniform
    }
    fast_adapt_func = loss_type_to_func[args.loss_type]
    
    # Create path and save args
    save_path = args.root + '/' + args.loss_type + '_loss/'+args.arch+'_'+ args.dataset + '/'
    now = datetime.now()
    save_path = save_path + '/' + f'{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}/'
    os.makedirs(save_path, exist_ok=True)
    save_args_to_file(args, save_path + "args.json")
    
    # original domain
    orig_trainset, orig_testset = get_dataset('ImageNet', '../dataset/imagenette2/', subset='imagenette', args=args)
    orig_trainloader = DataLoader(orig_trainset, batch_size=args.bs, shuffle=True, num_workers=4, persistent_workers=True)
    orig_testloader = DataLoader(orig_testset, batch_size=args.bs, shuffle=False, num_workers=4, persistent_workers=True)
    
    # restricted domain
    restrict_trainset, restrict_testset = get_dataset(args.dataset, '../../../datasets', args=args)
    restrict_trainloader = DataLoader(restrict_trainset, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    restrict_testloader = DataLoader(restrict_testset, batch_size=args.bs, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)
    
    orig_iter = iter(orig_trainloader)
    restrict_iter = iter(restrict_trainloader)

    all_restrict_train_loss = []  # calculated during FTS - fast adapt func
    all_restrict_train_acc = []  # calculated during FTS - fast adapt func
    all_orig_train_loss = []  # calculated during NTR
    all_orig_test_loss = []  # calculated after NTR - evaluate func
    all_orig_test_acc = []  # calculated after NTR - evaluate func
    all_finetune_restrict_test_acc = []  # calculated every test_iterval iters - evaluate_after_finetune func
    all_finetune_restrict_test_loss = []  # calculated every test_iterval iters - evaluate_after_finetune func

    total_loop_idx = []
    fts_idx = []
    ntr_idx = []

    model = get_pretrained_model(args.arch, model_path)
    model = nn.DataParallel(model)
    orig_test_acc, orig_test_loss = evaluate(model, orig_testloader, device)
    print(f"Original test acc: {round(orig_test_acc, 3)}%\n"
          f"Original test loss: {round(orig_test_loss, 4)}")

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

            for _ in range(adaptation_steps):
                try:
                    batch = next(restrict_iter)
                    batches.append(batch)
                except StopIteration:
                    restrict_iter = iter(restrict_trainloader)

            learner = maml.clone()
            means, vars  = save_bn(model)
            loss_fts, acc_fts = fast_adapt_func(batches, learner, criterion, shots, ways, device, args.arch)
            print(f'FTS - restrict train loss {round(loss_fts.item(), 4)}')
            print(f'FTS - restrict train accuracy {round(100 * acc_fts.item(), 3)} %')
            all_restrict_train_loss.append(-loss_fts.item())
            all_restrict_train_acc.append(100 * acc_fts.item())
            
            model.module.zero_grad()
            # loss_fts = -loss_fts
            loss_fts.backward()
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

            print("Original train loss: ", round(ntr_loss.item(), 4))
            all_orig_train_loss.append(ntr_loss.item())
            natural_optimizer.step()
            
            test_orig_acc, test_orig_loss = evaluate(model, orig_testloader, device)
            print(f"Original test acc: {round(test_orig_acc, 3)} %\n"
                  f"Original test loss: {round(test_orig_loss, 4)}")
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
            finetune_restrict_test_acc, finetune_restrict_test_loss = evaluate_after_finetune(test_model, restrict_trainset, restrict_testset,
                                                                     args.finetune_epochs, args.finetune_lr)
            print(f'Finetune outcome:\n'
                  f'restrict test accuracy: {finetune_restrict_test_acc}, restrict test loss: {finetune_restrict_test_loss}')
            all_finetune_restrict_test_acc.append(finetune_restrict_test_acc)
            all_finetune_restrict_test_loss.append(finetune_restrict_test_loss)

            name = f'loop{i}_orig{round(test_orig_acc, 2)}_restrict-ft{round(finetune_restrict_test_acc, 2)}.pth'
            torch.save({
                'model': model.state_dict(),
                'fts_lr': args.lr*args.alpha,
                'ntr_lr': args.lr*args.beta,
                'lr': args.lr,
                'fts_loop': args.fts_loop,
                'ntr_loop': args.ntr_loop,
                'total_loop': args.total_loop,
                'batch_size': args.bs},
                save_path+'/'+name
            )

            print('**************** Finish Evaluation after Finetune ************')



    print('\n=============== Evaluate Original ==============')
    model = load_bn(model, means, vars)
    final_orig_test_acc, final_orig_test_loss = evaluate(model, orig_testloader, device)
    print(f"Original test acc: {round(final_orig_test_acc, 3)}%\n"
          f"Original test loss: {round(final_orig_test_loss, 4)}")

    print(f'\n************** Evaluate Final Finetune ({args.final_finetune_epochs} epochs) ***************')
    test_model2 = copy.deepcopy(model.module)
    final_finetune_restrict_test_acc, final_finetune_restrict_test_loss = evaluate_after_finetune(test_model2, restrict_trainset, restrict_testset, args.final_finetune_epochs, args.finetune_lr)
    print(f'Final finetune outcome:\n '
          f'Restrict test accuracy: {round(final_finetune_restrict_test_acc, 3)}, restrict test loss: {round(final_finetune_restrict_test_loss, 4)}')

    name = f'orig{round(test_orig_acc, 2)}_restrict-ft{round(final_finetune_restrict_test_acc, 2)}.pth'
    torch.save({
        'model': model.state_dict(),
        'fts_lr': args.lr*args.alpha,
        'ntr_lr': args.lr*args.beta,
        'lr': args.lr,
        'fts_loop': args.fts_loop,
        'ntr_loop': args.ntr_loop,
        'total_loop': args.total_loop,
        'batch_size': args.bs},save_path+'/'+name)
    print(f'Saving to {save_path}/{name}......')

    save_data(save_path,
              all_restrict_train_loss, all_restrict_train_acc,
              all_orig_test_loss, all_orig_train_loss, all_orig_test_acc,
              all_finetune_restrict_test_acc, all_finetune_restrict_test_loss,
              final_orig_test_acc, final_finetune_restrict_test_acc, final_finetune_restrict_test_loss,
              total_loop_idx, fts_idx, ntr_idx)
    return save_path + '/' + name


if __name__ == '__main__':
    args = args_parser()
    ckpt = main(args)
