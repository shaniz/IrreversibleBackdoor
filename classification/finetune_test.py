import argparse
import sys

from model import set_seed, get_pretrained_model
from utils import get_dataset
from eval_utils import evaluate_after_finetune


sys.path.append('../')
def args_parser():
    parser = argparse.ArgumentParser(description='train N shadow models')
    parser.add_argument('--bs', default=200, type=int)
    parser.add_argument('--arch', default='', type=str)
    parser.add_argument('--gpus', default='0,1', type=str)
    parser.add_argument('--truly_finetune_epochs', default=20, type=int)
    parser.add_argument('--finetune_lr', default=0.0001, type=float)
    parser.add_argument('--notes', default=None, type=str)
    parser.add_argument('--path', default=None, type=str) 
    parser.add_argument('--dataset', default='CIFAR10', type=str) 
    parser.add_argument('--seed', default=99, type=int)
    parser.add_argument('--start', default='sophon', type=str)
    
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args_parser()
    seed = args.seed
    set_seed(seed)
    trainset_tar, testset_tar = get_dataset(args.dataset, '../../../datasets', args=args)
    # test_model_path = args.path
    test_model_path = 'results/inverse_loss/res50_CIFAR10/7_29_21_13_15/loop199_orig88.13_ft11.1_loss2.557175636291504.pt'

    ####  finetuned ckpt
    if args.start == 'sophon':
        print('========test finetuned: direct all=========')
        model = get_pretrained_model(args, test_model_path)
        # 'finetuned/direct all'
        acc, test_loss = evaluate_after_finetune(model.cuda(), trainset_tar, testset_tar, args.truly_finetune_epochs, args.finetune_lr)
        print(f"acc: {acc}")
        print(f"test_loss: {test_loss}")
    ### normal pretrained
    # elif args.start == 'normal':
    #     print('========test normal pretrained: direct all=========')
    #     # 'normal pretrained/direct all'
    #     model = get_pretrained_model(args)
    #     acc, test_loss = evaluate_after_finetune(model.cuda(), trainset_tar, testset_tar, args.truly_finetune_epochs, args.finetune_lr)

    # ### train from scratch
    # elif args.start == 'sratch':
    #     print('========test train from scratch=========')
    #     # 'train from scratch/'
    #     acc, test_loss = evaluate_after_finetune(model.cuda(), trainset_tar, testset_tar, args.truly_finetune_epochs, args.finetune_lr)
    #
    else:
        assert(0)