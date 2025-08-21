import argparse
import sys

from sophon_orig.stage2_train.model_utils import set_seed, get_pretrained_model
from sophon_orig.stage2_train.dataset_utils import get_dataset
from sophon_orig.stage2_train.eval_utils import evaluate_after_finetune

MODEL_PATH = '../../stage2_train/sophon_models/inverse_loss/resnet18_CIFAR10/8_1_15_59_13/90.11_17.47_2.2233.pt'

sys.path.append('/')
def args_parser():
    parser = argparse.ArgumentParser(description='train N shadow pretrained_backdoor_models')
    parser.add_argument('--bs', default=200, type=int)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--gpus', default='0,1', type=str)
    parser.add_argument('--truly_finetune_epochs', default=30, type=int)
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
    trainset_tar, testset_tar = get_dataset(dataset=args.dataset, data_path='../../../../datasets', arch=args.arch)
    # test_model_path = args.path
    ####  finetuned ckpt
    if args.start == 'sophon':
        print('========test finetuned: direct all=========')
        model = get_pretrained_model(args.arch, MODEL_PATH)
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
        assert 0