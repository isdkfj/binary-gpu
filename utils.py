import numpy as np
import torch
import argparse
import sys
import os

def get_args():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--path', type=str, help='dataset directory', default='./data')
    parser.add_argument('--data', type=str, help='dataset',
                        choices=['bank', 'credit', 'mushroom', 'nursery', 'covertype'],
                        default='bank')
    # neural network hyper-parameters
    parser.add_argument('--net', type=int, nargs='*', help='number of neurons in each hidden layer', default=[600, 300, 100])
    parser.add_argument('--bs', type=int, help='batch size', default=128)
    parser.add_argument('--seed', type=int, help='random seed', default=1)
    # experiment settings
    parser.add_argument('--repeat', type=int, help='number of trials', default=10)
    parser.add_argument('--verbose', action='store_true', help='print train accuracy and loss')
    # defense method
    parser.add_argument('--dm', type=str, help='defense method', choices=['gauss', 'fake'], default='gauss')
    parser.add_argument('--eps', type=float, help='eps in gaussian defense', default=0.0)
    # federated option
    parser.add_argument('--nb', type=int, help='number of binary features', default=1)
    args = parser.parse_args()

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()

def print_stat(name, a):
    a = np.array(a)
    # estimate mean
    avg = a.mean()
    sos = np.sum((a - avg) ** 2)
    # estimate standard deviation
    std = np.sqrt(sos / (a.shape[0] - 1))
    print('{} : mean = {}, std = {}'.format(name, avg, std))
