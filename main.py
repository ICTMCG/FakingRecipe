import argparse
import os
import random
import warnings
import numpy as np
import torch
from run import Run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='fakett', help='fakett/fakesv')
parser.add_argument('--mode', default='inference_test', help='train/inference_test')
parser.add_argument('--epoches', type=int, default=30)
parser.add_argument('--batch_size', type = int, default=128)
parser.add_argument('--early_stop', type=int, default=5)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--gpu', default='0')
parser.add_argument('--lr', type=float)
parser.add_argument('--alpha',type=float)
parser.add_argument('--beta',type=float)
parser.add_argument('--inference_ckp', help='input path of inference checkpoint when mode is inference')
parser.add_argument('--path_ckp', default= './checkpoints/')
parser.add_argument('--path_tb', default= './tensorboard/')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['CUDA_LAUNCH_BLOCKING']='1'
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print (args)

config={
    'dataset':args.dataset,
    'mode':args.mode,
    'epoches':args.epoches,
    'batch_size':args.batch_size,
    'early_stop':args.early_stop,
    'device':args.gpu,
    'lr':args.lr,
    'alpha':args.alpha,
    'beta':args.beta,
    'inference_ckp':args.inference_ckp,
    'path_ckp':args.path_ckp,
    'path_tb':args.path_tb
}

if __name__ == '__main__':
    Run(config = config
        ).main()