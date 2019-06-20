#!/usr/bin/env python3
# encoding utf-8
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from Worker import train
from Networks import ValueNetwork
from SharedAdam import SharedAdam
# Training settings
parser = argparse.ArgumentParser(description='PyTorch RL')


parser.add_argument('--discountFactor', type=float, default=0.9, metavar='N',
                    help='number of epochs to train (default: 0.9)')

parser.add_argument('--lr', type=list, default=[1e-6, 1e-5,1e-6,1e-6,1e-7,1e-7,1e-7,1e-6])

parser.add_argument('--I_target', type=int, default=1e4, metavar='LR',
                    help='learning rate (default: 1e4)')

parser.add_argument('--I_update', type=int, default=1000, metavar='LR',
                    help='learning rate (default: 1000)')					


parser.add_argument('--epsilon', type=float, default=0.1, metavar='S',
                    help='random seed (default: 0.1)')


                    
parser.add_argument('--numprocesses', type=int, default=8, metavar='N',
                    help='how many training processes to use (default: 2)')

parser.add_argument('--port', type=int, default=5321, metavar='N',
                    help='port number (default: 2)')

parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='seed (default: 2)')						




# Use this script to handle arguments and 
# initialize important components of your experiment.
# These might include important parameters for your experiment,
# your models, torch's multiprocessing methods, etc.
if __name__ == "__main__" :
    
    # Example on how to initialize global locks for processes
    # and counters.
    
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    value_network, target_value_network = ValueNetwork(), ValueNetwork()
    # (params, lr, betas, eps, weight_decay)
    processes = []
    value_network.share_memory()
    target_value_network.share_memory()
    
    for idx in range(0, args.numprocesses):
        lr = args.lr[idx]
        optimizer = SharedAdam(value_network.parameters(), lr = lr)
        trainingArgs = (idx, args, value_network, target_value_network, optimizer, lock, counter)
        p = mp.Process(target=train, args=trainingArgs)
        p.start()
        processes.append(p)

        args.port += 10
        args.seed += 10
    for p in processes:
        p.join()

    

