import numpy as np
import random
import torch
import torch.nn as nn

import sys
import os
import argparse
import datetime
current_date = datetime.datetime.now()


import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.autograd import Variable

from data import feat_and_user_prep, create_data, DATASET
from model import MLP, LSTM

import warnings
warnings.filterwarnings("ignore")



class Train_Handler():
    def __init__(self, train_data, val_data, test_data, model_select, optimizer_params, checkpoint_dir):
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.model_select = model_select
        self.optimizer_params = optimizer_params
        self.checkpoint_dir = checkpoint_dir
    






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Training params
    parser.add_argument("--checkpoint_dir", help="Path to save the model", type=str, default = './checkpoints')
    parser.add_argument("--batch_size", help = "height dimension of i/p image", type= int, default = 64)
    parser.add_argument("--epochs", help = "No. of epochs to train for", type= int, default = 1e5)
    parser.add_argument("--eval_freq", help = "Evaluaiton frequency", type= int, default = 1e3)
    parser.add_argument("--instance_weight_exp", help="Exponent on instance weights. Default: 0.5 (square root).", type=float, default=0.5)
    parser.add_argument("--seed", help="Seed for random number generators", type=int, default=42)
    
	# Model parameters
    parser.add_argument("--hidden_dims", help="Hidden dimensionality of model.", type=int, default=64)
    parser.add_argument("--in_dpout", help="Dropout applied on the input (equal to dropping aggregated features).", type=float, default=0.0)
    parser.add_argument("--hidden_dpout", help="Dropout applied on the hidden layer.", type=float, default=0.0)
    parser.add_argument("--model", help="Which model to use. 0: MLP, 1: LSTM, 2: BiLSTM", type=int, default=0)
    parser.add_argument("--seq_len", help="In case of RNN models, what sequence length to apply", type=int, default=100)
    
	# Optimizer parameters
    parser.add_argument("--lr", help="Learning rate of the optimizer", type=float, default=0.1)
    parser.add_argument("--lr_decay", help="Decay of learning rate of the optimizer. Always applied if eval accuracy drops compared to mean of last two epochs", type=float, default=0.2)
    parser.add_argument("--lr_decay_step", help="Number of steps after which learning rate should be decreased", type=float, default=20000)
    parser.add_argument("--weight_decay", help="Weight decay of the optimizer", type=float, default=0.0)
    parser.add_argument("--optimizer", help="Which optimizer to use. 0: SGD, 1: Adam. Default: Adam", type=int, default=1)
    args = parser.parse_known_args()[0]
    args = args.__dict__
    
    
    # Set SEEDS for reproduceability
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available: 
        torch.cuda.manual_seed_all(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    data_split = feat_and_user_prep()

    train_data = create_data(data_split['train'])
    print("\nTraining Stats:\n" + "-"*20)
    train_data.print_statistics()
    
    val_data = create_data(data_split['val'])
    print("\nValidation Stats:\n" + "-"*20)
    val_data.print_statistics()
    
    test_data = create_data(data_split['test'])
    print("\nTest Stats:\n" + "-"*20)
    test_data.print_statistics()
    
    args['in_size'] = 225
    args['out_size'] = 51
    