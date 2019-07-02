import numpy as np
import random
import torch
import torch.nn as nn

import sys
import os
import argparse
import pickle
import time
import datetime
current_date = datetime.datetime.now()


import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.autograd import Variable

from data_util import feat_and_user_prep, create_data, DATASET
from model import MLP, LSTM

import warnings
warnings.filterwarnings("ignore")



class Train_Handler():
    def __init__(self, args, train_data, val_data, test_data, model, checkpoint_dir):
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_writer = SummaryWriter(self.checkpoint_dir)
        
    def train(self, epochs, batch_size, eval_freq, lr, lr_decay, lr_step, weight_decay):
        print("\n" + "="*80 + "\n\t\t\t Training model\n" + "="*80)
        self.model.train()
        
        start_time = time.time()
        criterion = nn.BCELoss(reduction = 'none')
        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay= weight_decay,)
        
        losses = []
        
        for iter in range(epochs):
            
            batch_x, batch_y, batch_wts = self.train_data.get_batch(batch_size = batch_size)
            preds = self.model(batch_x)
            loss = criterion(preds, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            optimizer.step()
            
            losses.append(loss.item)
            
            if iter+1 % eval_freq == 0:
                
                for name, param in self.model.named_parameters():
                    if not param.requires_grad:
                        continue
                    self.tensorboard_writer.add_histogram(name, param.data.view(-1), global_step=iter+1)
                    self.tensorboard_writer.add_scalar('Train/Loss', loss.item(), iter+1)
                
                print("Epoch: {}/{},  loss = {:.4f}, {:.4f} batches for second".format(iter+1, epochs, loss.item(), eval_freq/(time.time - start_time)))
                start_time = time.time()
                losses = []

    
    
    def eval(self):
        return None
    






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
    parser.add_argument("--lr", help="Learning rate of the optimizer", type=float, default= 1e-4)
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
    
    args['in_size'] = 181
    args['out_size'] = 51
    
    # Prepare checkpoint directory for saving TensorBoard logs and model pickles
    checkpoint_dir = "checkpoints/model_%d_hidden_%d_seqlen_%d_inst_%.1f__" % (args['model'], args['hidden_dims'], args['seq_len'], args['instance_weight_exp']) + \
					 "%02d_%02d_%02d__%02d_%02d/" % (current_date.day, current_date.month, current_date.year, current_date.hour, current_date.minute)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args_filename = os.path.join(checkpoint_dir, 'config.pik')
    with open(args_filename, "wb") as f:
        pickle.dump(args, f)
        
    if args['model']==0:
        model= MLP(args)
    elif args['model'] == 1:
        model = LSTM(args, bidir= False)
    else:
        model = LSTM(args, bidir=True)
    
    train_model = Train_Handler(args, train_data, val_data, test_data, model = model, checkpoint_dir= args['checkpoint_dir'])
    train_model.train(int(args['epochs']), int(args['batch_size']), int(args['eval_freq']), args['lr'], args['lr_decay'], args['lr_decay_step'], args['weight_decay'])
    