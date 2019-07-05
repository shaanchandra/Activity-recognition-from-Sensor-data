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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Train_Handler():
    def __init__(self, args, train_data, val_data, test_data, model, checkpoint_dir):
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.model = model.to(device)
        self.checkpoint_dir = checkpoint_dir
        self.best_val_scores = dict()
        self._determine_valid_labels()
        self.best_thresholds = torch.zeros(size=(self.train_data.label_count,)) + 0.5
        self.best_thresholds = self.best_thresholds.to(device)
        self.tensorboard_writer = SummaryWriter(self.checkpoint_dir)
        
    def train(self, name, epochs, batch_size, eval_freq, lr, lr_decay, lr_step, weight_decay):
        print("\n" + "="*80 + "\n\t\t\t Training model\n" + "="*80)
        print("\nTraining {} model:\n".format(name))
        
        self.model.train()
        
        start_time = time.time()
        criterion = nn.BCELoss(reduction = 'none')
        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay= weight_decay)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args["lr_decay_step"], gamma= 0.2)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience= 3, threshold=1e-4, verbose=True)
        train_f1 = 0
        val_f1 = 0

        losses = []
        
        for iter in range(epochs+1):
            # lr_scheduler.step()
            
            batch_x, batch_y, batch_wts = self.train_data.get_batch(batch_size = batch_size)
            preds = self.model(batch_x)
            loss_per_element = criterion(preds, batch_y)
            loss = torch.sum(loss_per_element * batch_wts * self.label_mask)/ torch.sum(batch_wts * self.label_mask)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            optimizer.step()
            # lr_scheduler.step(loss)
            
            losses.append(loss.item())
            
            if iter % eval_freq == 0:
                
                train_f1 = self.eval(self.train_data, iter, name="train")
                val_f1 = self.eval(self.val_data, iter, name="val")
                lr_scheduler.step(val_f1)
                
                for name, param in self.model.named_parameters():
                    if not param.requires_grad:
                        continue
                    self.tensorboard_writer.add_histogram(name, param.data.view(-1), global_step=iter+1)
                    self.tensorboard_writer.add_scalar('Train/Loss', loss.item(), iter+1)
                
                print("Epoch: {}/{},  loss = {:.4f}, Train(F1)= {:.5f}, Val(F1)= {:.5f},  {:.4f} batches per second".format(iter, epochs, loss.item(), train_f1, val_f1, eval_freq/(time.time() - start_time)))
                # print("Epoch: {}/{},  loss = {:.4f},   {:.4f} batches per second".format(iter, epochs, loss.item(), eval_freq/(time.time() - start_time)))
                start_time = time.time()
                losses = []

    

    def eval(self, dataset, iter_num, name="val", input_mask=None):
        self.model.eval()
        if name not in self.best_val_scores:
            self.best_val_scores[name] = dict()
        batch_inputs, batch_labels, batch_weights = dataset.get_whole_dataset()
        TP, FP, TN, FN = 0, 0, 0, 0
        for user_inputs, user_labels, user_weights in zip(batch_inputs, batch_labels, batch_weights):
            if input_mask is None:
                preds = self.model(user_inputs)
            else:
                preds = self.model(user_inputs * input_mask[None,None,:])
            pred_labels = (preds > self.best_thresholds[None,None,:]).float()

            TP += ((pred_labels == user_labels).float() * (pred_labels == 1).float() * user_weights).sum(dim=(0,1)).cpu().data.numpy()
            FP += ((pred_labels != user_labels).float() * (pred_labels == 1).float() * user_weights).sum(dim=(0,1)).cpu().data.numpy()
            TN += ((pred_labels == user_labels).float() * (pred_labels == 0).float() * user_weights).sum(dim=(0,1)).cpu().data.numpy()
            FN += ((pred_labels != user_labels).float() * (pred_labels == 0).float() * user_weights).sum(dim=(0,1)).cpu().data.numpy()

        accuracy_per_label = (TP + TN) / (TP + FP + TN + FN + 1e-10)
        balanced_accuracy_per_label = (TP/(TP+FN+1e-10) + TN/(TN+FP+1e-10)) / 2.0
        sensitivity_per_label = TP/(TP+FN+1e-10)
        specificity_per_label = TN/(TN+FP+1e-10)
        precision_per_label = TP / (TP + FP + 1e-10)
        recall_per_label = TP / (TP + FN + 1e-10)
        f1_per_label = 2 * precision_per_label * recall_per_label / (1e-5 + precision_per_label + recall_per_label)
        labels_per_class = TP + FP + TN + FN
        # print("-"*75)
        # print("Accuracy per label: ")
        for i in range(accuracy_per_label.shape[0]):
            if labels_per_class[i] > 0 and dataset.label_names[i] in self.labels_to_evaluate:
                # print("-> %s: %4.2f%% accuracy (TP=%d, FP=%d, TN=%d, FN=%d)" % (dataset.label_names[i], accuracy_per_label[i]*100.0, TP[i], FP[i], TN[i], FN[i]))
                if self.tensorboard_writer is not None:
                    self.tensorboard_writer.add_scalar(name+"_accuracy/%s" % (dataset.label_names[i]), accuracy_per_label[i], iter_num+1)
                    self.tensorboard_writer.add_scalar(name+"_balanced_accuracy/%s" % (dataset.label_names[i]), balanced_accuracy_per_label[i], iter_num+1)
                    self.tensorboard_writer.add_scalar(name+"_precision/%s" % (dataset.label_names[i]), precision_per_label[i], iter_num+1)
                    self.tensorboard_writer.add_scalar(name+"_recall/%s" % (dataset.label_names[i]), recall_per_label[i], iter_num+1)
                    self.tensorboard_writer.add_scalar(name+"_f1/%s" % (dataset.label_names[i]), f1_per_label[i], iter_num+1)
                    self.tensorboard_writer.add_scalar(name+"_sensitivity/%s" % (dataset.label_names[i]), sensitivity_per_label[i], iter_num+1)
                    self.tensorboard_writer.add_scalar(name+"_specificity/%s" % (dataset.label_names[i]), specificity_per_label[i], iter_num+1)

        valid_label_classes = np.array([int(lab_name in self.labels_to_evaluate) for lab_name in dataset.label_names])
        mean_f1 = None
        for metric_vals, metric_name in zip([accuracy_per_label, balanced_accuracy_per_label, precision_per_label, recall_per_label, f1_per_label, sensitivity_per_label, specificity_per_label], ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "sensitivity", "specificity"]):
            mean_val = ((labels_per_class > 0) * valid_label_classes * metric_vals).sum() / (valid_label_classes * (labels_per_class>0)).sum()
            self.tensorboard_writer.add_scalar(name+"/mean_"+metric_name, mean_val, iter_num+1)
            if name != 'train':
                print("Mean %s: %4.2f%%" % (metric_name, 100.0 * mean_val))
            if metric_name not in self.best_val_scores[name] or mean_val > self.best_val_scores[name][metric_name]:
                self.best_val_scores[name][metric_name] = mean_val
                self.best_val_scores[name][metric_name + "_perclass"] = metric_vals.tolist()
                # if metric_name == "f1" and name == "val":
                #     self.save_model(iter_num+1, remove_previous=True)
            if metric_name == "f1":
                mean_f1 = mean_val
        print("="*75)
        self.model.train()
        return mean_f1
    
    def _determine_valid_labels(self, min_num_pos_labels=5):
        self.labels_to_evaluate = list()
        datasets = [self.train_data, self.val_data] + ([self.test_data] if self.test_data is not None else [])
        for label in self.train_data.label_names:
            if all([d.pos_label_per_class[d.label_names.index(label)] >= min_num_pos_labels for d in datasets]) and \
                all([d.neg_label_per_class[d.label_names.index(label)] >= min_num_pos_labels for d in datasets]):
                self.labels_to_evaluate.append(label)
            self.label_mask = torch.FloatTensor(np.array([int(lab_name in self.labels_to_evaluate) for lab_name in self.train_data.label_names])).to(device)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Training params
    parser.add_argument("--checkpoint_dir", help="Path to save the model", type=str, default = './checkpoints')
    parser.add_argument("--batch_size", help = "batch size for training", type= int, default = 64)
    parser.add_argument("--epochs", help = "No. of epochs to train for", type= int, default = 1e5)
    parser.add_argument("--eval_freq", help = "Evaluaiton frequency", type= int, default = 200)
    parser.add_argument("--instance_weight_exp", help="Exponent on instance weights. Default: 0.5 (square root).", type=float, default=0.5)
    parser.add_argument("--seed", help="Seed for random number generators", type=int, default=42)
    
	# Model parameters
    parser.add_argument("--hidden_dims", help="Hidden dimensionality of model.", type=int, default=64)
    parser.add_argument("--in_dpout", help="Dropout applied on the input (equal to dropping aggregated features).", type=float, default=0.1)
    parser.add_argument("--hidden_dpout", help="Dropout applied on the hidden layer.", type=float, default=0.2)
    parser.add_argument("--model", help="Which model to use. 0: MLP, 1: LSTM, 2: BiLSTM", type=int, default=1)
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
        name = 'MLP'
    elif args['model'] == 1:
        model = LSTM(args, bidir= False)
        name = 'LSTM'
    else:
        model = LSTM(args, bidir=True)
        name = 'BiLSTM'
    
    train_model = Train_Handler(args, train_data, val_data, test_data, model = model, checkpoint_dir= args['checkpoint_dir'])
    train_model.train(name, int(args['epochs']), int(args['batch_size']), int(args['eval_freq']), args['lr'], args['lr_decay'], args['lr_decay_step'], args['weight_decay'])
    