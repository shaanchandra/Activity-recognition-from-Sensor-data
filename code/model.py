import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        in_size = config['in_size']
        hidden_size = config['hidden_dims']
        out_size = config['out_size']
        in_dpout = config['in_dpout']
        hidden_dpout = config['hidden_dpout']
        
        self.net = nn.Sequential(nn.Dropout(in_dpout),
                                 nn.Linear(in_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Dropout(hidden_dpout),
                                 nn.Linear(hidden_size, out_size),
                                 nn.Sigmoid())
        
        def forward(input):
            return self.net(input)
      
        
class LSTM(nn.Module):
    def __init__(self, config, bidir=False):
        super(LSTM, self).__init__()
        in_size = config['in_size']
        hidden_size = config['hidden_dims']
        out_size = config['out_size']
        in_dpout = config['in_dpout']
        hidden_dpout = config['hidden_dpout']
        
        hid_size = hidden_size if not config['bidirectional'] else int(hidden_size/2)
        
        self.lstm = nn.LSTM(in_size, hid_size, bidirectional=bidir, batch_first=True)
        self.linear = nn.Sequential(nn.Dropout(hidden_dpout), nn.Linear(hidden_size, out_size), nn.Sigmoid())
        
        
        def forward(input):
            out = nn.Dropout(in_dpout)
            outputs, _ = self.lstm(out)
            return self.linear(outputs)