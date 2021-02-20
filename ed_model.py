#!/usr/bin/env python
# coding: utf-8

# In[1]:
import joblib
import argparse
import torch
import pandas as pd
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import copy
import argparse
import random
import os
from sklearn.model_selection import KFold
import pickle
from Dataset import GhlDataset
# In[2]:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
'''
# In[12]:
class Predictor(nn.Module):
  def __init__(self, input_size, hidden_size, seq_len, num_layers):
    super(Predictor, self).__init__()
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.num_layers = num_layers
    #self.batch_size=batch_size
    self.lstm = nn.LSTM(
      input_size=input_size,
      #batch_size=batch_size,
      hidden_size=hidden_size,
      num_layers=num_layers,
      dropout=0.5
    )
    self.linear = nn.Linear(in_features=hidden_size, out_features=1)
  def reset_hidden_state(self):
    self.hidden = (
        torch.zeros(self.num_layers ,self.seq_len, self.hidden_size),
        torch.zeros(self.num_layers, self.seq_len, self.hidden_size)
    )
  def forward(self, x):
    lstm_out, self.hidden = self.lstm(
      x.view(len(x), self.seq_len, -1),
      self.hidden
    )
    last_time_step = \
      lstm_out.view(self.seq_len, len(x), self.hidden_size)[-1]
    y_pred = self.linear(last_time_step)
    return y_pred
'''

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, p=0.5, p1=0.5, p2=0.4):
        super(RNN, self).__init__()
        self.p1 = p1
        self.p2 = p2

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = nn.LSTM(input_size, hidden_size, 1, batch_first=True) 
        self.decoder = nn.LSTM(hidden_size, input_size, 1, batch_first=True)
        self.linear = nn.Linear(input_size,input_size)

    
    
    def forward(self, x, batch_size, hidden_size, input_size):
        # Set initial hidden and cell states 
        h0 = torch.zeros(1, batch_size, hidden_size).to(device)#numlayers*numdirections, batch, hiddensize/bidirectional=True=>numdirections 2, False=>1
        c0 = torch.zeros(1, batch_size, hidden_size).to(device)#frist cellstate numlayers*numdirections, batch, hiddensize
        
        h1 = torch.zeros(1, batch_size, input_size).to(device)#numlayers*numdirections, batch, hiddensize
        c1 = torch.zeros(1, batch_size, input_size).to(device)#numlayers*numdirections, batch, hiddensize cell stae
        
        
        # Forward propagate LSTM
        x = nn.functional.dropout(x,p=self.p1, training=False)#p1=dropout rate
        x, _ = self.encoder(x, (h0,c0))  # out: tensor of shape (batch_size, seq_length, hidden_size
        x, _ = self.decoder(x, (h1,c1))
        x = nn.functional.dropout(x,p=self.p2, training=False)#p2=dropout rate
        x = self.linear(x)

        return x




