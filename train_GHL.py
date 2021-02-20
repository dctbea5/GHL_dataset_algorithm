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
from ed_model import *
import train

# In[2]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    #GHL_Anomaly_detection
    
    parser = argparse.ArgumentParser(description='GHL Anomaly Detection Module')
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = 'train_1500000_seed_11_vars_23.csv',type=str)
    parser.add_argument("--seq_len", dest = "seq_len", help = "seqeunce length", default = 30)
    parser.add_argument("--pred_d", dest = "pred_d", help = "predict d", default = 0)
    parser.add_argument("--lr", dest = 'lr', help = 
                        "learning rate",default = 0.0002)
    parser.add_argument("--k_num",type=str, dest='k_num' ,default=3, help="k fold number")
    parser.add_argument("--input_size", dest = 'input_size', help = 
                        "input size",default = 5)
    parser.add_argument("--hidden_size", dest = 'hidden_size', help = 
                        "hidden size",default =10)
    parser.add_argument("--batch_size", dest = "batch_size", help = "RNN num layers", default = 30)
    parser.add_argument("--num_layers", dest = "num_layers", help = "RNN num layers", default = 2)
    parser.add_argument("--n_epoch", dest = "n_epoch", help = "RNN num layers", default = 1)

    return parser.parse_args()

# In[10]:
args = arg_parse()
seq_len=int(args.seq_len)
pred_d=int(args.pred_d)
k_num=int(args.k_num)
batch_size=int(args.batch_size)
lr=float(args.lr)
input_size=int(args.input_size)
hidden_size=int(args.hidden_size)
learning_rate=lr
n_epoch=int(args.n_epoch)
num_layers=int(args.num_layers)

def main():
    torch.cuda.set_device(device)
    model = RNN(input_size, hidden_size, num_layers)
    model = model.to(device)
    print(model)
    model, history = train.train_model(model, n_epochs =n_epoch, batch_size=batch_size, hidden_size=hidden_size, input_size=input_size, seq_len=seq_len, pred_d=pred_d, k_num=k_num)
    torch.save(model.state_dict(), 'model_best.ckpt')
    file_name = 'k-fold_GHL_test.pkl' 
    joblib.dump(model, file_name)
    with open('model_history_next.pickle', 'wb') as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)
    
    with open('model_history_next.pickle', 'wb') as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)  

if __name__ == "__main__":
    print('GHL_start!')
    main()
      





