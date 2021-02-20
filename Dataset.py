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

class GhlDataset(Dataset):

    def __init__(self, path ,seq_len , pred_d, k_num=3):
        #Load csv Files
        self.csv_files = path  
        self.seq_len = seq_len
        self.pred_d = pred_d
        
        data = pd.read_csv(path)
        data=data.loc[:,['Time','RT_level', 'RT_temperature.T', 'HT_temperature.T', 'inj_valve_act', 'heater_act']]
        data = data.astype({'Time':'int'})
        self.data = data.drop_duplicates(["Time"]).reset_index(drop=True)
    
        self.length = len(self.data)
        self.k_num = k_num
        self.kf = KFold(n_splits=k_num,shuffle=True, random_state=True)
        #self.batch_size=batch_size
        len_ = self.length - (self.seq_len-1)- self.pred_d*self.seq_len
        
        self.x = [[0] for _ in range(len_) ]
        self.kf_gen = self.kf.split(self.x)
        self.for_num = self.kf.get_n_splits(self.x)
        
        self.trains = []
        self.tests = []
        
        for train_idx, test_idx  in self.kf_gen:
            random.shuffle(train_idx)
            random.shuffle(test_idx)
            
            self.trains.append(train_idx)
            self.tests.append(test_idx)
        
        print("data frame total data length: {}".format(self.length))
        print("sequence length: {}".format(self.seq_len))
        print("loop num:{}".format(self.for_num))
        
    def getXSeq(self, idx):
        return torch.FloatTensor(self.data[idx : idx + self.seq_len].drop(['Time'], axis=1).values)
    
    def getYSeq(self, idx):
        return torch.FloatTensor(self.data[idx+self.seq_len*self.pred_d : idx+(self.pred_d+1)*self.seq_len].drop(['Time'], axis=1).values)
                                       
        
    def usegroup(self, n):
        self.list_idx = n
        return
        
    def setTrain(self):
        self.dataset_idx = self.trains[self.list_idx]
        return
    

    def setTest(self):
        self.dataset_idx = self.tests[self.list_idx]
        return
    
    def getForNum(self):
        return self.for_num
    
        
    def __len__(self):
        # number of csv sample files
        return len(self.dataset_idx)-len(self.dataset_idx)%30#batch_size
    
    
    
    def __getitem__(self, idx):
        
        data_idx = self.dataset_idx[idx]
        
        return   (self.getXSeq(data_idx),self.getYSeq(data_idx))



