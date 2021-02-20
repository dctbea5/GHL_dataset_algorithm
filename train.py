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
from Dataset import *
from ed_model import *
# In[2]:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Loss and optimizer
def train_model(model, n_epochs, batch_size,hidden_size,input_size, seq_len , pred_d, k_num):
    criterion = nn.MSELoss(reduction='mean')
    global best_model_dic
    
    count = 0
    best_loss = 10000000
    history = []
    models = []
    best_losses = []
    for _ in range(4):#dataset.getForNum()
        
        history.append(dict(train=[], val=[]))
        models.append(model)
        best_losses.append(best_loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_model_dic = copy.deepcopy(model.state_dict())   
    path = r'GHL_trainset'
    file_list=glob.glob(os.path.join(path,"*.csv"))

    for file in file_list:
        print({file})
        #data_csv=pd.read_csv(file,engine='python')
        dataset = GhlDataset(file ,seq_len , pred_d, k_num)
        

        print('Start training')

        for epoch in range(n_epochs): #epoch 200
            for n in range(dataset.getForNum()):
                train_losses=[]
                val_losses = [] 
                dataset.usegroup(n)
                dataset.setTrain()
                train_dataloader = DataLoader(dataset, batch_size = batch_size, shuffle= True, num_workers=3)
                train_step = len(train_dataloader)
        
                model = models[n].to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
                model = model.train()
        
        
                for k, (prev_seq, next_seq) in enumerate(train_dataloader):

                    prev_seq = prev_seq.to(device)
                    next_seq = next_seq.to(device)
            # Forward pass
                    outputs= model(prev_seq, batch_size, hidden_size, input_size)
                    train_loss = criterion(outputs, next_seq)

            # Backward and optimize
                    optimizer.zero_grad()

                    train_loss.backward()

                    optimizer.step()

                    train_losses.append(train_loss.item())

           # if (k+1) % 1 == 0:
           #     print ( 'Epoch:{} Group:{} Step [{}/{}], Train Loss: {}'.format(epoch+1 , n+1,k+1,train_step,train_loss.item()))
            
               
                model.eval()

                dataset.setTest()
                val_dataloader = DataLoader(dataset, batch_size = batch_size, shuffle= True, num_workers=3)

                with torch.no_grad():
                    for k, (prev_seq, next_seq) in enumerate(val_dataloader):
                        prev_seq = prev_seq.to(device)
                        next_seq = next_seq.to(device)
                # Forward pass
                        outputs= model(prev_seq, batch_size, hidden_size, input_size)
                        val_loss = criterion(outputs, next_seq)

                        val_losses.append(val_loss.item())

                train_loss_m = np.mean(train_losses)
                val_loss_m = np.mean(val_losses)


                history[n]['train'].append(train_loss_m)
                history[n]['val'].append(val_loss_m)
        
                if val_loss < best_losses[n]:
                    best_losses[n] = val_loss
                    models[n] = copy.deepcopy(model)
                    torch.save(model.state_dict(), f'model_{n}_{epoch}.ckpt')

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_dic = copy.deepcopy(model.state_dict())
                #if (epoch+1) % 5 == 0:
                print(f'Epoch {epoch+1}: Group {n+1} train loss {train_loss} val loss {val_loss}')
        


    model.load_state_dict(best_model_dic)
    return model.eval(), history

