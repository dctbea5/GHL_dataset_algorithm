#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import copy
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
torch.manual_seed(1)
import matplotlib as mpl
import matplotlib.pylab as plt
import time
import glob,pickle
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import KFold


# In[34]:


df=pd.read_csv('train_1500000_seed_11_vars_23.csv')
df = df.sample(frac=1).reset_index(drop=True)
#1272833
train_data=df#[:100000]
print(df.shape)
#val_data=df[1272833:1403906]


# In[35]:


train_data=train_data.loc[:,['RT_level_ini','RT_temperature.T','HT_temperature.T','RT_level','dT_rand']]
#val_data=val_data.loc[:,['RT_level_ini','RT_temperature.T','HT_temperature.T','RT_level','dT_rand']]
df_input_x=train_data.values
n_features=df_input_x.shape[1]
df_input_y = np.zeros(shape=(1535118,), dtype=np.float64)
df_input_x.shape


# In[19]:

# In[95]:


#class dataprocessing(Dataset):
def datapro(idx_train,idx_test,input_x,input_y,k=3,timesteps=30,n_features=5):
    data_x_train,data_x_test=input_x[idx_train],input_x[idx_test]
    data_y_train,data_y_test=input_y[idx_train],input_y[idx_test]        
    def temporalize(X, y, timesteps):
        output_X = []
        output_y = []
        for i in range(len(X) - timesteps - 1):
            t = []
            for j in range(1, timesteps + 1):
        # Gather the past records upto the lookback period
                t.append(X[[(i + j + 1)], :])
            output_X.append(t)
            output_y.append(y[i + timesteps + 1])
        return np.squeeze(np.array(output_X)), np.array(output_y)
    in_df_x,in_df_y = temporalize(data_x_train,data_y_train, timesteps)
    x_train, x_test, y_train, y_test = train_test_split(in_df_x, in_df_y, test_size=0.3)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3)
    def flatten(x) :
        num_instances, num_time_steps, num_features = x.shape
        x = np.reshape(x, newshape=(-1, num_features))
        return x 

    def scale(x,scaler) :
        scaled_x = scaler.transform(x)
        return scaled_x

    def reshape(scaled_x , x) :
        num_instances, num_time_steps, num_features = x.shape
        reshaped_scaled_x =    np.reshape(scaled_x, newshape=(num_instances, num_time_steps, num_features))
        return reshaped_scaled_x
    x_train_y0 = x_train[y_train == 0]
    x_valid_y0 = x_valid[y_valid == 0]
    scaler = StandardScaler().fit(flatten(x_train_y0))
    x_train_y0_scaled = reshape(scale(flatten(x_train_y0), scaler),x_train_y0)
    #x_valid_scaled = reshape(scale(flatten(x_valid), scaler),x_valid)
    x_valid_y0_scaled = reshape(scale(flatten(x_valid_y0), scaler),x_valid_y0)
    #x_test_scaled = reshape(scale(flatten(x_test), scaler),x_test)
    #print(self.x_train_y0_scaled.shape)
    return x_train_y0_scaled, x_valid_y0_scaled

    
# In[ ]:



# In[13]:


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=16):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = (
            embedding_dim, 2 * embedding_dim
        )
        self.rnn1 = nn.LSTM(
          input_size=n_features,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )
        self.rnn2 = nn.LSTM(
          input_size=self.hidden_dim,
          hidden_size=embedding_dim,
          num_layers=1,
          batch_first=True
        )
    def forward(self, x):
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return  x[:,-1,:]


# In[14]:


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


# In[15]:


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=16, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
          input_size=input_dim,
          hidden_size=input_dim,
          num_layers=1,
          batch_first=True
        )
        self.rnn2 = nn.LSTM(
          input_size=input_dim,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )
        self.output_layer = torch.nn.Linear(self.hidden_dim, n_features)
        self.timedist = TimeDistributed(self.output_layer)
        
    def forward(self, x):
        x=x.reshape(-1,1,self.input_dim).repeat(1,self.seq_len,1)       
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        return self.timedist(x)


# In[16]:


class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=16):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim)#.to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)#.to(device)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# In[17]:


class AutoencoderDataset(Dataset): 
    def __init__(self,x):
        self.x = x
    def __len__(self): 
        return len(self.x)
    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x[idx,:,:])
        return x


# In[98]:


def train_model(model, input_x,input_y,n_epochs,batch_size):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss().to(device)
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    history = []
    models = []
    best_losses = []
    for _ in range(4):
        history.append(dict(train=[], val=[]))
        models.append(model)
        best_losses.append(best_loss)
    print("start!")
    #scores = np.zeros(3)
    cv = KFold(3, shuffle=True, random_state=0)
    for i, (idx_train, idx_test) in enumerate(cv.split(input_x)):
        df_train,df_test=datapro(idx_train,idx_test,input_x,input_y,k=3,timesteps=30,n_features=5)
        # = train_dataset.iloc[idx_train]
        #df_test = train_dataset.iloc[idx_test]
        train_dataset_ae = AutoencoderDataset(df_train)
        tr_dataloader = DataLoader(train_dataset_ae, batch_size=batch_size, 
                               shuffle=False,num_workers=8)
        val_dataset_ae = AutoencoderDataset(df_test)
        va_dataloader = DataLoader(val_dataset_ae, batch_size=len(df_test),
                               shuffle=False,num_workers=8)
        for epoch in range(1, n_epochs + 1):
            model = models[i].to(device)
            model=model.train()
            train_losses = []
            for batch_idx, batch_x in enumerate(tr_dataloader):
                optimizer.zero_grad()
                batch_x_tensor = batch_x.to(device)
                seq_pred = model(batch_x_tensor)
                loss = criterion(seq_pred, batch_x_tensor)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            val_losses = []
            model = model.eval()
            with torch.no_grad():
                va_x  =next(va_dataloader.__iter__())
                va_x_tensor = va_x.to(device)
                seq_pred=model(va_x_tensor)
                loss_pred = criterion(seq_pred, va_x_tensor)
                val_losses.append(loss_pred.item())
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            history[i]['train'].append(train_loss)
            history[i]['val'].append(val_loss)
            if val_loss < best_losses[i]:
                best_losses[i] = val_loss
                models[i] = copy.deepcopy(model)
                torch.save(model.state_dict(), f'model_{i}_{epoch}.ckpt')    
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_dic = copy.deepcopy(model.state_dict())
            if (epoch+1) % 10 == 0:
                results_files=sorted([file for file in glob.glob('GHL_result/'+'*.pkl')])
            print(f'Epoch {epoch} / Group {i} : train loss {train_loss} val loss {val_loss}')
    model.load_state_dict(best_model_dic)
    return model.eval(), history


# In[21]:

timesteps=30
n_features=5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
model = RecurrentAutoencoder(timesteps, n_features, 16)
model = model.to(device)
model, history = train_model(model, df_input_x , df_input_y,
                             n_epochs = 10, batch_size=256)


# In[1]:


import joblib
#from sklearn.externals import joblib 

file_name = 'k-fold_30.pkl' 
joblib.dump(model, file_name)


# In[ ]:




