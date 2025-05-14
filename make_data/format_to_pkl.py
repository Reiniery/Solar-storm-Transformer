# -*- coding: utf-8 -*-
"""
Created on Thu May  8 10:55:31 2025

@author: beast
"""


import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import data_utils as du


data = du.Data("omni_data_fmt.csv")
data.load_omni()
df= data.filter_time("2024-12-01", "2024-12-20")


############ Choose Varaibles to Use #######################

modality1 = df["Dst"]
modality2 = df["Bz"]
modality3 = df["channel_3"]
modality4 = df["channel_4"]



#split
X =list(zip(modality1.values,modality2.values,modality3.values,modality4.values))
y = df["label"]

# Split into train and temp (70% train, 30% temp)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y)

# Split temp into validation and test (15% val, 15% test)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp)

def unzip_modalities(X):
    modality1, modality2, modality3, modality4 = zip(*X)
    return (
        np.stack(modality1).reshape(len(X),-1,1),
        np.stack(modality2).reshape(len(X),-1,1),
        np.stack(modality3).reshape(len(X),-1,1),
        np.stack(modality4).reshape(len(X),-1,1)
    )


modality1_train, modality2_train, modality3_train, modality4_train = unzip_modalities(X_train)
modality1_val, modality2_val, modality3_val, modality4_val = unzip_modalities(X_val)
modality1_test, modality2_test, modality3_test, modality4_test = unzip_modalities(X_test)

train_ids = np.arange(1, len(modality1_train) + 1).astype(np.int64)
val_ids = np.arange(1, len(modality1_val) + 1).astype(np.int64)
test_ids = np.arange(1, len(modality1_test) + 1).astype(np.int64)


split_data ={
    'train':{'id':train_ids.reshape(-1,1,1), "modality1":modality1_train,"modality2":modality2_train,"modality3":modality3_train,"modality4":modality4_train,"label": y_train.to_numpy().reshape(-1,1,1)},
    'valid':{'id':val_ids.reshape(-1,1,1),"modality1":modality1_val,"modality2":modality2_val,"modality3":modality3_val,"modality4":modality4_val,"label": y_val.to_numpy().reshape(-1,1,1)},
    'test':{'id':test_ids.reshape(-1,1,1),"modality1":modality1_test,"modality2":modality2_test,"modality3":modality3_test,"modality4":modality4_test,"label": y_test.to_numpy().reshape(-1,1,1)}    
    
    }
with open('Combined.pkl', 'wb') as f:
    pickle.dump(split_data, f)
   
with open('cogload.pkl', 'rb') as f:
    
    cogload = pickle.load(f)
    
    
with open("combined.pkl", "rb") as f:
    combined = pickle.load(f)

# print(dataset.keys())  # Should show: train, valid, test
# print(dataset['train'].keys())  # Should show: id, modality1, modality2, modality3, modality4
