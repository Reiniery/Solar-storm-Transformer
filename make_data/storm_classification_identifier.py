# -*- coding: utf-8 -*-
"""
Created on Tue May 13 11:33:29 2025

@author: beast
"""
import matplotlib.pyplot as plt

import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import data_utils as du




data = du.Data("omni_data_fmt.csv")
data.load_omni()
df= data.filter_time("2022-12-01", "2024-12-20")

X= du.labeled_storm_dataset(df,64, 24, ["Dst", "T","V","Bx"], 0.25)

split=du.split_and_save_data(X)

import pickle
with open("omni.pkl", 'rb') as f:
    data = pickle.load(f)
