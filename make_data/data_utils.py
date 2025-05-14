# -*- coding: utf-8 -*-
"""
Created on Thu May  8 11:41:00 2025

@author: beast
"""
import pandas as pd
import numpy as np 
import pickle
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

## CLASS - Functions for Data file used ##
class Data:
    
    def __init__(self, file):
        self.df = None
        self.file = file
        
    def load_omni(self):
        """
        This function will load datafram from csv file,
        Replace missing Data values,
        """

        df = pd.read_csv(self.file, sep='\t')    # load file into dataframe

        #dictionary for missing data values for selected columns
        error_cd ={
            "B":999.9,
            "Bx":999.9,
            "By":999.9,
            "Bz":999.9,
            "T":9999999,
            "N":999.9,
            "V":9999,
            "P":99.99,
            "Kp":99,
            "AE":9999,
            "Dst":99999,
            "SSN":999,
            "f10_7":999.9
            }
        #replace missing value with values from dictionary
        for column in df.columns[1:len(df.columns)]:
            df[column] = df[column].replace(error_cd[column], np.nan)
        # convert the date to Python datetime format 
        df['Date'] = pd.to_datetime(df['Date'])
        self.df=df
        return df

#Filter within certain points in time
    def filter_time(self, start, end):
        plot_start = pd.to_datetime(start, format='%Y-%m-%d')
        plot_stop  = pd.to_datetime(end, format='%Y-%m-%d')

        filtered_df = self.df.loc[(self.df['Date'] >= plot_start)
                           & (self.df['Date'] <  plot_stop)]
        return filtered_df

"""--------------Functions for Solar Storms-------------------------------"""

## Break and Label Data
def labeled_storm_dataset(df,back_window,front_window, var,zero_ratio):
    moderate = -50
    intense = -100
    extreme = -250
    
    # m1 =[]
    # m2=[]
    # m3=[]
    # m4=[]
    # label=[]
    all_data=[]
    
    
    for i in range(len(df)-front_window-back_window):
        #use data to see if there was storm in fronnt_window hours
       m1_backWindow = df[var[0]].iloc[i:i+back_window].values
       m2_backWindow= df[var[1]].iloc[i:i+back_window].values
       m3_backWindow = df[var[2]].iloc[i:i+back_window].values
       m4_backWindow = df[var[3]].iloc[i:i+back_window].values
       
       # m1_FrontWindow = df[var[0]].iloc[i+back_window:i+front_window+back_window].values
       # m2_FrontWindow= df[var[1]].iloc[i+back_window:i+front_window+back_window].values
       # m3_FrontWindow = df[var[2]].iloc[i+back_window:i+front_window+back_window].values
       # m4_FrontWindow = df[var[3]].iloc[i+back_window:i+front_window+back_window].values
        
       #label data depending on magnitude ov dst
       stormType = df["Dst"].iloc[i+front_window]   
       if(stormType > moderate):
            l=0 #no storm
       elif(moderate >= stormType > intense):
            l =1 #moderate
       elif( intense >= stormType > extreme):
            l =2#intense
       else:
            l=3#extreme
            
       # m1.append(m1_backWindow)
       # m2.append(m2_backWindow)
       # m3.append(m3_backWindow)
       # m4.append(m4_backWindow)
       # label.append(l)
       
       #store properties used with label
       sample = {
            'm1': m1_backWindow,
            'm2': m2_backWindow,
            'm3': m3_backWindow,
            'm4': m4_backWindow,
            'label': l
       }
       all_data.append(sample)
       
    # use percentage of no storms
    label_0 = [s for s in all_data if s['label'] == 0]
    storms = [s for s in all_data if s['label'] != 0]
    n_keep = int(len(label_0) * zero_ratio)
    label_0_sampled = resample(label_0, n_samples=n_keep, replace=False, random_state=42)
    final_data = label_0_sampled + storms
    np.random.shuffle(final_data)
    # m1=np.expand_dims(np.array(m1), axis=-1)
    # m2=np.expand_dims(np.array(m2), axis=-1)
    # m3=np.expand_dims(np.array(m3), axis=-1)
    # m4=np.expand_dims(np.array(m4), axis=-1)
    # label=np.array(label)
    # Convert to arrays
    m1 = np.expand_dims(np.array([s['m1'] for s in final_data]), axis=-1)
    m2 = np.expand_dims(np.array([s['m2'] for s in final_data]), axis=-1)
    m3 = np.expand_dims(np.array([s['m3'] for s in final_data]), axis=-1)
    m4 = np.expand_dims(np.array([s['m4'] for s in final_data]), axis=-1)
    labels = np.array([s['label'] for s in final_data])

    data ={
        'm1':m1,
        'm2':m2,
        'm3':m3,
        'm4':m4,
        'labels':labels
        
        
        }
    
    return data
            
            
        
def identifyStormsDst(df, option=None):
    moderate = -50
    intense = -100
    extreme = -250
    
    moderateArray = df.loc[(df["Dst"]<=moderate)&(df["Dst"]>intense)]
    intenseArray = df.loc[(df["Dst"]<=intense)&(df["Dst"]>extreme)]
    extremeArray = df.loc[df["Dst"]<extreme]
    if(option==None):
        return [moderateArray, intenseArray, extremeArray]
    else:
        return {
            "moderate": moderateArray,
            "intense": intenseArray,
            "extreme": extremeArray}[option]   

"""---------- Make Data ready for model -------------"""

def split_and_save_data(data, test_size=0.15, val_size=0.15, filename='omni.pkl'):
    import numpy as np
    import pickle
    from sklearn.model_selection import train_test_split

    m1 = data['m1']
    m2 = data['m2']
    m3 = data['m3']
    m4 = data['m4']
    y = data['labels']

    assert m1.shape[0] == y.shape[0], "Sample mismatch"

    indices = np.arange(len(y))

    # Split indices into train_val and test
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=y, random_state=42
    )

    # Then split train and val
    y_train_val = y[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size / (1 - test_size), stratify=y_train_val, random_state=42
    )

    def get_split(idx):
        return {
            'm1': m1[idx],
            'm2': m2[idx],
            'm3': m3[idx],
            'm4': m4[idx],
            'label': y[idx].reshape(-1,1,1).astype(np.int64)
        }

    data_dict = {
        'train': get_split(train_idx),
        'valid': get_split(val_idx),
        'test': get_split(test_idx)
    }

    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)

  
