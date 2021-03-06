# -*- coding: utf-8 -*-
"""Emotion Analyzer BERT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cNPCwUUQTa6kueQxOU09QZpN6isXRJtZ
"""

import numpy as np
from tqdm import tqdm 
import pandas as pd

# we will do a stratified split
from sklearn.model_selection import train_test_split
import torch
# tokenize 
from transformers import BertTokenizer
from torch.utils.data import TensorDataset  

def data_wrangle():
    print("Wrangling.........")

    df = pd.read_csv('data/smile-annotations-final.csv',header = None,names=['id','text','category'])
    df.set_index('id',inplace = True)

    df = df[~df['category'].str.contains('\|')]
    df = df[df['category']!='nocode']

    df['category'].value_counts()

    labels = df['category'].unique()
    label_dict = {}
    for idx,label in enumerate(labels):
        label_dict[label] = idx
        
    df['label'] = df['category'].map(label_dict)

    X_train,X_val,y_train,y_val = train_test_split(
        df.index.values,
        df.label.values,
        test_size = 0.10,
        stratify = df.label.values
    )

    df['data_type'] = ['not']*df.shape[0]
    

    df.loc[X_train,'data_type'] = 'train'
    df.loc[X_val,'data_type'] = 'val'

    print("Finished")

    return df
    