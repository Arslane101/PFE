from cv2 import merge
import pandas as pd
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

"""wut"""
np.random.seed(123)
ratings = pd.read_csv('ratings.csv',parse_dates=['timestamp'])
n_users = ratings['userId'].unique().shape[0]
n_items = ratings['movieId'].unique().shape[0]
th = 4
for i in range(ratings.shape[0]):
        if ratings['rating'][i] >= float(th):
            ratings.loc[i,'rating']=float(1)
        else: ratings.loc[i,'rating']=float(0)

pivot = ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating',fill_value=0)
print(pivot)