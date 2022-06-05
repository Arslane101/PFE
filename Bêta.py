from keras.engine.input_layer import Input
import random
import pandas as pd
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout
import matplotlib.pyplot as plt
"""Chargement du Dataset (le préfiltrage se fera dans cette partie) et Transformation en relevant et non-relevant"""
def ChargerDataset(path,th):
    ratings = pd.read_csv(path,parse_dates=['timestamp'])
    for i in range(ratings.shape[0]):
        if ratings['rating'][i] >= float(th):
            ratings.loc[i,'rating']=float(1)
        else: ratings.loc[i,'rating']=float(0)
    return ratings
def GenTrainTest(nb_users,per):
    nbgen = int(nb_users*per)
    train = random.sample(range(1,nb_users),nbgen)
    test =  list()
    i=0
    while(i< (nb_users-nbgen)):
        x = random.randrange(1,nb_users)
        if train.count(x) == 0:
            test.append(x)
        else: i+=1     
    return train,test
def ListRelevant(matrix,n_items,ind):
    relevants = []
    for i in range(n_items):
        if(matrix[ind,i]==1):
            relevants.append(i)
    return relevants   
def GenInputTargetUser(pivot,n_items,ind):
    i=0
    Input = np.zeros((nbrel,n_items))
    Target = np.zeros((nbrel))
    for nb in train:
     for j in  ListRelevant(pivot,n_items,nb):
        Input[i] = np.array(pivot.iloc[nb,:],copy=True)
        Input[i,j]=0
        Target[i]=j
        i+=1 
    return Input,Target
def ListRel(array):
    relevants = []
    for i in range(len(array)):
        if(array[i]==1):
            relevants.append(i)
    return relevants 
movies = pd.read_csv("ml-100k/dbpediamovies.csv",delimiter=";")
list_countries = movies['country'].unique().tolist()
list_countries.remove(np.nan)
count_countries = np.zeros(len(list_countries))
j=0
for country in list_countries:
    count=0
    for i in range(movies.shape[0]):
        if(movies['country'][i]==country):
            count+=1
    count_countries[j]=count
    j+=1
values_countries = np.sort(count_countries)[::-1]
count_countries = np.argsort(count_countries)[::-1]
print(list_countries)
print(count_countries)
print(values_countries)

