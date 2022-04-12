import random
from tkinter.tix import InputOnly
from cv2 import merge
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
np.random.seed(123)
"""Chargement du Dataset (le préfiltrage se fera dans cette partie) et Transformation en releveant et non-relevant"""
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
    test =  random.sample(range(1,nb_users),nb_users-nbgen)
    return train,test
def ListRelevant(matrix,n_items,ind):
    relevants = []
    for i in range(n_items):
        if(matrix[ind,i]==1):
            relevants.append(i)
    return relevants   
def GenInputUser(matrix,n_items,ind):
    Input = list()
    for i, j in zip(range(len(ListRelevant(matrix,n_items,nb))), ListRelevant(matrix,n_items,nb)):
     copy = np.array(matrix[ind-1,:],copy=True)
     copy[j]=0
     Input.append(copy)   
    return Input
    
"""Création des inputs et targets du RDN"""
ratings = ChargerDataset("ratings.csv",4)
pivot = ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating',fill_value=0)

n_users = pivot.index.unique().shape[0]
n_items = pivot.columns.unique().shape[0]
list_movies = pivot.columns.unique()
list_users = pivot.index.unique()
matrix = np.zeros((n_users,n_items))
for i in range(n_users):
    for j in range(n_items):
        if pivot.iloc[i,j]==1:
            matrix[i,j]=1

train = GenTrainTest(n_users,0.8)[0]
test = GenTrainTest(n_users,0.8)[1]
InputTr = list()
TargetTr = list()
for nb in train:
 for i, j in zip(range(len(ListRelevant(matrix,n_items,nb))), ListRelevant(matrix,n_items,nb)):
    copy = np.array(matrix[nb-1,:],copy=True)
    copy[j]=0
    target = np.zeros(n_items)
    target[j]=1
    TargetTr.append(target)
    InputTr.append(copy)
print("WTF")
clf = MLPClassifier(
hidden_layer_sizes=(200,100),max_iter=80,activation='relu',solver='adam',random_state=1)
clf.fit(InputTr,TargetTr)
print("Training Done")
Input = GenInputUser(matrix,n_items,test[0]-1)
print(test[0])
pred = clf.predict_proba(Input) 
TopN = 20
ListTop = list()
for i in range(TopN):
    ListTop.append(np.where(pred[i] == np.amax(pred[i])))
print("Voici les films recommandés à l'utilisateur")
print(ListTop)