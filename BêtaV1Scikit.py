from audioop import reverse
import random
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from sklearn.neural_network import MLPClassifier
import pickle
np.random.seed(123)
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
def GenInputTargetUser(matrix,n_items,ind):
    Input = list()
    Target = list()
    for i, j in zip(range(len(ListRelevant(matrix,n_items,nb))), ListRelevant(matrix,n_items,nb)):
     copy = np.array(matrix[ind-1,:],copy=True)
     copy[j]=0
     Target.append(j)
     Input.append(copy)   
    return Input,Target
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

##clf = pickle.load(open('modelrec.sav','rb'))

clf = MLPClassifier(
hidden_layer_sizes=(400,200),max_iter=100,activation='relu',solver='adam')
train = GenTrainTest(n_users,0.8)[0]
test = GenTrainTest(n_users,0.8)[1]

InputTr=list()
TargetTr=list()
taille = len(train)
k=0
for nb in train:
 for i, j in zip(range(len(ListRelevant(matrix,n_items,nb))), ListRelevant(matrix,n_items,nb)):
    copy = np.array(matrix[nb-1,:],copy=True)
    copy[j]=0
    target = np.zeros(n_items)
    target[j]=1
    TargetTr.append(target)
    InputTr.append(copy)
 if(k==(taille/4)-1):
        clf.fit(InputTr,TargetTr)
        InputTr.clear()
        TargetTr.clear()
        k=0
        print("step")
 else : k+=1
        
print("WTF")
    

print("Training Done")
pickle.dump(clf,open('modelrec.sav','wb'))

Input = GenInputTargetUser(matrix,n_items,test[0]-1)[0]
Target = GenInputTargetUser(matrix,n_items,test[0]-1)[1]
pred = clf.predict(Input)
print(pred)
print("--------")
print(Target)
print(precision_score(Target,pred,average=None))


"""
Input = list()
copy = np.array(matrix[test[0]-1,:],copy=True)
Input.append(copy)
pred = clf.predict_proba(Input)

print(np.argsort(pred[0])[::-1])
TopN = 20
recID = list()
for i in range(TopN):
    recID.append(list_movies[pred[0][i]])
print("Liste des Films Recommendés")
print(recID)"""
