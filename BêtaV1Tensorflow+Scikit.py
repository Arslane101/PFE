from keras.engine.input_layer import Input
import random
import csv
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv1D,Dropout,Flatten
import tensorflow as tf
import matplotlib.pyplot as plt
import gc
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
        if(matrix.iloc[ind,i]==1):
            relevants.append(i)
    return relevants   
def GenInputTargetUser(matrix,n_items,ind):
    Input = list()
    Target = list()
    for i, j in zip(range(len(ListRelevant(matrix,n_items,nb))), ListRelevant(matrix,n_items,nb)):
     copy = np.array(matrix[ind,:],copy=True)
     copy[j]=0
     Target.append(j)
     Input.append(copy)   
    return Input,Target
"""Création des inputs et targets du RDN"""
gc.enable()
ratings = ChargerDataset("../input/movies/ratings.csv",4)
pivot = ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating',fill_value=0)
ratings = None
n_users = pivot.index.unique().shape[0]
n_items = pivot.columns.unique().shape[0]
list_movies = pivot.columns.unique()
list_users = pivot.index.unique()
train = GenTrainTest(n_users,0.8)[0]
test = GenTrainTest(n_users,0.8)[1]
i=0
nbrel=0
for nb in train:
    nbrel = nbrel + len(ListRelevant(pivot,n_items,nb))
    
InputTr = np.zeros((nbrel,n_items))
TargetTr = np.zeros((nbrel))
for nb in train:
  for j in  ListRelevant(pivot,n_items,nb):
        InputTr[i] = np.array(pivot.iloc[nb,:],copy=True)
        InputTr[i,j]=0
        TargetTr[i]=j
        i+=1

model = Sequential()
model.add(Input(shape=(nbrel,n_items)))
model.add(Dense(700, activation='selu'))
model.add(Dropout(rate=0.2))
model.add(Dense(500,activation='selu'))
model.add(Dropout(rate=0.2))
model.add(Dense(n_items,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(InputTr,TargetTr,epochs=50,batch_size=150)
"""
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()"""