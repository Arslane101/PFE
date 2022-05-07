from keras.engine.input_layer import Input
import random
import csv
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv1D,Flatten
import matplotlib as plt
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
ratings = ChargerDataset("/content/ratings.csv",4)
pivot = ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating',fill_value=0)

n_users = pivot.index.unique().shape[0]
n_items = pivot.columns.unique().shape[0]
list_movies = pivot.columns.unique()
list_users = pivot.index.unique()
matrix = np.zeros((n_users,n_items))
for i in range(n_users):
    for j in range(n_items):
            matrix[i,j]=pivot.iloc[i,j]

train = GenTrainTest(n_users,0.8)[0]
test = GenTrainTest(n_users,0.8)[1]
InputT=list()
TargetT=list()
for nb in train:
 for i, j in zip(range(len(ListRelevant(matrix,n_items,nb))), ListRelevant(matrix,n_items,nb)):
    copy = np.array(matrix[nb-1,:],copy=True)
    copy[j]=0
    copy = np.reshape(copy,(1, n_items))
    target = np.zeros(n_items)
    target[j]=1
    target = np.reshape(target,(1, n_items))
    InputT.append(copy)
    TargetT.append(target)
print(len(InputT))
print(TargetT)
InputTr = np.zeros((len(InputT),n_items))
TargetTr = np.zeros((len(TargetT),n_items))
for i in range (len(InputT)):
  InputTr[i]=InputT[i]
  TargetTr[i]=TargetT[i]
model = Sequential()
model.add(Input(shape=(len(InputTr),n_items)))
model.add(Dense(50, activation='relu'))
model.add(Dense(n_items,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(InputTr,TargetTr, epochs=100)
 
      
print("WTF")


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
plt.show()