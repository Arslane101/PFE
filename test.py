from keras.engine.input_layer import Input
import random
import csv
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv1D,Dropout,Flatten
import tensorflow as tf
import matplotlib as plt
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
def Shit(n):
    largest_divisor = 0
    for i in range(2, n):
        if n % i == 0:
            largest_divisor = i
    return largest_divisor
"""Création des inputs et targets du RDN"""
gc.enable()
ratings = ChargerDataset("../input/the-movies-dataset/ratings_small.csv",4)
pivot = ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating',fill_value=0)
ratings = None
n_users = pivot.index.unique().shape[0]
n_items = pivot.columns.unique().shape[0]
list_movies = pivot.columns.unique()
list_users = pivot.index.unique()
train = GenTrainTest(n_users,0.8)[0]
test = GenTrainTest(n_users,0.8)[1]
InputTr=list()
TargetTr=list()
model = Sequential()
model.add(Input(shape=(1,n_items)))
model.add(Dense(20, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(10,activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(n_items,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
for nb in train:
 for j in  ListRelevant(pivot,n_items,nb):
    copy = np.array(pivot.iloc[nb,:],copy=True)
    copy[j]=0
    copy = np.reshape(copy,(1, n_items))
    target = np.zeros(n_items)
    target[j]=1
    target = np.reshape(target,(1, n_items))
    InputTr.append(copy)
    TargetTr.append(target)
print(InputTr[0].shape)
InputT = np.zeros((len(InputTr),n_items))
TargetT = np.zeros((len(TargetTr),1))
for i in range (len(InputTr)):
  InputT[i]=InputTr[i]
  TargeT[i]=TargetrT[i]
InputTr.clear()
TargetTr.clear()
Input = tf.data.Dataset.from_tensor_slices((InputT,TargetT))

print("Wow")
history = model.fit(Input, epochs=10,batch_size=100)

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