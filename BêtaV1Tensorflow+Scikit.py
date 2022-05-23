from keras.engine.input_layer import Input
import random
import pandas as pd
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout
import tensorflow as tf
import matplotlib.pyplot as plt
from yaml import load
"""Chargement du Dataset (le préfiltrage se fera dans cette partie) et Transformation en relevant et non-relevant"""
def ChargerDataset(path,th):
    ratings = pd.read_csv(path,parse_dates=['timestamp'])
    """rand_userIds = np.random.choice(ratings['userId'].unique(), 
                                size=int(len(ratings['userId'].unique())*per), 
                                replace=False)

    ratings = ratings.loc[ratings['userId'].isin(rand_userIds)]
    ls = []
    ls.extend(ratings.index[(ratings['rating']>=0)])"""
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
"""Création des inputs et targets du RDN"""
"""
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
nbrel=0
for nb in test:
    nbrel = nbrel + len(ListRelevant(pivot,n_items,nb))
InputTe = np.zeros((nbrel,n_items))
TargetTe = np.zeros((nbrel))
i=0
for nb in test:
  for j in  ListRelevant(pivot,n_items,nb):
        InputTe[i] = np.array(pivot.iloc[nb,:],copy=True)
        InputTe[i,j]=0
        TargetTe[i]=j
        i+=1

np.savetxt("InputTe.txt",InputTe.astype(int),fmt='%d')
np.savetxt("TargetTe.txt",TargetTe.astype(int),fmt='%d')
np.savetxt("InputTr.txt",InputTr.astype(int),fmt='%d')
np.savetxt("TargetTr.txt",TargetTr.astype(int),fmt='%d')
"""
InputTr = np.loadtxt("InputTr.txt")
InputTe = np.loadtxt("InputTe.txt")
TargetTr = np.loadtxt("TargetTr.txt")
TargetTe = np.loadtxt("TargetTe.txt")
"""
model = Sequential()
model.add(Input(shape=(InputTr.shape[1])))
model.add(Dense(200, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(100,activation='relu'))
model.add(Dense(InputTr.shape[1],activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

print("Evaluate on test data")
results = model.evaluate(InputTe, TargetTe, batch_size=128)
print("test loss, test acc:", results)
"""
model = load_model("model1m")
movies = pd.read_csv("ml-100k/item.csv",delimiter=";")
ratings = ChargerDataset("ml-100k/ratings.csv",4)
pivot = ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating',fill_value=0)
testUser = np.array(pivot.iloc[19,:],copy=True)
testUser = testUser.reshape(1,testUser.shape[0])
rev=ListRelevant(pivot,testUser.shape[1],19)
testUser[0,rev[0]]=0
results = model.predict(testUser)
results = np.argsort(results.reshape(testUser.shape[1]))[::-1]
print(results.shape)
print(results[:5])
for i in range(5):
    print(movies['Title'][results[i]])
print("-------")
print("The target : ")
print(movies['Title'][rev[0]])
