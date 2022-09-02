import calendar
import json
import math
from os import cpu_count
import pickle as pk
from math import nan
from re import I
from unittest import result
from keras.engine.input_layer import Input
import random
from datetime import date, datetime,timedelta
import pandas as pd
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from yaml import load
"""Chargement du Dataset (le préfiltrage se fera dans cette partie) et Transformation en relevant et non-relevant"""
    
def ChargerDataset(ratings,th):
    for i in range(ratings.shape[0]):
        if math.isnan(ratings['rating'][i]) :
            ratings.loc[i,'rating'] = int(0)
        if int(ratings['rating'][i]) >= int(th):
            ratings.loc[i,'rating']=int(1)
        else: ratings.loc[i,'rating']=int(0) 
def CheckValues():
    ratings = pd.read_csv("ml-100k/ratings.csv",delimiter=";")
    pivot = ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating',fill_value=0)
    list_items = pivot.columns.unique()
    movies = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
    list_movies = movies['movieId'].unique().tolist()
    for i in range(len(list_items)):
        if(list_movies.count(list_items[i]) ==0 ):
            ratings.drop(ratings[ratings['movieId']==list_items[i]].index,inplace=True)
    ratings.to_csv("specificratings.csv")
    return ratings
def ContextualisationDataset(ratings,th,genrelist,country):
    items = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
    movies = pd.read_csv("ml-100k/dbpediamovies.csv",delimiter=";")
    specificmovies = movies[movies['country'].isin(country)]
    uniques = specificmovies['name'].unique()
    for i in range(ratings.shape[0]):
        listid = GenresSpecificMovie(ratings['movieId'][i])
        temp = items.loc[items['movieId']==ratings['movieId'][i]]
        title = temp['SPARQLTitle'][temp.index[0]]
        if(len(list(set(genrelist).intersection(listid)))!=0 ):
                ratings.loc[i,'rating']=float(1)
        else: ratings.loc[i,'rating']=float(0)
    return ratings
def GenresSpecificMovie(id):
    movies = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
    moviegenre = list()
    genrelist = open("ml-100k/genres.txt","r").readlines()
    for i in range(len(genrelist)):
        temp = movies.loc[movies['movieId']==id]
        val =temp.index
        if(len(val)!=0):
            if(temp[genrelist[i].strip()][val[0]]==1):
             moviegenre.append(genrelist[i].strip())
    return moviegenre  
def GenTrainTest(nb_users,per):
    nbgen = int(nb_users*per)
    train = random.sample(range(1,nb_users),nbgen)
    test =  list()
    i=0
    while(i< (nb_users-nbgen)):
        x = random.randrange(1,nb_users)
        if train.count(x) == 0:
            test.append(x)
            i+=1     
    return train,test
def ListRelevant(matrix,n_items,ind):
    relevants = []
    for i in range(n_items):
        if(matrix.iloc[ind,i]==1):
            relevants.append(i)
    return relevants   
def ListRel(array):
    relevants = []
    for i in range(len(array)):
        if(array[i]==1):
            relevants.append(i)
    return relevants 
def ListSpecRel(array):
    relevants = []
    for i in range(len(array)):
        if(array[i]==1):
            relevants.append(list_movies[i])
    return relevants 
def Relevant(matrix):
    relevants = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if(matrix.iloc[i,j]==1) and j not in relevants:
              relevants.append(j)
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
def MostRelevantMoviesbyContext(ratings):
    currentdate = datetime.now()
    currentday = currentdate.strftime("%A")
    weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
    weekend = ["Saturday","Sunday"]
    listmovies = list()
    if(currentday in weekdays):
      for i in range(ratings.shape[0]):
        if(ratings["rating"][i]==1) and calendar.day_name[ratings["timestamp"][i].weekday()] in weekdays:
            if(ratings['movieId'][i] not in listmovies):
                listmovies.append(ratings['movieId'][i])
    else : 
      for i in range(ratings.shape[0]):
        if(ratings["rating"][i]==1 and calendar.day_name[ratings["timestamp"][i].weekday()] in weekend):
            if(ratings['movieId'][i] not in listmovies):
                listmovies.append(ratings['movieId'][i])
    return listmovies
def ColdStartUsers():
    coldstart = list()
    list_users = pivot.index.unique().tolist()
    for user in list_users:
        sum = ratings[ratings["userId"]==user][ratings["rating"] == 4].shape[0] + ratings[ratings["userId"]==user][ratings["rating"] == 5].shape[0] 
        if sum <20 :
            coldstart.append(user)
    return coldstart
def Commons(subset,subsets):
    count = 0
    for i in range(len(subsets)):
        if(len(set(subset).intersection(subsets[i]))!=0):
            count+=1
    return count
def RandomSubsets(n_items,nb):
    subsets = list()
    subset = list(range(0,n_items,1))
    for i in range(nb):
        sub = random.sample(subset,int(n_items/nb))
        if(len(subset) != int(n_items/nb)):
            subset = list(set(subset)-set(sub))
        subsets.append(sub)
    return subsets
def where(arr,nb):
    for i in range(arr.shape[0]):
        if(arr[i]==nb):
            return i
def EnsembleSamplesTraining():
  itemslist = np.loadtxt("Subsets.txt")
  i=0
  nbrel=0
  for i in range(pivot.shape[0]):
    nbrel = nbrel + len(ListRelevant(pivot,n_items,i))
  k=0
  Input = np.zeros((nbrel,n_items))
  Target = np.zeros((nbrel))
  for i in range(pivot.shape[0]):
    for j in  ListRelevant(pivot,n_items,i):
        Input[k] = np.array(pivot.iloc[i,:],copy=True)
        Input[k,j]=0
        Target[k]=j
        k+=1

  #Splitting the Data
  i=0
  for  i in range(itemslist.shape[0]):
   itembis = itemslist[i,:]
   count = 0
   for j in range(len(Target)):
        if(Target[j] in itembis):
                count+=1
   InputT = np.zeros((count,n_items))
   TargetT= np.zeros((count))
   k=0
   j=0       
   while j<len(InputT) or k<count:
        if(Target[j] in itembis):
            InputT[k]=Input[j,:]
            TargetT[k]=where(itembis,Target[j])
            k+=1
        j+=1
   InputTrain,InputTest,TargetTrain,TargetTest = train_test_split(InputT,TargetT,test_size=0.2,random_state=28)
   np.savetxt("InputTe"+str(i)+".txt",InputTest.astype(int),fmt='%d')
   np.savetxt("TargetTe"+str(i)+".txt",TargetTest.astype(int),fmt='%d')
   np.savetxt("InputTr"+str(i)+".txt",InputTrain.astype(int),fmt='%d')
   np.savetxt("TargetTr"+str(i)+".txt",TargetTrain.astype(int),fmt='%d')
def EnsembleSamplesTesting(nb):
    itemslist = np.loadtxt("Subsets.txt")
    itemlist = np.concatenate(itemslist)
    values = list()
    for i in range(itemslist.shape[0]):
        model = load_model(str(i))
        testUser = np.array(pivot.iloc[nb,:],copy=True)
        testUser = testUser.reshape(1,testUser.shape[0])
        results = model.predict(testUser)
        values.append(results)
    results = np.concatenate(np.asarray(values))
    results = np.argsort(results.reshape(itemlist.shape[0]))[::-1] 
    for i in range(results.shape[0]):
        results[i] = int(itemlist[results[i]]) 
    return results
def EnsembleLearning():
 itembis = np.loadtxt("Subsets.txt")
 i=0
 for i in range(itembis.shape[0]):
  liste = itembis[i,:]
  InputTest = np.loadtxt("InputTe"+str(i)+".txt")
  TargetTest = np.loadtxt("TargetTe"+str(i)+".txt")
  InputTrain =  np.loadtxt("InputTr"+str(i)+".txt")
  TargetTrain = np.loadtxt("TargetTr"+str(i)+".txt")
  model = Sequential()
  model.add(Input(shape=InputTrain.shape[1]))
  model.add(Dense(200, activation='relu'))
  model.add(Dropout(rate=0.2))
  model.add(Dense(100, activation='relu'))
  model.add(Dropout(rate=0.2))
  model.add(Dense(len(liste),activation='softmax'))
  model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
  model.summary()
  history = model.fit(InputTrain,TargetTrain,validation_data=(InputTest,TargetTest),epochs=80,batch_size=150)
  model.save(format(i))
def MitigateColdStart():
    coldstartusers = ColdStartUsers()
    popularmovies = pd.read_csv("popularmovies.csv",delimiter=";")
    titles = popularmovies.movieId.unique().tolist()
    newratings = pd.DataFrame(columns=['movieId','userId','rating','review_date','review_content'])
    for user in coldstartusers:
        movielist = ratings[ratings["userId"]==user]["movieId"].unique().tolist()
        uniquetitles = list(set(titles)-set(movielist))
        print(user)
        print(len(uniquetitles))
        sum = ratings[ratings["userId"]==user][ratings["rating"] == 4].shape[0] + ratings[ratings["userId"]==user][ratings["rating"] == 5].shape[0] 
        newmovies = random.sample(uniquetitles,20-sum)
        for mv in newmovies:
            newrating = random.randrange(4,5)
            newratings.loc[len(newratings.index)] = [mv,user,newrating,'','']
    newratings.to_csv("new_ratings.csv")


"""Création des inputs et targets du RDN"""
"""
ratings = pd.read_csv("normalizedreviews.csv",delimiter=";",parse_dates=['review_date'],infer_datetime_format=True)
popularmovies = pd.read_csv("popularmovies.csv",delimiter=";")
movies = pd.read_csv("movies.csv",delimiter=";")
ratings = pd.merge(ratings,movies,on='movieId')
ChargerDataset(ratings,4)
pivot = ratings.pivot_table(index=['userId'],columns=['movie_title'],values='rating',fill_value=0)
n_users = pivot.index.unique().shape[0]
n_items = pivot.columns.unique().shape[0]
list_movies = pivot.columns.unique().tolist()
list_users = pivot.index.unique().tolist()
"""
"""

i=0
nbrel=0
for i in range(pivot.shape[0]):
    nbrel = nbrel + len(ListRelevant(pivot,n_items,i))
k=0
InputA = np.zeros((nbrel,n_items))
Target = np.zeros((nbrel))

for i in range(pivot.shape[0]):
  for j in  ListRelevant(pivot,n_items,i):
        InputA[k] = np.array(pivot.iloc[i,:],copy=True)
        InputA[k,j]=0
        Target[k]=j
        k+=1
traintest = GenTrainTest(nbrel,0.8)
train = traintest[0]
test = traintest[1]
InputTr = np.zeros((len(train),n_items))
InputTe = np.zeros((len(test),n_items))
TargetTr = np.zeros((len(train)))
TargetTe = np.zeros((len(test)))
for i in range(len(train)):
    InputTr[i]=InputA[train[i]-1,:]
    TargetTr[i]=Target[train[i]-1]
for i in range(len(test)):
    InputTe[i]=InputA[test[i]-1,:]
    TargetTe[i]=Target[test[i]-1]

np.savetxt("InputTe.txt",InputTe.astype(int),fmt='%d')
np.savetxt("TargetTe.txt",TargetTe.astype(int),fmt='%d')
np.savetxt("InputTr.txt",InputTr.astype(int),fmt='%d')
np.savetxt("TargetTr.txt",TargetTr.astype(int),fmt='%d')

"""


"""
model = Sequential()
model.add(Input(shape=InputTr.shape[1]))
model.add(Dense(100, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(InputTr.shape[1],activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(InputTr,TargetTr,validation_data=(InputTe,TargetTe),epochs=80,batch_size=250)

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


relevanttotal = Relevant(pivot)
testmovies = random.sample(relevanttotal,80)
testusers = list()
i=0
while i <pivot.shape[0]:
    relevants = ListRelevant(pivot,pivot.shape[1],i)
    if(len(relevants)>20):
      if(len(set(relevants).intersection(testmovies))>0 ):
        testusers.append(i)
    i+=1
if(len(testusers)>0):
    testusers = random.sample(testusers,25)
"""


"""
j=0
n=96
totalprec = list()
totalrec = list()
for j in range(pivot.shape[0]):
 recalls = list()
 precisions = list()
 recalls.append(j)
 precisions.append(j)
 i=1 
 testUser = np.array(pivot.iloc[j,:],copy=True)
 rev  = ListSpecRel(testUser)
 if(len(rev)!=0):
  results = EnsembleSamplesTesting(j)
  recalls.append(len(rev))
  precisions.append(len(rev))    
  while(i<n):   
    hr=0
    temp =results[:i]
    for k in range(len(temp)):
         if  list_movies[int(temp[k])] in rev:
          hr+=1
    prec = (hr)/i
    rec =  (hr)/len(rev)
    precisions.append(prec)
    recalls.append(rec)
    i+=5
  totalprec.append(np.asarray(precisions))
  totalrec.append(np.asarray(recalls))
np.savetxt("AllPrecisions.txt", np.vstack(totalprec).astype(float),fmt='%.2f')
np.savetxt("AllRecalls.txt",np.vstack(totalrec).astype(float),fmt='%.2f')
"""