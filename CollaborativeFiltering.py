import calendar
import json
import pickle as pk
from math import nan
from re import I
from keras.engine.input_layer import Input
import random
from datetime import date, datetime,timedelta
import pandas as pd
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from sklearn.metrics import jaccard_score
import tensorflow as tf
import matplotlib.pyplot as plt
from yaml import load
"""Chargement du Dataset (le préfiltrage se fera dans cette partie) et Transformation en relevant et non-relevant"""
    
def ChargerDataset(path,th):
    ratings = pd.read_csv(path,delimiter=";",parse_dates=['timestamp'])
    """rand_movies = np.random.choice(ratings['movieId'].unique(), 
                                size=int(len(ratings['movieId'].unique())*per), 
                                replace=False)

    ratings = ratings.loc[ratings['movieId'].isin(rand_movies)]
    ls = []
    ls.extend(ratings.index[(ratings['rating']>=0)])"""
    for i in range(ratings.shape[0]):
        if ratings['rating'][i] >= float(th):
            ratings.loc[i,'rating']=float(1)
        else: ratings.loc[i,'rating']=float(0) 
    ratings.to_csv("filteredratings.csv")
    return ratings
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
def GetTrendsMovies(listmovies):
    genrelist = open("ml-100k/genres.txt","r").readlines()
    movies = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
    trends = np.zeros(len(genrelist))
    for i in range(len(genrelist)):
        for id in listmovies:
            temp = movies.loc[movies['movieId']==id]
            val =temp.index
            if(len(val)!=0):
             if(temp[genrelist[i].strip()][val[0]]==1):
                 trends[i]+=1
    return trends
def AllMoviesbyCountry(country):
    items = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
    movies = pd.read_csv("ml-100k/dbpediamovies.csv",delimiter=";")
    specificmovies = movies[movies['country'].isin(country)]['name'].unique()
    uniqueids = items[items['SPARQLTitle'].isin(specificmovies)]['movieId'].unique()
    return uniqueids
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
def RelevantContextMovies(ratings,country):
    uniqueids = AllMoviesbyCountry(country)
    currentdate = datetime.now()
    currentday = currentdate.strftime("%A")
    weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
    weekend = ["Saturday","Sunday"]
    listmovies = list()
    if(currentday in weekdays):
      for i in range(ratings.shape[0]):
        if(ratings['movieId'][i] in uniqueids and ratings["rating"][i]==1) and calendar.day_name[ratings["timestamp"][i].weekday()] in weekdays:
            listmovies.append(ratings['movieId'][i])
    else : 
      for i in range(ratings.shape[0]):
        if(ratings['movieId'][i] in uniqueids and ratings["rating"][i]==1 and calendar.day_name[ratings["timestamp"][i].weekday()] in weekend):
            listmovies.append(ratings['movieId'][i])
    return listmovies
def MostSuccesfulMovies():
    movie = pd.read_csv("ml-100k/dbpediamovies.csv",delimiter=";")
    filmsenrichis = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
    succesfulltitles = list()
    succesfullids = list()
    for i in range(movie.shape[0]):
       if(movie['gross'][i] != nan and movie['budget'][i] != nan):
        if(movie['gross'][i] > 2*movie['budget'][i]):
            if(movie['name'][i] not in succesfulltitles):
                succesfulltitles.append(movie['name'][i])
    for i in range(filmsenrichis.shape[0]):
        if(filmsenrichis['SPARQLTitle'][i] in succesfulltitles):
            succesfullids.append(filmsenrichis['movieId'][i])
    return succesfullids
def ColdStartUsers():
    coldstart = list()
    list_users = pivot.index.unique().tolist()
    for i in range(pivot.shape[0]):
        if(len(ListRelevant(pivot,pivot.shape[1],i))<20):
            coldstart.append(list_users[i])
    return coldstart
def CorrespondingMovieIds(rev,list_movies):
    movieids = list()
    for i in range(len(rev)):
        movieids.append(list_movies[rev[i]])
    return movieids
def ModifyUser():
    coldusers = ColdStartUsers()
    success = MostSuccesfulMovies()
    for i in range(pivot.shape[0]):
        if(i+1 in coldusers):
            rev = ListRelevant(pivot,pivot.shape[1],i)
            newmovies = random.sample(success,20-len(rev))
            while(len(set(CorrespondingMovieIds(rev,list_movies)).intersection(newmovies))>0):
                newmovies = random.sample(success,20-len(rev))
            for j in newmovies:
                pivot.iloc[i,list_movies.index(j)]=1 
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
def EnsembleSamplesTraining(InputTr,InputTe,TargetTr,TargetTe):
    itemslist = np.loadtxt("Tests.txt")
    for i in range(itemslist.shape[0]):
        itembis = itemslist[i,:]
        count = 0
        for j in range(len(TargetTr)):
            if(TargetTr[j] in itembis):
                count+=1
        InputTrain = np.zeros((count,n_items))
        TargetTrain = np.zeros((count))
        k=0
        j=0       
        while j<len(InputTr) or k<count:
            if(TargetTr[j] in itembis):
                InputTrain[k]=InputTr[j,:]
                TargetTrain[k]=where(itembis,TargetTr[j])
                k+=1
            j+=1
        count = 0
        for j in range(len(TargetTe)):
            if(TargetTe[j] in itembis):
                count+=1
        InputTest = np.zeros((count,n_items))
        TargetTest = np.zeros((count))
        j=0 
        k=0
        count = 0
        while j<len(InputTe) or k<count:
            if(TargetTe[j] in itembis):
                InputTest[k]=InputTe[j,:]
                TargetTest[k]= where(itembis,TargetTe[j])
                k+=1
            j+=1
        print(itembis.shape[0])
        model = Sequential()
        model.add(Input(shape=InputTrain.shape[1]))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(itembis.shape[0],activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
        model.summary()
        history = model.fit(InputTrain,TargetTrain,validation_data=(InputTest,TargetTest),epochs=80,batch_size=150)
        model.save(format(i))
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
        results = model.evaluate(InputTest, TargetTest, batch_size=128)
        print("test loss, test acc:", results)
    return itemslist
def EnsembleSamplesTesting(nb):
    itemslist = np.loadtxt("Tests.txt")
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
    for i in range(itemlist.shape[0]):
        results= np.where(results==results[i],itemlist[results[i]],results)
    return results
        
               
            

"""Création des inputs et targets du RDN"""

ratings = pd.read_csv("ml-100k/filteredratings.csv",delimiter=";",parse_dates=['timestamp'])
pivot = ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating',fill_value=0)

n_users = pivot.index.unique().shape[0]
n_items = pivot.columns.unique().shape[0]
list_movies = pivot.columns.unique().tolist()
list_users = pivot.index.unique()
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

"""
InputTr = np.loadtxt("InputTr.txt")
TargetTr = np.loadtxt("TargetTr.txt")
InputTe = np.loadtxt("InputTe.txt")
TargetTe = np.loadtxt("TargetTe.txt")

"""np.savetxt("InputTe.txt",InputTe.astype(int),fmt='%d')
np.savetxt("TargetTe.txt",TargetTe.astype(int),fmt='%d')
np.savetxt("InputTr.txt",InputTr.astype(int),fmt='%d')
np.savetxt("TargetTr.txt",TargetTr.astype(int),fmt='%d')

model = Sequential()
model.add(Input(shape=InputTr.shape[1]))
model.add(Dense(300, activation='relu'))
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

print("Evaluate on test data")
results = model.evaluate(InputTe, TargetTe, batch_size=128)
print("test loss, test acc:", results)



model = load_model("ml-100k")




movies = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")

relevanttotal = Relevant(pivot)
testmovies = random.sample(relevanttotal,80)
testusers = list()
i=0
while i <pivot.shape[0]:
    relevants = ListRelevant(pivot,pivot.shape[1],i)
    if(len(set(relevants).intersection(testmovies))>0):
        testusers.append(i)
    i+=1
if(len(testusers)>30):
    testusers = random.sample(testusers,40)

randuser = random.randrange(1,InputTe.shape[0])
testUser = InputTe[randuser,:]
print(testUser.shape)
rev=ListRel(testUser)
rev.append(TargetTe[randuser].astype(int))
testUser = testUser.reshape(1,testUser.shape[0])
results = model.predict(testUser)
results = np.argsort(results.reshape(testUser.shape[1]))[::-1]
"""
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
 testUser = testUser.reshape(1,testUser.shape[0])
 results = EnsembleSamplesTesting(j)
 if(len(rev)!=0):
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
usertable = np.array(pivot.iloc[0,:],copy=True)
testUser = usertable.reshape(1,usertable.shape[0])
results = model.predict(testUser)
results = np.argsort(results.reshape(testUser.shape[1]))[::-1] 
n=96
i=1
recalls = []
precisions = []
rev = ListRelevant(pivot,pivot.shape[1],0)
while(i<n):
 rec = 0
 prec = 0
 hr=0
 temp =results[:i]
 for k in range(len(temp)):
    if  temp[k] in rev:
        hr+=1
 prec =  (hr)/i
 rec =  (hr)/len(rev)
 i+=5
 precisions.append(prec)
 recalls.append(rec)    
print(recalls)
print(precisions)
print(usertable[:100])
print(rev)"""