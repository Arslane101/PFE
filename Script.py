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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
"""Chargement du Dataset (le préfiltrage se fera dans cette partie) et Transformation en relevant et non-relevant"""
    
def ChargerDataset(ratings,th):
    for i in range(ratings.shape[0]):
        if pd.isnull(ratings['rating'][i]) :
            ratings.loc[i,'rating'] = int(0)
        if int(ratings['rating'][i]) >= int(th):
            ratings.loc[i,'rating']= int(1)
        else: ratings.loc[i,'rating']=int(0) 
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
def MostRelevantMoviesbyContext(ratings):
    currentdate = datetime.now()
    popularmovies = MostRatedMovies(ratings)
    currentday = currentdate.strftime("%A")
    weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
    weekend = ["Saturday","Sunday"]
    listmovies = list()
    if(currentday in weekdays):
        for i in range(ratings.shape[0]):
          if pd.isnull(ratings["review_date"][i]) == False and calendar.day_name[ratings["review_date"][i].weekday()] in weekdays and ratings['rating'][i]>=3:
            if(ratings['movieId'][i] in popularmovies and ratings['movieId'][i] not in listmovies):
                listmovies.append(ratings['movieId'][i])
    else : 
        for i in range(ratings.shape[0]):
          if pd.isnull(ratings["review_date"][i]) == False and calendar.day_name[ratings["review_date"][i].weekday()] in weekend and ratings['rating'][i]>=3:
            if(ratings['movieId'][i] in popularmovies and ratings['movieId'][i] not in listmovies):
                listmovies.append(ratings['movieId'][i])
    return listmovies
def ColdStartUsers():
    coldstart = list()
    list_users = ratings.userId.unique()
    for user in list_users:
        sum = ratings[ratings["userId"]==user][ratings["rating"] == 4].shape[0] + ratings[ratings["userId"]==user][ratings["rating"] == 5].shape[0] + ratings[ratings["userId"]==user][pd.isnull(ratings["rating"])].shape[0] 
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
    subset = list(range(0,n_items))
    for i in range(nb):
        sub = random.sample(subset,int(n_items/nb))
        subset = list(set(subset)-set(sub))
        subsets.append(sub)
    return subsets
def where(arr,nb):
    for i in range(len(arr)):
        if(arr[i]==nb):
            return i
def EnsembleSamplesTraining():
  itemslist = np.loadtxt("LOD/Subsets.txt")
  i=0
  nbrel= ratings[ratings["rating"] == 1.0].shape[0]
  k=0
  Input = np.zeros((nbrel,n_items),dtype=np.int8)
  Target = np.zeros((nbrel),dtype=np.int16)
  for i in range(pivot.shape[0]):
    for j in  ListRelevant(pivot,n_items,i):
        Input[k] = np.array(pivot.iloc[i,:],copy=True)
        Input[k,j]=0
        Target[k]=j
        k+=1
  print(Input.shape) 
  print(Target.shape)     
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
    itemslist = np.loadtxt("LOD/Subsets.txt")
    itemlist = np.concatenate(itemslist)
    values = list()
    for i in range(itemslist.shape[0]):
        model = load_model("SentimentsLOD/"+str(i))
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
  itembis = np.loadtxt("LOD/Subsets.txt")
  i=3
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
  model.fit(InputTrain,TargetTrain,validation_data=(InputTest,TargetTest),epochs=80,batch_size=150)
  model.save(format(i))
def MitigateColdStart():
    coldstartusers = ColdStartUsers()
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
def MostRatedMovies(ratings):
    ratings = ratings.groupby(['movieId'])[['rating']].mean()
    ratings = ratings[ratings["rating"] >= 3]
    return ratings.index.unique().tolist()
def FilterContext(results,movies):
    movie = list()
    for mov in movies:
        movie.append(where(list_movies,mov))
    result = list()
    for elt in results:
        if( elt in movie):
            result.append(elt)
    return result
def EnsembleSamples(nb):
    itemslist = np.loadtxt("LOD/Subsets.txt")
    itemlist = np.concatenate(itemslist)
    values = list()
    for i in range(itemslist.shape[0]):
        model = load_model("SentimentsLOD/"+str(i))
        testUser = np.array(pivot.iloc[nb,:],copy=True)
        testUser = testUser.reshape(1,testUser.shape[0])
        results = model.predict(testUser)
        values.append(results)
    results = np.concatenate(np.asarray(values))
    results = results.reshape(itemlist.shape[0])
    result = pd.DataFrame(columns=['movieId','probability'])
    for i in range(results.shape[0]):
        result.loc[len(result.index)]=[list_movies[int(itemlist[i])],results[i]]
    return result
def contentbased(user,movies,ratings):
    allmovies = ratings.movieId.unique()
    allmovies = list(set(allmovies)-set(movies.rotten_tomatoes_link))
    tf = TfidfVectorizer(stop_words='english')
    tfidf_matrix_item = tf.fit_transform(movies['movie_info'])
    userRate = ratings[ratings['userId'] == user ]
    relRating = userRate[userRate['rating'] == 1]
    userM = movies[movies['rotten_tomatoes_link'].isin(userRate['movieId'])]            
    featureMat = pd.DataFrame(tfidf_matrix_item.todense(),
                                    columns=tf.get_feature_names_out(),
                                    index=movies.rotten_tomatoes_link)
    featureMatU = featureMat[featureMat.index.isin(userM['rotten_tomatoes_link'])]
    featureMatU = (pd.DataFrame((featureMatU.mean()),
                                    columns=['similarity'])).transpose()       
    cosine_sim = cosine_similarity(featureMatU, tfidf_matrix_item)
    cosine_sim_df = pd.DataFrame(columns=['movieId','similarity'])
    cosine_sim = cosine_sim.T
    for i in range(cosine_sim.shape[0]):
            cosine_sim_df.loc[len(cosine_sim_df.index)]= [movies['rotten_tomatoes_link'][i],cosine_sim[i][0]]
    for movie in allmovies:
        cosine_sim_df.loc[len(cosine_sim_df.index)]= [movie,0]
    return relRating.movieId.unique(),cosine_sim_df
"""Création des inputs et targets du RDN"""
ratings = pd.read_csv("lodratings.csv",delimiter=";",parse_dates=['review_date'],infer_datetime_format=True)
ratings2 = pd.read_csv("BinarizedratingsLOD.csv",delimiter=";",parse_dates=['review_date'],infer_datetime_format=True)
context = MostRelevantMoviesbyContext(ratings)
print(len(context))
pivot = ratings2.pivot_table(index=['userId'],columns=['movieId'],values='rating',fill_value=0)
n_users = pivot.index.unique().shape[0]
n_items = pivot.columns.unique().shape[0]
list_movies = pivot.columns.unique().tolist()
list_users = pivot.index.unique().tolist()
print(n_users)
print(n_items)
def Hybrid(alpha,nb):
    cbresults = contentbased(list_users[nb],movies,ratings)[1]
    cfresults = EnsembleSamples(nb)
    cbresults = cbresults.sort_values(by=['movieId'])
    cbresults = cbresults.reset_index()
    cbresults['similarity'] = cbresults['similarity'].apply(lambda x:  x/int(round(cbresults['similarity'].sum())))
    cfresults = cfresults.sort_values(by=['movieId'])
    cfresults = cfresults.reset_index()
    cfresults['probability']= cfresults['probability'].apply(lambda x:  x/int(round(cfresults['probability'].sum())))
    hybrid = pd.DataFrame(columns=['movieId','probability'])
    for i in range(cfresults.shape[0]):
        x = alpha*cfresults['probability'][i]+(1-alpha)*cbresults['similarity'][i]
        hybrid.loc[len(hybrid.index)]=[cfresults['movieId'][i],x]
    return hybrid


j=0
n=96
totalprec = list()
totalrec = list()
totalf = list()
for j in range(10):
 print(j)
 recalls = list()
 precisions = list()
 recalls.append(j)
 precisions.append(j)
 i=1 
 testUser = np.array(pivot.iloc[j,:],copy=True)
 rev  = ListSpecRel(testUser)
 if(len(rev)!=0):
  results = EnsembleSamplesTesting(j)
  results = FilterContext(results,context)
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