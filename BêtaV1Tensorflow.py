from keras.engine.input_layer import Input
import random
from datetime import datetime,timedelta
import pandas as pd
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from sklearn.metrics import jaccard_score
import tensorflow as tf
import matplotlib.pyplot as plt
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
def BuildProfile(ratings,ind):
   with open("ml-100k/genres.txt") as f:
    genres = f.readlines()
   movies = pd.read_csv("ml-100k/item.csv",delimiter=";")
   size = len(genres)
   profile = np.zeros(size)
   userratings= ratings.loc[ratings['userId']==ind]
   ratedmovies = userratings['movieId'].unique()
   for i in ratedmovies:
       for j in range(len(genres)):
           genre = genres[j].strip()
           if(movies.loc[i,genre]==1):
               profile[j]=1
   return profile
def UserMostMoviesbyCountry(pivot,country):
    items = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
    movies = pd.read_csv("ml-100k/dbpediamovies.csv",delimiter=";")
    specificmovies = movies[movies['country'].isin(country)]['name'].unique()
    uniqueids = items[items['name'].isin(specificmovies)]['movieId'].unique()
    maxmovies= len(set(uniqueids).intersection(relevant(pivot,pivot.shape[1],0)))  
    maxuser = 0
    maxusers = list()
    i=1
    for i in range(pivot.shape[0]):
        relevant = ListRelevant(pivot,pivot.shape[1],i)
        if(len(set(uniqueids).intersection(relevant(pivot,pivot.shape[1],i)))>maxmovies):
            maxmovies = len(set(uniqueids).intersection(relevant(pivot,pivot.shape[1],i)))
            maxuser = i
    i=0
    for i in range(pivot.shape[0]):
        if(len(len(set(uniqueids).intersection(relevant(pivot,pivot.shape[1],i)))==maxmovies)):
            maxusers.append(i)
    return maxuser,maxusers


"""Création des inputs et targets du RDN"""
ratings = pd.read_csv("ml-100k/filteredratings.csv",delimiter=";",parse_dates=['timestamp'])

pivot = ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating',fill_value=0)
"""
ratings = ContextualisationDataset(ratings,4,[],['United States'])
moviesr = Relevant(pivot)
items = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
movies = pd.read_csv("ml-100k/dbpediamovies.csv",delimiter=";")
for movie in moviesr:
    title = items.loc[items['movieId']==movie,'SPARQLTitle']
    nationality = movies.loc[movies['name']==title,'country'].values.tolist()
    print(nationality)
n_users = pivot.index.unique().shape[0]
n_items = pivot.columns.unique().shape[0]
list_movies = pivot.columns.unique()
list_users = pivot.index.unique()"""

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

model = Sequential()
model.add(Input(shape=InputTr.shape[1]))
model.add(Dense(300, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(175, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(75, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(InputTr.shape[1],activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(InputTr,TargetTr,validation_data=(InputTe,TargetTe),epochs=200,batch_size=300)
model.save("ml-100k")

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

InputTr = np.loadtxt("InputTr.txt")
TargetTr = np.loadtxt("TargetTr.txt")
InputTe = np.loadtxt("InputTe.txt")
TargetTe = np.loadtxt("TargetTe.txt")
model = load_model("ml-100k")
movies = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
"""
relevanttotal = Relevant(pivot)
testmovies = random.sample(relevanttotal,80)
testusers = list()
i=0
while i <pivot.shape[0]:
    relevants = ListRelevant(pivot,pivot.shape[1],i)
    if(len(relevants)>40 and len(relevants)<100):
        testusers.append(i)
    i+=1
if(len(testusers)>30):
    testusers = random.sample(testusers,25)

        
"""
"""
randuser = random.randrange(1,InputTe.shape[0])
testUser = InputTe[randuser,:]
print(testUser.shape)
rev=ListRel(testUser)
rev.append(TargetTe[randuser].astype(int))
testUser = testUser.reshape(1,testUser.shape[0])
results = model.predict(testUser)
results = np.argsort(results.reshape(testUser.shape[1]))[::-1]"""
"""
n=96
i=1
recalls = []
precisions = []

while(i<n):
 totalrec = 0
 totalprec = 0
 for j in testusers:
    testUser = np.array(pivot.iloc[j,:],copy=True)
    rev  = ListRelevant(pivot,pivot.shape[1],j)
    testUser = testUser.reshape(1,testUser.shape[0])
    results = model.predict(testUser)
    results = np.argsort(results.reshape(testUser.shape[1]))[::-1]   
    hr=0
    temp =results[:i]
    for k in range(len(temp)):
         if  temp[k] in rev:
          hr+=1
    totalprec = totalprec + (hr)/i
    totalrec = totalrec + (hr)/len(rev)
 i+=5
 precisions.append(totalprec/len(testusers))
 recalls.append(totalrec/len(testusers))

print(precisions)
print("_______________")
print(recalls)"""
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
print(rev)