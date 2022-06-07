from copy import copy
import random
import pandas as pd
import numpy as np
import calendar
import datetime as dt
from keras.models import load_model
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
            relevants.append(matrix.columns.unique()[i])
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
              relevants.append(matrix.columns.unique()[j])
    return relevants   
def AllMoviesbyCountry(country):
    items = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
    movies = pd.read_csv("ml-100k/dbpediamovies.csv",delimiter=";")
    specificmovies = movies[movies['country'].isin(country)]['name'].unique()
    uniqueids = items[items['SPARQLTitle'].isin(specificmovies)]['movieId'].unique()
    return uniqueids
def MostRelevantMoviesbyContext(ratings):
    currentdate = dt.datetime.now()
    currentday = currentdate.strftime("%A")
    currentime = GetTimeDay(currentdate.hour)
    weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
    weekend = ["Saturday","Sunday"]
    listmovies = list()
    if(currentday in weekdays):
      for i in range(ratings.shape[0]):
        if(ratings["rating"][i]==1 and calendar.day_name[ratings["timestamp"][i].weekday()] in weekdays and GetTimeDay(ratings["timestamp"][i].to_pydatetime().hour)==currentime):
            if(ratings['movieId'][i] not in listmovies):
                listmovies.append(ratings['movieId'][i])
    else : 
      for i in range(ratings.shape[0]):
        if(ratings["rating"][i]==1 and calendar.day_name[ratings["timestamp"][i].weekday()] in weekend and GetTimeDay(ratings["timestamp"][i].to_pydatetime().hour)==currentime):
            if(ratings['movieId'][i] not in listmovies):
                listmovies.append(ratings['movieId'][i])
    return listmovies
def RelevantContextMovies(ratings,country):
    uniqueids = AllMoviesbyCountry(country)
    currentdate = dt.now()
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
    genreids = np.argsort(trends)[::-1]
    sortedgenres = list()
    for i in genreids:
        sortedgenres.append(genrelist[i].strip())
    print(trends) 
    return sortedgenres
def UserMostContextualMovies(pivot,listmovies):
    numberrel = np.zeros(pivot.shape[0])
    for i in range(pivot.shape[0]):
        numberrel[i] = len(set(ListRelevant(pivot,pivot.shape[1],i)).intersection(listmovies))
    return np.argsort(numberrel)[::-1]
def GetTimeDay(hour):
    if hour>=8 and hour<12:
        return "Morning"
    elif hour>=12 and hour <=13:
        return "Noon"
    elif hour>13 and hour<=16:
        return "Afternooon"
    elif hour>=17 and hour<=19:
        return "Evening"
    elif hour>=20 and hour<=00:
        return "Night"
    else: return "Late Night"
    
""""
movie = pd.read_csv("ml-100k/dbpediamovies.csv",delimiter=";")
items = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
user = UserMostMoviesbyCountry(pivot,['Spain'])
movielist = ListRelevant(pivot,pivot.shape[1],user)
for i in movielist:
    #print(movie[movie['name']==items.loc[movielist[i],'SPARQLTitle']]) 
    print(i)
    print(items[items['movieId']==i]['SPARQLTitle'])  
print(len(movielist))
print(user)"""
ratings = pd.read_csv("ml-100k/filteredratings.csv",delimiter=";",parse_dates=['timestamp'])
movie = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
pivot = ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating',fill_value=0)
model = load_model("ml-100k")
list_movies = pivot.columns.unique()

movieslist = list()
n=96
k=0
relevantmovies = MostRelevantMoviesbyContext(ratings)
testUserid= UserMostContextualMovies(pivot,relevantmovies)[0]
testUser = np.array(pivot.iloc[testUserid,:],copy=True)
totalprec = list()
totalrec = list()
recalls = list()
precisions = list()
testUser = testUser.reshape(1,testUser.shape[0])
results = model.predict(testUser)
results=results.reshape(testUser.shape[1])
copyresults = np.argsort(results)[::-1]
lol = np.sort(results)[::-1]
recommendedmovies = np.zeros(96)
i=0
while k<96:
    if(list_movies[copyresults[i]] in relevantmovies):
        recommendedmovies[k]=list_movies[copyresults[i]]
        k+=1
    i+=1
for i in recommendedmovies:
    movieslist.append(movie[movie['movieId']==i]['Title'].unique().tolist())
print(movieslist)