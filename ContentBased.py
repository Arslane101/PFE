
from cgi import test
import random
import numpy as np
import pandas as pd

def AllMoviesbyDirector(director):
    items = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
    movies = pd.read_csv("ml-100k/dbpediamovies.csv",delimiter=";")
    specificmovies = movies[movies['director'].isin(director)]['name'].unique()
    uniqueids = items[items['SPARQLTitle'].isin(specificmovies)]['movieId'].unique()
    return uniqueids
def AllMoviesbyCountry(country):
    specificmovies = movies[movies['country'].isin(country)]['name'].unique()
    uniqueids = items[items['SPARQLTitle'].isin(specificmovies)]['movieId'].unique()
    return uniqueids
def EnrichissementparPays():
    movies = pd.read_csv("Content-based/moviesgenrescountries.csv",delimiter=";")
    specificmovies = pd.read_csv("ml-100k/dbpediamovies.csv",delimiter=";")
    directors = specificmovies['director'].unique().tolist()
    for i in directors:
        movies.insert(movies.shape[1],i,0)
    i=0
    for j in directors:
        for i in range(movies.shape[0]):
            ids = AllMoviesbyDirector([j])
            if(movies["movieId"][i] in ids):
                movies[j][i]=1
    movies.to_csv("euh.csv")
    return movies    
def ListRelevant(userId,th):
    ratings = pd.read_csv("Content-based/specificratings.csv",delimiter=";",parse_dates=["timestamp"])
    testUser = ratings[ratings["userId"]==userId]
    testUser = testUser.reset_index(drop=True)
    relevant = list()
    for i in range(testUser.shape[0]):
        if(testUser['rating'][i]>=th):
            relevant.append(testUser['movieId'][i])
    return relevant
def ContentBasedNoLOD(userId):
    movies = pd.read_csv("Content-based/movies.csv",delimiter=";")
    testUser = ratings[ratings["userId"]==userId]
    testUser = testUser.sort_values(by=['movieId'])
    usermovies = movies[movies['movieId'].isin(testUser["movieId"].tolist())]
    usermovies = usermovies.reset_index(drop=True)
    userprofile = usermovies.transpose().dot(testUser['rating'].reset_index(drop=True))
    genreTable = movies.set_index(movies['movieId'])
    genreTable = genreTable.drop('movieId', 1)
    recommendation = ((genreTable*userprofile).sum(axis=1))/(userprofile.sum())
    sorted = recommendation.sort_values(ascending=False)
    return sorted.head(96).keys()
def ContentBasedLOD(userId):
    movies = EnrichissementparPays()
    print(movies.head(15))
    testUser = ratings[ratings["userId"]==userId]
    testUser = testUser.sort_values(by=['movieId'])
    usermovies = movies[movies['movieId'].isin(testUser["movieId"].tolist())]
    usermovies = usermovies.reset_index(drop=True)
    usermovies  =usermovies.drop('movieId',1)
    userprofile = usermovies.transpose().dot(testUser['rating'].reset_index(drop=True))
    genreTable = movies.set_index(movies['movieId'])
    genreTable = genreTable.drop('movieId', 1)
    recommendation = ((genreTable*userprofile).sum(axis=1))/(userprofile.sum())
    sorted = recommendation.sort_values(ascending=False)
    return sorted.head(96).keys()
items = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
movies = pd.read_csv("ml-100k/dbpediamovies.csv",delimiter=";")
ratings = pd.read_csv("Content-based/specificratings.csv",delimiter=";",parse_dates=["timestamp"])
moviecopy = pd.read_csv("Content-based/filmsenrichis.csv",delimiter=";")
EnrichissementparPays()
"""ratings = ratings.drop('timestamp',1)
n=96
totalprec = list()
totalrec = list()
list_users = ratings['userId'].unique().tolist()
print(len(list_users))
for j in list_users :
 rev = ListRelevant(j,3)
 print(j)
 if(len(rev)>15):
  recalls = list()
  precisions = list()
  recalls.append(j)
  precisions.append(j)
  recommended = ContentBasedLOD(j)
  recalls.append(len(rev))
  precisions.append(len(rev))
  i=10
  while(i<50): 
   hr=0
   temp= recommended[:i]
   for k in range(len(temp)):
     if(temp[k] in rev):
        hr+=1
   prec = (hr)/i
   rec =  (hr)/len(rev)
   precisions.append(prec)
   recalls.append(rec)
   i+=10
  totalprec.append(np.asarray(precisions))
  totalrec.append(np.asarray(recalls))
np.savetxt("AllPrecisions.txt", np.vstack(totalprec).astype(float),fmt='%.2f')
np.savetxt("AllRecalls.txt",np.vstack(totalrec).astype(float),fmt='%.2f')"""