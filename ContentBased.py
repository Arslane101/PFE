
from cgi import test
import random
import numpy as np
import pandas as pd


def AllMoviesbyCountry(country):
    items = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
    movies = pd.read_csv("ml-100k/dbpediamovies.csv",delimiter=";")
    specificmovies = movies[movies['country'].isin(country)]['name'].unique()
    uniqueids = items[items['SPARQLTitle'].isin(specificmovies)]['movieId'].unique()
    return uniqueids
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
    movies = pd.read_csv("Content-based/moviesgenrescountries.csv",delimiter=";")
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


ratings = pd.read_csv("Content-based/specificratings.csv",delimiter=";",parse_dates=["timestamp"])
moviecopy = pd.read_csv("Content-based/filmsenrichis.csv",delimiter=";")
ratings = ratings.drop('timestamp',1)
n=96
totalprec = list()
totalrec = list()
list_users = ratings['userId'].unique().tolist()
for j in list_users :
 rev = ListRelevant(j,3)
 print(j)
 if(len(rev)>15):
  recalls = list()
  precisions = list()
  recalls.append(j)
  precisions.append(j)
  recommended = ContentBasedNoLOD(j)
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
np.savetxt("AllRecalls.txt",np.vstack(totalrec).astype(float),fmt='%.2f')