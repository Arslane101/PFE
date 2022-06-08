
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
    moviecopy = pd.read_csv("Content-based/filmsenrichis.csv",delimiter=";")
    testUser = ratings[ratings["userId"]==userId]
    usermovies = movies[movies['movieId'].isin(testUser["movieId"].tolist())]
    usermovies = usermovies.reset_index(drop=True)
    usermovies  =usermovies.drop('movieId',1)
    userprofile = usermovies.transpose().dot(testUser['rating'].reset_index(drop=True))
    genreTable = movies.set_index(movies['movieId'])
    genreTable = genreTable.drop('movieId', 1)
    recommendation = ((genreTable*userprofile).sum(axis=1))/(userprofile.sum())
    sorted = recommendation.sort_values(ascending=False)
    listmovies = moviecopy.loc[moviecopy['movieId'].isin(sorted.head(20).keys())]
    return listmovies
def ContentBasedLOD(userId):
    movies = pd.read_csv("Content-based/moviesgenrescountries.csv",delimiter=";")
    testUser = ratings[ratings["userId"]==userId]
    usermovies = movies[movies['movieId'].isin(testUser["movieId"].tolist())]
    usermovies = usermovies.reset_index(drop=True)
    usermovies  =usermovies.drop('movieId',1)
    userprofile = usermovies.transpose().dot(testUser['rating'].reset_index(drop=True))
    genreTable = movies.set_index(movies['movieId'])
    genreTable = genreTable.drop('movieId', 1)
    recommendation = ((genreTable*userprofile).sum(axis=1))/(userprofile.sum())
    sorted = recommendation.sort_values(ascending=False)
    listmovies = moviecopy.loc[moviecopy['movieId'].isin(sorted.head(96).keys())]
    return listmovies
ratings = pd.read_csv("Content-based/specificratings.csv",delimiter=";",parse_dates=["timestamp"])
moviecopy = pd.read_csv("Content-based/filmsenrichis.csv",delimiter=";")
ratings = ratings.drop('timestamp',1)
n=96
totalprec = list()
totalrec = list()
list_users = random.sample(ratings['userId'].unique().tolist(),20)
j=0
for j in range(len(list_users)) :
 rev = ListRelevant(list_users[j],3)
 hr=0
 recalls = list()
 precisions = list()
 recalls.append(list_users[j])
 precisions.append(list_users[j])
 recommended = ContentBasedLOD(list_users[j])
 recommended = recommended.reset_index(drop=True)
 if(len(rev)!=0):
  recalls.append(len(rev))
  precisions.append(len(rev)) 
  temp= recommended.head(10)
  for k in range(temp.shape[0]):
     if(temp['movieId'][k] in rev):
        hr+=1
  prec = (hr)/10
  rec =  (hr)/len(rev)
  precisions.append(prec)
  recalls.append(rec)
  j+=1
  print(j)
 totalprec.append(np.asarray(precisions))
 totalrec.append(np.asarray(recalls))
np.savetxt("AllPrecisions.txt", np.vstack(totalprec).astype(float),fmt='%.2f')
np.savetxt("AllRecalls.txt",np.vstack(totalrec).astype(float),fmt='%.2f')