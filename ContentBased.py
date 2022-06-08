from cgi import test
import imp
from cv2 import sort
from matplotlib.pyplot import axis
import pandas as pd
import matplotlib as plt
import numpy as np
def AllMoviesbyCountry(country):
    items = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
    movies = pd.read_csv("ml-100k/dbpediamovies.csv",delimiter=";")
    specificmovies = movies[movies['country'].isin(country)]['name'].unique()
    uniqueids = items[items['SPARQLTitle'].isin(specificmovies)]['movieId'].unique()
    return uniqueids
def ContentBasedNoLOD(userId):
    ratings = pd.read_csv("Content-based/specificratings.csv",delimiter=";",parse_dates=["timestamp"])
    ratings = ratings.drop('timestamp',1)
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
    ratings = pd.read_csv("Content-based/specificratings.csv",delimiter=";",parse_dates=["timestamp"])
    ratings = ratings.drop('timestamp',1)
    movies = pd.read_csv("Content-based/moviesgenrescountries.csv",delimiter=";")
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
