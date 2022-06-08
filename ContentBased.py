from cgi import test
import imp
import pandas as pd
import matplotlib as plt
import numpy as np
def ListRel(array):
    relevants = []
    for i in range(len(array)):
        if(array[i]>=4):
            relevants.append(i)
    return relevants 

ratings = pd.read_csv("Content-based/specificratings.csv",delimiter=";",parse_dates=["timestamp"])
ratings = ratings.drop('timestamp',1)
movies = pd.read_csv("Content-based/movies.csv",delimiter=";")
testUser = ratings[ratings["userId"]==1]
usermovies = movies[movies['movieId'].isin(testUser["movieId"].tolist())]
usermovies = usermovies.reset_index(drop=True)
usermovies  =usermovies.drop('movieId',1)
print(pd.DataFrame(testUser['rating']))
transpose = usermovies.transpose()
print(transpose.head())
userprofile = transpose.dot(pd.DataFrame(testUser['rating']))
print(userprofile)