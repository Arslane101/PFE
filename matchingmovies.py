import pandas as pd
import numpy as np
import jellyfish 
movies = pd.read_csv("ml-100k/ratings.csv",delimiter=";")
filmsenrichis =pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
empty = filmsenrichis[filmsenrichis['SPARQLTitle'].isnull]
print(empty.shape)
for i in range(empty.shape[0]):
    print(empty["movieId"][i])
    movies.drop(movies.loc[movies['movieId']==empty["movieId"][i]].index)
movies.to_csv("done.csv")