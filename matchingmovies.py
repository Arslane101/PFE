from unittest import result
import pandas as pd
import numpy as np
import jellyfish 

movies = pd.read_csv("ml-100k/dbpediamovies.csv",delimiter=";")
results = pd.read_csv("ml-100k/resultssparql.csv",delimiter=";")
newdata = pd.merge(movies,results[["name","budget"]],on='name')
newdata.to_csv("results.csv")