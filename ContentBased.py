from traceback import print_tb
from unittest import result
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy.lib.npyio import itertools
from sklearn.model_selection import train_test_split


def contentbased(user,movies,ratings):
      allmovies = ratings.movieId.unique()
      allmovies = list(set(allmovies)-set(movies.rotten_tomatoes_link))
      tf = TfidfVectorizer(stop_words='english')
      tfidf_matrix_item = tf.fit_transform(movies['movie_info'])
      userRate = ratings[ratings['userId'] == user]
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
    
ratings = pd.read_csv("BinarizedSentimentLODRatings.csv",delimiter=";",parse_dates=['review_date'],infer_datetime_format=True)
#ratings.dropna(axis=0, subset=['rating'], inplace=True)
#ratings.drop_duplicates(subset=['movieId', 'userId'], inplace=True)
movies = pd.read_csv('movies.csv', delimiter=';')
def FilterContext2(results,movies):
    result = list()
    for i in range(150):
        if(results['movieId'][i] in movies):
            result.append(results['movieId'][i])
    return result


j=0
n=96
totalprec = list()
totalrec = list()
totalf = list()
for j in range(100):
 print(j)
 recalls = list()
 precisions = list()
 recalls.append(j)
 precisions.append(j)
 i=1
 rev,results = contentbased(ratings.userId.unique()[j],movies,ratings)
 results = results.sort_values(by=['similarity'],ascending=False)
 results = results.reset_index()
 if(len(rev)>=10):
  while(i<n):   
    hr=0
    temp = results.loc[0:i-1,:] 
    for k in range(len(temp)):
         if  temp['movieId'][k] in rev:
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
