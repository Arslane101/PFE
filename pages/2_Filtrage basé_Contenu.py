
import itertools
from unittest import result
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity




def ContextFiltering(results):
  return results
@st.experimental_singleton
def Evaluations():
  sentiments = st.session_state["sentiments"]
  user = list_users[int(st.session_state['select1'])-1]
  if(sentiments==True):
      results = contentbased(ratings,movies,user)
  else : 
    results = Evaluations(sentimentratings,movies,user)
  return results
def MovieListCB():
    num = int(st.session_state['numrec3'])
    user = list_users[int(st.session_state['select1'])-1]
    results = contentbased(ratings,movies,user)[1]
    results = results.sort_values(by=['similarity'],ascending=False)
    results = results.reset_index()
    temp = results.loc[0:num-1,:]
    movieslist = list()
    for i in range(len(temp)):
      movieslist.append(movies[movies['rotten_tomatoes_link']==temp['movieId'][i]]['movie_title'].unique())
    return movieslist
@st.experimental_singleton
def contentbased(ratings,movies,user):
        movies.dropna(subset=['movie_info'], inplace=True)
        movies = movies.reset_index()
        allmovies = ratings.movieId.unique()
        allmovies = list(set(allmovies)-set(movies.rotten_tomatoes_link))
        tf = TfidfVectorizer(stop_words='english')
        tfidf_matrix_item = tf.fit_transform(movies['movie_info'])
        userRate = ratings[ratings['userId'] == user]
        relRating = userRate[userRate['rating'] == 1 ]
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
def PlotsCB():
    user = list_users[int(st.session_state['select1'])-1]
    rev,results = contentbased(ratings,movies,user)
    copyresults = results
    results = results.sort_values(by=['similarity'],ascending=False)
    results = results.reset_index()
    n=96
    i=1
    recalls = []
    precisions = []
    while(i<n):
      rec = 0
      prec = 0
      hr=0
      temp = results.loc[0:i-1,:] 
      for k in range(len(temp)):
        if temp['movieId'][k] in rev:
         hr+=1
      prec =  (hr)/i
      if(len(rev)!=0):
        rec =  (hr)/len(rev)
      else: rec=0
      i+=1
      precisions.append(prec)
      recalls.append(rec)
    mesures = np.column_stack((precisions,recalls))       
    return copyresults,mesures,len(rev)

st.set_page_config(
    page_title="Filtrage basé Contenu",
    page_icon="👋",
    layout="wide",
)
st.sidebar.write("Paramètres Filtrage Collaboratif")
st.sidebar.checkbox('Analyse de Sentiments',key='sentiments')
ratings = st.session_state.ratings
pivot = st.session_state.pivot
pivotsentiment = st.session_state.pivotsentiment
sentimentratings = st.session_state.sentimentratings
movies = st.session_state.movies

list_users = pivot.index.unique()
list_movies = pivot.columns.unique()
number1  = st.number_input("Saisissez votre numéro d'utilisateur",1,8690,key='select1')
numberec2 = st.slider("Séléctionnez le nombre de recommandations à afficher",1,96,on_change=MovieListCB,key='numrec3')
movie = MovieListCB()
for i in movie:
   st.text(i)
Results = PlotsCB()
if('cbresults' not in st.session_state):
    st.session_state['cbresults']= Results[0]
if(Results[2]!=0):
   results = pd.DataFrame(Results[1],columns=['Précision','Rappel'])
   st.line_chart(data=results)
else: st.write("Cet utilisateur n'a aucun film pertinent")

    



