
import itertools
from unittest import result
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Accueil import  ratings,sentimentratings,pivot,pivotsentiment,movies



def ContextFiltering(results):
  return results
def Evaluations():
  sentiment = st.session_state["lod"]
  if(sentiment==True):
      results = contentbased(ratings,movies)
  else : 
    results = Evaluations(sentimentratings,movies)
  return results
@st.experimental_memo
def MovieListCB():
    num = st.session_state['numrec3']
    results = contentbased(movies)[1]
    results = results.sort_values(by=['similarity'],ascending=False)
    results = results.reset_index()
    temp = results.loc[0:num-1,:]
    movieslist = list()
    for i in range(len(temp)):
      movieslist.append(temp['movieId'][i])
    return movieslist
@st.experimental_singleton
def contentbased(ratings,movies):
        user = list_users[int(st.session_state['select1'])-1]
        movies.dropna(subset=['movie_info'], inplace=True)
        movies = movies.reset_index()

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
                cosine_sim_df.loc[len(cosine_sim_df.index)]= [movies['rotten_tomatoes_link'][i],cosine_sim[i]]
        return relRating.movieId.unique(),cosine_sim_df
def PlotsCB():
    rev,results = contentbased(movies)
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
    return mesures,len(rev)

st.set_page_config(
    page_title="Comparaison des Différentes Approches",
    page_icon="👋",
    layout="wide",
)
st.sidebar.write("Paramètres Filtrage Collaboratif")
lod = st.sidebar.checkbox('Linked Open Data',key='lod')

ratings,sentimentratings,pivot,pivotsentiment,movies = LoadData()
list_users = pivot.index.unique()
list_movies = pivot.columns.unique()
number1  = st.number_input("Saisissez votre numéro d'utilisateur",1,8690,key='select1')
numberec2 = st.slider("Séléctionnez le nombre de recommandations à afficher",1,96,key='numrec3')
movie = MovieListCB()
for i in movie:
   st.text(i)
Results = PlotsCB()
if(Results[1]!=0):
   results = pd.DataFrame(Results[0],columns=['Précision','Rappel'])
   st.line_chart(data=results)
else: st.write("Cet utilisateur n'a aucun film pertinent")

    



