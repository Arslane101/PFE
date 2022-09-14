import calendar
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import keras
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def MostRatedMovies(ratings):
    ratings = ratings.groupby(['movieId'])[['rating']].mean()
    ratings = ratings[ratings["rating"] >= 3]
    return ratings.index.unique().tolist()
def ListRelevant(matrix,n_items,ind):
    relevants = []
    for i in range(n_items):
        if(matrix.iloc[ind,i]==1):
            relevants.append(i)
    return relevants   
def ListRel(array):
    relevants = []
    for i in range(len(array)):
        if(array[i]==1):
            relevants.append(i)
    return relevants 
def Relevant(matrix):
    relevants = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if(matrix.iloc[i,j]==1) and j not in relevants:
              relevants.append(j)
    return relevants   
def where(arr,nb):
    for i in range(len(arr)):
        if(arr[i]==nb):
            return i
@st.experimental_memo
def ContextFiltering(results,movies):
    movie = list()
    for mov in movies:
        movie.append(where(list_movies,mov))
    result = list()
    for elt in results:
        if( elt in movie):
            result.append(elt)
    return result
def Evaluations():
  lod = st.session_state["lod"]
  sentiment =st.session_state["sentiment"]
  context =  st.session_state["context"]
  nb = st.session_state['select']
  nb = int(nb)-1
  if(lod==True and sentiment==True):
      results = EnsembleSamplesTesting("LOD/Subsets.txt","SentimentsLOD/",nb,pivotsentimentlod)
      if(context):
        cont = MostRelevantMoviesbyContext(clsentimentlodratings)
        results = ContextFiltering(results,cont)
  elif(lod==True and  sentiment==False): 
     results = EnsembleSamplesTesting("LOD/Subsets.txt","LOD/",nb,pivotlod)
     if(context):
        cont = MostRelevantMoviesbyContext(cllodratings)
        results = ContextFiltering(results,cont)
  elif(sentiment==True and lod==False):
        results = EnsembleSamplesTesting("Classic/Subsets.txt","Sentiments/",nb,pivotsentiment)
        if(context):
         cont = MostRelevantMoviesbyContext(clsentimentratings)
         results = ContextFiltering(results,cont)
  else : 
    results = EnsembleSamplesTesting("Classic/Subsets.txt","Classic/",nb,pivot)
    if(context):
        cont = MostRelevantMoviesbyContext(clratings)
        results = ContextFiltering(results,cont)
  return results
@st.experimental_memo
def EnsembleSamplesTesting(subsets,mode,nb,pivot):
        itemslist = np.loadtxt(subsets)
        itemlist = np.concatenate(itemslist)
        values = list()
        for i in range(itemslist.shape[0]):
         model = load_model(mode+str(i))
         testUser = np.array(pivot.iloc[nb,:],copy=True)
         testUser = testUser.reshape(1,testUser.shape[0])
         results = model.predict(testUser)
         values.append(results)
        results = np.concatenate(np.asarray(values))
        results = np.argsort(results.reshape(itemlist.shape[0]))[::-1] 
        for i in range(results.shape[0]):
         results[i] = int(itemlist[results[i]]) 
        
        return results
@st.experimental_memo
def MostRelevantMoviesbyContext(ratings):
    currentdate = datetime.now()
    popularmovies = MostRatedMovies(ratings)
    currentday = currentdate.strftime("%A")
    weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
    weekend = ["Saturday","Sunday"]
    listmovies = list()
    if(currentday in weekdays):
        for i in range(ratings.shape[0]):
          if pd.isnull(ratings["review_date"][i]) == False and calendar.day_name[ratings["review_date"][i].weekday()] in weekdays and ratings['rating'][i]>=3:
            if(ratings['movieId'][i] in popularmovies and ratings['movieId'][i] not in listmovies):
                listmovies.append(ratings['movieId'][i])
    else : 
        for i in range(ratings.shape[0]):
          if pd.isnull(ratings["review_date"][i]) == False and calendar.day_name[ratings["review_date"][i].weekday()] in weekend and ratings['rating'][i]>=3:
            if(ratings['movieId'][i] in popularmovies and ratings['movieId'][i] not in listmovies):
                listmovies.append(ratings['movieId'][i])
    return listmovies 
def MovieList():
    num = st.session_state['numrec2']
    results = Evaluations()
    temp = results[:num]
    movieslist = list()
    for i in temp:
      movieslist.append(list_movies[i])
    return movieslist
def PlotsColab():
    lod = st.session_state["lod"]
    sentiment =st.session_state["sentiment"]
    results = Evaluations()
    n=96
    i=1
    recalls = []
    precisions = []
    if(lod==True and sentiment==True):
      rev = ListRelevant(pivotsentimentlod,pivotsentimentlod.shape[1],int(st.session_state['select'])-1)
    elif(lod==True and  sentiment==False):
      rev = ListRelevant(pivotlod,pivotlod.shape[1],int(st.session_state['select'])-1)
    elif(sentiment==True and lod==False):
      rev = ListRelevant(pivotsentiment,pivotsentiment.shape[1],int(st.session_state['select'])-1)
    else : rev = ListRelevant(pivot,pivot.shape[1],int(st.session_state['select'])-1)  
    while(i<n):
      rec = 0
      prec = 0
      hr=0
      temp = results[:i]
      for k in range(len(temp)):
        if  temp[k] in rev:
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
    page_title="Filtrage Collaboratif",
    page_icon="👋",
    layout="wide",
)
st.sidebar.write("Paramètres Filtrage Collaboratif")
lod = st.sidebar.checkbox('Linked Open Data',key='lod')
context = st.sidebar.checkbox('Informations Contextuelles',key='context')
sentiment = st.sidebar.checkbox('Analyse de Sentiments',key='sentiment')
pivot = st.session_state["pivot"]
pivotlod = st.session_state["pivotlod"]
pivotsentiment = st.session_state["pivotsentiment"]
pivotsentimentlod = st.session_state["pivotsentimentlod"]
ratings = st.session_state["ratings"]
pivotsentiment = st.session_state["pivotsentiment"]
sentimentratings = st.session_state["sentimentratings"]
movies = st.session_state["movies"]
lodratings = st.session_state["lodratings"]
sentimentlodratings = st.session_state["sentimentlodratings"]
cllodratings = st.session_state["cllodratings"]
clsentimentlodratings = st.session_state["clsentimentlodratings"]
clsentimentratings =  st.session_state["clsentimentratings"]
clratings =  st.session_state["clratings"]
 
list_users = pivot.index.unique()
list_movies = pivot.columns.unique()

number = st.number_input("Saisissez votre numéro d'utilisateur",1,8690,key='select',value=1)
numberec = st.slider("Séléctionnez le nombre de recommandations à afficher",1,96,key='numrec2',on_change=MovieList,value=1)
movie = MovieList()
for i in movie:
  st.text(i)
Results = PlotsColab()
if(Results[1]!=0):
  results = pd.DataFrame(Results[0],columns=['Précision','Rappel'])
  st.line_chart(data=results)
else: st.write("Cet utilisateur n'a aucun film pertinent")
