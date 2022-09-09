
import itertools
from unittest import result
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model


@st.experimental_memo
def LoadData():
  ratings = pd.read_csv("binarizedratings.csv",delimiter=";",parse_dates=['review_date'])
  sentimentratings = pd.read_csv("BinarizedSentimentRatings.csv",delimiter=";",parse_dates=['review_date'])
  lodratings =  pd.read_csv("binarizedratingsLOD.csv",delimiter=";",parse_dates=['review_date'])
  sentimentlodratings =  pd.read_csv("BinarizedSentimentLODRatings.csv",delimiter=";",parse_dates=['review_date'])
  movies = pd.read_csv("movies.csv",delimiter=";")
  pivot = ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating',fill_value=0)
  pivotsentiment = sentimentratings.pivot_table(index=['userId'],columns=['movieId'],values='rating',fill_value=0)
  pivotlod = lodratings.pivot_table(index=['userId'],columns=['movieId'],values='rating',fill_value=0)
  pivotsentimentlod = sentimentlodratings.pivot_table(index=['userId'],columns=['movieId'],values='rating',fill_value=0)
  return pivot,pivotlod,pivotsentiment,pivotsentimentlod,movies
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
def ContextFiltering(results):
  return results
def Evaluations():
  lod = st.session_state["lod"]
  sentiment =st.session_state["sentiment"]
  context =  st.session_state["context"]
  nb = st.session_state['select']
  nb = int(nb)-1
  if(lod==True and sentiment==True):
      results = EnsembleSamplesTesting("LOD/Subsets.txt","SentimentsLOD/",nb,pivotsentimentlod)
      if(context):
        results = ContextFiltering()
  elif(lod==True and  sentiment==False): 
     results = EnsembleSamplesTesting("LOD/Subsets.txt","LOD/",nb,pivotlod)
     if(context):
      results = ContextFiltering()
  elif(sentiment==True and lod==False):
        results = EnsembleSamplesTesting("Classic/Subsets.txt","Sentiments/",nb,pivotsentiment)
        if(context):
         results = ContextFiltering()
  else : 
    results = EnsembleSamplesTesting("Classic/Subsets.txt","Classic/",nb,pivot)
    if(context):
      results = ContextFiltering()
  return results
@st.experimental_singleton
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
def MovieList():
    num = st.session_state['numrec2']
    results = Evaluations()
    temp = results[:num]
    movieslist = list()
    for i in temp:
      movieslist.append(list_movies[i])
    return movieslist
def Plots():
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
    page_title="Comparaison des Différentes Approches",
    page_icon="👋",
    layout="wide",
)
st.sidebar.write("Paramètres Filtrage Collaboratif")
lod = st.sidebar.checkbox('Linked Open Data',key='lod')
context = st.sidebar.checkbox('Informations Contextuelles',key='context')
sentiment = st.sidebar.checkbox('Analyse de Sentiments',key='sentiment')
st.sidebar.write("Paramètres Hybride")
st.sidebar.number_input("Alpha",0.2,1.0,step=0.1)
col1, col2,col3 = st.tabs(["Filtrage Collaboratif","Filtrage basé Contenu","Filtrage Hybride"])

pivot,pivotlod,pivotsentiment,pivotsentimentlod,movies = LoadData()

list_movies = pivot.columns.unique()
with col1:
  number = st.number_input("Saisissez votre numéro d'utilisateur",1,943,key='select',value=1)
  numberec = st.slider("Séléctionnez le nombre de recommandations à afficher",1,96,key='numrec2',on_change=MovieList,value=1)
  movie = MovieList()
  for i in movie:
   st.text(i)
  Results = Plots()
  if(Results[1]!=0):
    results = pd.DataFrame(Results[0],columns=['Précision','Rappel'])
    st.line_chart(data=results)
  else: st.write("Cet utilisateur n'a aucun film pertinent")
with col2:
  number = st.number_input("Saisissez votre numéro d'utilisateur",1,8690,key='select1')
  numberec = st.slider("Séléctionnez le nombre de recommandations à afficher",1,96,key='numrec3')
with col3:
  number = st.number_input("Saisissez votre numéro d'utilisateur",1,8690,key='select2')
  numberec = st.slider("Séléctionnez le nombre de recommandations à afficher",1,96,key='numrec4')
    



