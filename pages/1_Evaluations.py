
import itertools
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model

from CollaborativeFiltering import ChargerDataset
@st.experimental_memo
def LoadData():
  ratings = pd.read_csv("binarizedratings.csv",delimiter=";",parse_dates=['review_date'])
  movies = pd.read_csv("movies.csv",delimiter=";")
  return ratings,movies
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
@st.experimental_singleton
def EnsembleSamplesTesting(nb):
    lod = st.session_state['lod']
    context = st.session_state['context']
    sentiment = st.session_state['context']
    if(lod):
      itemslist = np.loadtxt("LOD/Subsets.txt")
      itemlist = np.concatenate(itemslist)
      values = list()
      for i in range(itemslist.shape[0]):
        model = load_model("Classic/"+str(i))
        testUser = np.array(pivot.iloc[nb,:],copy=True)
        testUser = testUser.reshape(1,testUser.shape[0])
        results = model.predict(testUser)
        values.append(results)
      results = np.concatenate(np.asarray(values))
      results = np.argsort(results.reshape(itemlist.shape[0]))[::-1] 
      for i in range(results.shape[0]):
        results[i] = int(itemlist[results[i]]) 
    elif(context):
      if(lod):
        itemslist = np.loadtxt("LOD/Subsets.txt")
        itemlist = np.concatenate(itemslist)
        values = list()
        for i in range(itemslist.shape[0]):
         model = load_model("Classic/"+str(i))
         testUser = np.array(pivot.iloc[nb,:],copy=True)
         testUser = testUser.reshape(1,testUser.shape[0])
         results = model.predict(testUser)
         values.append(results)
        results = np.concatenate(np.asarray(values))
        results = np.argsort(results.reshape(itemlist.shape[0]))[::-1] 
        for i in range(results.shape[0]):
         results[i] = int(itemlist[results[i]]) 
    elif(sentiment):
      if(lod):
        itemslist = np.loadtxt("SentimentLOD/Subsets.txt")
        itemlist = np.concatenate(itemslist)
        values = list()
        for i in range(itemslist.shape[0]):
         model = load_model("Classic/"+str(i))
         testUser = np.array(pivot.iloc[nb,:],copy=True)
         testUser = testUser.reshape(1,testUser.shape[0])
         results = model.predict(testUser)
         values.append(results)
        results = np.concatenate(np.asarray(values))
        results = np.argsort(results.reshape(itemlist.shape[0]))[::-1] 
        for i in range(results.shape[0]):
         results[i] = int(itemlist[results[i]]) 
      else: 
        itemslist = np.loadtxt("Sentiment/Subsets.txt")
        itemlist = np.concatenate(itemslist)
        values = list()
        for i in range(itemslist.shape[0]):
         model = load_model("Classic/"+str(i))
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
    results = EnsembleSamplesTesting(int(number)-1)
    temp = results[:(int(num))]
    movieslist = list()
    for i in temp:
      movieslist.append(list_movies[i])
    return movieslist
def Plots():
    number = st.session_state['select']
    results = EnsembleSamplesTesting(int(number)-1)
    n=96
    i=1
    recalls = []
    precisions = []
    rev = ListRelevant(pivot,pivot.shape[1],int(st.session_state['select'])-1)
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
    return precisions,recalls,len(rev)
st.set_page_config(
    page_title="Comparaison des Différentes Approches",
    page_icon="👋",
    layout="wide",
)
st.sidebar.write("Paramètres Filtrage Collaboratif")
lod = st.sidebar.checkbox('Linked Open Data',on_change=EnsembleSamplesTesting,key='lod')
context = st.sidebar.checkbox('Informations Contextuelles',on_change=EnsembleSamplesTesting,key='context')
sentiment = st.sidebar.checkbox('Analyse de Sentiments',on_change=EnsembleSamplesTesting,key='sentiment')
st.sidebar.write("Paramètres Hybride")
st.sidebar.number_input("Alpha",0.2,1.0,step=0.1)
col1, col2,col3 = st.tabs(["Filtrage Collaboratif","Filtrage basé Contenu","Filtrage Hybride"])

ratings,movies = LoadData()
pivot = ratings.pivot_table(index=['userId'],columns=['movie_title'],values='rating',fill_value=0)
list_movies = pivot.columns.unique()
with col1:
  number = st.number_input("Saisissez votre numéro d'utilisateur",1,943,key='select',on_change=Plots)
  numberec = st.slider("Séléctionnez le nombre de recommandations à afficher",1,96,key='numrec2',on_change=MovieList)
  movie = MovieList()
  for i in movie:
   st.text(i)
  Results = Plots()
  if(Results[2]!=0):
    st.line_chart(data=Results[0])
    st.line_chart(data=Results[1])
  else: st.write("Cet utilisateur n'a aucun film pertinent")
with col2:
  number = st.number_input("Saisissez votre numéro d'utilisateur",1,8690,key='select')
  numberec = st.slider("Séléctionnez le nombre de recommandations à afficher",1,96,key='numrec2')
with col3:
  number = st.number_input("Saisissez votre numéro d'utilisateur",1,8690,key='select')
  numberec = st.slider("Séléctionnez le nombre de recommandations à afficher",1,96,key='numrec2')
    



