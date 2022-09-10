from contextlib import nullcontext
import itertools
from unittest import result
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def Traitement():
  return nullcontext
def Hybrid(alpha):
    cbresults = st.session_state['cbresults']
    cfresults = st.session_state['resultcf']
    cbresults = cbresults.sort_values(by=['movieId'])
    cbresults = cbresults.reset_index()
    cbresults['similarity'] = cbresults['similarity'].apply(lambda x:  x/int(round(cbresults['similarity'].sum())))
    cfresults = cfresults.sort_values(by=['movieId'])
    cfresults = cfresults.reset_index()
    cfresults['probability']= cfresults['probability'].apply(lambda x:  x/int(round(cfresults['probability'].sum())))

def MovieList():
    num = st.session_state['numrec2']
    result,results = Evaluations()
    temp = results[:num]
    movieslist = list()
    for i in temp:
      movieslist.append(movies[movies['rotten_tomatoes_link']==list_movies[i]]['movie_title'].unique())
    return movieslist
def PlotsColab():
    lod = st.session_state["lod"]
    sentiment =st.session_state["sentiment"]
    result,results = Evaluations()
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
    return result,mesures,len(rev) 

number = st.number_input("Saisissez votre numéro d'utilisateur",1,8690,key='select',value=1)
numberec = st.slider("Séléctionnez le nombre de recommandations à afficher",1,96,key='numrec2',on_change=MovieList,value=1)
Hybrid(0)
"""movie = MovieList()
for i in movie:
  st.text(i)
Results = PlotsColab()
if(Results[1]!=0):
  results = pd.DataFrame(Results[0],columns=['Précision','Rappel'])
  st.line_chart(data=results)
else: st.write("Cet utilisateur n'a aucun film pertinent")"""
