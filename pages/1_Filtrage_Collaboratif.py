
import itertools
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model

from CollaborativeFiltering import ChargerDataset

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
def EnsembleSamplesTesting(nb):
    itemslist = np.loadtxt("Subsets.txt")
    itemlist = np.concatenate(itemslist)
    values = list()
    for i in range(itemslist.shape[0]):
        model = load_model(str(i))
        testUser = np.array(pivot.iloc[nb,:],copy=True)
        testUser = testUser.reshape(1,testUser.shape[0])
        results = model.predict(testUser)
        values.append(results)
    results = np.concatenate(np.asarray(values))
    results = np.argsort(results.reshape(itemlist.shape[0]))[::-1] 
    for i in range(results.shape[0]):
        results[i] = int(itemlist[results[i]]) 
    return results
def WriteMovieList():
    number = st.session_state['select']
    num = st.session_state['numrec2']
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
      rec =  (hr)/len(rev)
      i+=5
      precisions.append(prec)
      recalls.append(rec)    
    temp = results[:(int(num))]
    movieslist = list()
    for i in temp:
      movieslist.append(movies[movies['movieId']==list_movieids[i]]['Title'])
    return precisions,recalls,list(itertools.chain(*movieslist))

ratings = pd.read_csv("normalizedreviews.csv",delimiter=";",parse_dates=['review_date'])
ChargerDataset(ratings,4)
movies = pd.read_csv("movies.csv",delimiter=";")
pivot = ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating',fill_value=0)
list_movieids = pivot.columns.unique()
st.set_page_config(
    page_title="Comparaison de Systèmes de Recommandation",
    page_icon="👋",
)
number = st.number_input("Saisissez votre numéro d'utilisateur",1,943,key='select',on_change=WriteMovieList)
numberec = st.slider("Séléctionnez le nombre de recommandations à afficher",1,96,key='numrec2',on_change=WriteMovieList)
Results = WriteMovieList()
for i in Results[2]:
  st.text(i)
st.line_chart(data=Results[0])
st.line_chart(data=Results[1])



