
import itertools
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model

def ChargerDataset(ratings,th):
    for i in range(ratings.shape[0]):
        if pd.isnull(ratings['rating'][i]) :
            ratings.loc[i,'rating'] = int(0)
        if int(ratings['rating'][i]) >= int(th):
            ratings.loc[i,'rating']= int(1)
        else: ratings.loc[i,'rating']=int(0) 
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
st.set_page_config(
    page_title="Comparaison des Différentes Approches",
    page_icon="👋",
    layout="wide",
)
col1,col2 = st.columns(2)
col1.subheader("Dataset des Evaluations")
col2.subheader("Dataset des Items")
stats = pd.DataFrame(columns=['Nombre Utilisateurs','Nombre Items','Nombre Evaluations','Nombre Commentaires'])
with col1:
    upload = st.file_uploader('Charger le dataset ',key='charger')
    if(upload is not None):
        path = upload.name
        file = pd.read_csv(path,delimiter=";",infer_datetime_format=True)
        stats.loc[len(stats.index)] = [len(file.userId.unique()),len(file.movieId.unique()),file['rating'].count(),file['review_content'].count()] 
        st.write(stats)
        write= st.dataframe(file)
    button = st.button("Binariser")
    if(button):
        ChargerDataset(file,4)      
        write = st.dataframe(file)
with col2:
    upload2 = st.file_uploader('Charger le dataset ',key='charger2')
    if(upload2 is not None):
        path = upload2.name
        file = pd.read_csv(path,delimiter=";",infer_datetime_format=True)
        st.write(file)
    


