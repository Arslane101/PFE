
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
  return ratings,sentimentratings,pivot,pivotlod,pivotsentiment,pivotsentimentlod,movies
st.set_page_config(
    page_title="Comparaison des Différentes Approches",
    page_icon="👋",
    layout="wide",
)
ratings,sentimentratings,pivot,pivotlod,pivotsentiment,pivotsentimentlod,movies = LoadData()
if('ratings' not in st.session_state):
    st.session_state.ratings = ratings
if('sentimentratings' not in st.session_state):
    st.session_state.sentimentratings = sentimentratings
if('pivot' not in st.session_state):
    st.session_state.pivot = pivot
if('pivotlod' not in st.session_state):
    st.session_state.pivotlod = pivotlod
if('pivotsentiment'):
    st.session_state.pivotsentiment = pivotsentiment
if('pivotsentimentlod'):
    st.session_state.pivotsentimentlod = pivotsentimentlod
if('movies' not in st.session_state):
    st.session_state.movies = movies
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
