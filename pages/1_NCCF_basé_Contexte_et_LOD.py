
import calendar
import itertools
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model
import datetime as dt

def ListRelevant(matrix,n_items,ind):
    relevants = []
    for i in range(n_items):
        if(matrix.iloc[ind,i]==1):
            relevants.append(i)
    return relevants   
def ListSpecRel(array):
    relevants = []
    for i in range(len(array)):
        if(array[i]==1):
            relevants.append(list_movieids[i])
    return relevants 
def Relevant(matrix):
    relevants = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if(matrix.iloc[i,j]==1) and j not in relevants:
              relevants.append(j)
    return relevants   
def WriteMovieList():
    number = st.session_state['select']
    num = st.session_state['numrec2']
    usertable = np.array(pivot.iloc[int(number)-1,:],copy=True)
    testUser = usertable.reshape(1,usertable.shape[0])
    results = model.predict(testUser)
    results = np.argsort(results.reshape(testUser.shape[1]))[::-1]
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
def PredictionNewUser():
    number = st.session_state["slider1"]
    numberec = st.session_state["numrec"]
    usertable = randomusers[int(number)-1,:]
    rev = ListSpecRel(usertable)
    globalrev = MostRelevantMoviesbyContext(ratings)
    interesection = list(set(rev).intersection(globalrev))
    testUser = usertable.reshape(1,usertable.shape[0])
    results = model.predict(testUser)
    results = np.argsort(results.reshape(testUser.shape[1]))[::-1]
    temp = results[:(int(numberec))]
    movieslist = list()
    for i in temp:
     movieslist.append(movies[movies['movieId']==list_movieids[i]]['Title'].unique().tolist())
    n=96
    i=1
    recalls = []
    precisions = []
    while(i<n):
      rec = 0
      prec = 0
      hr=0
      temp = results[:i]
      for k in range(len(temp)):
        if  list_movieids[temp[k]] in interesection :
         hr+=1
      prec =  (hr)/i
      rec =  (hr)/len(interesection)
      i+=5
      precisions.append(prec)
      recalls.append(rec)    
    return precisions,recalls,list(itertools.chain(*movieslist))
def AllMoviesbyCountry(country):
    items = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
    movies = pd.read_csv("ml-100k/dbpediamovies.csv",delimiter=";")
    specificmovies = movies[movies['country'].isin(country)]['name'].unique()
    uniqueids = items[items['SPARQLTitle'].isin(specificmovies)]['movieId'].unique()
    return uniqueids
def MostRelevantMoviesbyContext(ratings):
    currentdate = dt.datetime.now()
    currentday = currentdate.strftime("%A")
    weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
    weekend = ["Saturday","Sunday"]
    listmovies = list()
    if(currentday in weekdays):
      for i in range(ratings.shape[0]):
        if(ratings["rating"][i]==1) and calendar.day_name[ratings["timestamp"][i].weekday()] in weekdays:
            if(ratings['movieId'][i] not in listmovies):
                listmovies.append(ratings['movieId'][i])
    else : 
      for i in range(ratings.shape[0]):
        if(ratings["rating"][i]==1 and calendar.day_name[ratings["timestamp"][i].weekday()] in weekend):
            if(ratings['movieId'][i] not in listmovies):
                listmovies.append(ratings['movieId'][i])
    return listmovies
def RelevantContextMovies(ratings,country):
    uniqueids = AllMoviesbyCountry(country)
    currentdate = dt.now()
    currentday = currentdate.strftime("%A")
    weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
    weekend = ["Saturday","Sunday"]
    listmovies = list()
    if(currentday in weekdays):
      for i in range(ratings.shape[0]):
        if(ratings['movieId'][i] in uniqueids and ratings["rating"][i]==1) and calendar.day_name[ratings["timestamp"][i].weekday()] in weekdays:
            listmovies.append(ratings['movieId'][i])
    else : 
      for i in range(ratings.shape[0]):
        if(ratings['movieId'][i] in uniqueids and ratings["rating"][i]==1 and calendar.day_name[ratings["timestamp"][i].weekday()] in weekend):
            listmovies.append(ratings['movieId'][i])
    return listmovies
def GetTrendsMovies(listmovies):
    genrelist = open("ml-100k/genres.txt","r").readlines()
    movies = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
    trends = np.zeros(len(genrelist))
    for i in range(len(genrelist)):
        for id in listmovies:
            temp = movies.loc[movies['movieId']==id]
            val =temp.index
            if(len(val)!=0):
             if(temp[genrelist[i].strip()][val[0]]==1):
                 trends[i]+=1
    genreids = np.argsort(trends)[::-1]
    sortedgenres = list()
    for i in genreids:
        sortedgenres.append(genrelist[i].strip())
    print(trends) 
    return sortedgenres
def GenresSpecificMovie(id):
    movies = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
    moviegenre = list()
    genrelist = open("ml-100k/genres.txt","r").readlines()
    for i in range(len(genrelist)):
        temp = movies.loc[movies['movieId']==id]
        val =temp.index
        if(len(val)!=0):
            if(temp[genrelist[i].strip()][val[0]]==1):
             moviegenre.append(genrelist[i].strip())
    return moviegenre 
ratings = pd.read_csv("ml-100k/filteredratings.csv",delimiter=";",parse_dates=['timestamp'])
pivot = ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating',fill_value=0)
randomusers = np.loadtxt("RandomUsers.txt")
model = load_model("ml-100k")
list_movieids = pivot.columns.unique()
movies = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
st.set_page_config(
    page_title="Comparaison de Systèmes de Recommandation",
    page_icon="👋",
)
st.text("Nouvel utilisateur : ")
options = st.selectbox("Séléctionnez votre numéro d'utilisateur"
,['1','2','3','4','5','6','7','8','9','10'],key='slider1',on_change=PredictionNewUser)
lod1 = st.checkbox("Include la Localisation")
if(lod1):
    country = st.selectbox("Si vous êtes dans l'un de ces pays, sélectionnez le",["United States","France","United Kingdom",
    "Italy","Canada"])
numberec2 = st.slider("Sélectionnez le nombre de recommandations à afficher",1,96,key='numrec',on_change=PredictionNewUser)

Results = PredictionNewUser()
for i in Results[2]:
  st.text(i)
st.line_chart(data=Results[0])
st.line_chart(data=Results[1])




