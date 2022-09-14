import calendar
from datetime import datetime
import keras
from unittest import result
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
            relevants.append(list_movies[i])
    return relevants 
def Relevant(matrix):
    relevants = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if(matrix.iloc[i,j]==1) and j not in relevants:
              relevants.append(j)
    return relevants   
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
def MostRatedMovies(ratings):
    ratings = ratings.groupby(['movieId'])[['rating']].mean()
    ratings = ratings[ratings["rating"] >= 3]
    return ratings.index.unique().tolist()
@st.experimental_memo
def Hybrid(subsets,mode,alpha,nb,pivot,ratings):
    cbresults = contentbased(list_users[nb],movies,ratings)
    cfresults = EnsembleSamples(subsets,mode,nb,pivot)
    cbresults = cbresults.sort_values(by=['movieId'])
    cbresults = cbresults.reset_index()
    cbresults['similarity'] = cbresults['similarity'].apply(lambda x:  x/int(round(cbresults['similarity'].sum())))
    cfresults = cfresults.sort_values(by=['movieId'])
    cfresults = cfresults.reset_index()
    cfresults['probability']= cfresults['probability'].apply(lambda x:  x/int(round(cfresults['probability'].sum())))
    hybrid = pd.DataFrame(columns=['movieId','probability'])
    for i in range(cfresults.shape[0]):
        x = alpha*cfresults['probability'][i]+(1-alpha)*cbresults['similarity'][i]
        hybrid.loc[len(hybrid.index)]=[cfresults['movieId'][i],x]
    return hybrid
@st.experimental_memo
def FilterContext2(results,movies):
    results = results.sort_values(by=['probability'],ascending=False)
    results = results.reset_index()
    result = list()
    for i in range(results.shape[0]):
        if(results['movieId'][i] in movies):
            result.append(results['movieId'][i])
    return result
def Evaluations():
  lod = st.session_state["lod2"]
  sentiment =st.session_state["sentiment2"]
  context2 =  st.session_state["context2"]
  alpha = st.session_state["alpha"]
  nb = st.session_state['select2']
  nb = int(nb)-1
  if(lod==True and sentiment==True):
      results = Hybrid("LOD/Subsets.txt","SentimentsLOD/",alpha,nb,pivotsentimentlod,sentimentlodratings)
      if(context2 == True):
        context = MostRelevantMoviesbyContext(clsentimentlodratings)
        results = FilterContext2(results,context)
  elif(lod==True and  sentiment==False): 
     results = Hybrid("LOD/Subsets.txt","LOD/",alpha,nb,pivotlod,lodratings)
     if(context2 == True):
        context = MostRelevantMoviesbyContext(cllodratings)
        results =FilterContext2(results,context)
  elif(sentiment==True and lod==False):
        results = Hybrid("Classic/Subsets.txt","Sentiments/",alpha,nb,pivotsentiment,sentimentratings)
        if(context2 == True):
         context = MostRelevantMoviesbyContext(clsentimentratings)
         results =FilterContext2(results,context)
  else : 
    results = Hybrid("Classic/Subsets.txt","Classic/",alpha,nb,pivot,ratings)
    if(context2 == True):
        context = MostRelevantMoviesbyContext(clratings)
        results = FilterContext2(results,context)
  return results
@st.experimental_memo
def EnsembleSamples(subsets,mode,nb,pivot):
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
    results = results.reshape(itemlist.shape[0])
    result = pd.DataFrame(columns=['movieId','probability'])
    for i in range(results.shape[0]):
        result.loc[len(result.index)]=[list_movies[int(itemlist[i])],results[i]]
    return result
@st.experimental_memo
def contentbased(user,movies,ratings):
    allmovies = ratings.movieId.unique()
    allmovies = list(set(allmovies)-set(movies.rotten_tomatoes_link))
    tf = TfidfVectorizer(stop_words='english')
    tfidf_matrix_item = tf.fit_transform(movies['movie_info'])
    userRate = ratings[ratings['userId'] == user ]
    print(user)
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
    return cosine_sim_df
def MovieList():
  results= PlotHybrid()[0]
  movieslist = list()
  num = st.session_state['numrec5']
  if(st.session_state['context2']==False):
    results = results.movieId.unique()
    temp = results[:num]
    for i in temp:
     movieslist.append(i)
  else : 
    temp = results[:num]
    for i in temp:
     movieslist.append(i)
  return movieslist


def PlotHybrid():
  lod = st.session_state["lod2"]
  sentiment =st.session_state["sentiment2"]
  n=96
  i=1
  recalls = []
  precisions = []
  results = Evaluations() 
  if(lod==True and sentiment==True):
    testUser = np.array(pivotsentimentlod.iloc[int(st.session_state['select2'])-1,:],copy=True)
    rev = ListSpecRel(testUser)
  elif(lod==True and  sentiment==False):
    testUser = np.array(pivotlod.iloc[int(st.session_state['select2'])-1,:],copy=True)
    rev = ListSpecRel(testUser)
  elif(sentiment==True and lod==False):
    testUser = np.array(pivotsentiment.iloc[int(st.session_state['select2'])-1,:],copy=True)
    rev = ListSpecRel(testUser)
  else : 
    testUser = np.array(pivot.iloc[int(st.session_state['select2'])-1,:],copy=True)
    rev = ListSpecRel(testUser)
  if(st.session_state["context2"] == True):
    
    i=1
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
  else: 
    results = results.sort_values(by=['probability'],ascending=False)
    results = results.reset_index()
    while(i<n):
      rec = 0
      prec = 0
      hr=0
      temp = results.loc[0:i-1,:]
      for k in range(len(temp)):
        if  temp['movieId'][k] in rev:
          hr+=1
      prec =  (hr)/i
      if(len(rev)!=0):
        rec =  (hr)/len(rev)
      else: rec=0
      i+=1
      precisions.append(prec)
      recalls.append(rec)
    mesures = np.column_stack((precisions,recalls))  
  return results,mesures,len(rev) 

st.set_page_config(
    page_title="Filtrage Hybride",
    page_icon="👋",
    layout="wide",
)
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
st.sidebar.write("Paramètres Filtrage Hybride")
lod = st.sidebar.checkbox('Linked Open Data',key='lod2')
context = st.sidebar.checkbox('Informations Contextuelles',key='context2')
sentiment = st.sidebar.checkbox('Analyse de Sentiments',key='sentiment2')
alpha = st.sidebar.number_input("Alpha",0.2,1.0,step=0.1,key='alpha')
number = st.number_input("Saisissez votre numéro d'utilisateur",1,8690,key='select2',value=1)
numberec = st.slider("Séléctionnez le nombre de recommandations à afficher",1,96,key='numrec5',on_change=MovieList,value=1)
Results = PlotHybrid()
for i in MovieList():
  st.text(i)
if(Results[2]!=0):
  results = pd.DataFrame(Results[1],columns=['Précision','Rappel'])
  st.line_chart(data=results)
else: st.write("Cet utilisateur n'a aucun film pertinent")
