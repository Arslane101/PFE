import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model
from BêtaV1Tensorflow import ListRelevant
ratings = pd.read_csv("ml-100k/filteredratings.csv",delimiter=";",parse_dates=['timestamp'])
pivot = ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating',fill_value=0)
model = load_model("ml-100k")
list_movieids = pivot.columns.unique()
movies = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
st.set_page_config(
    page_title="NCCF",
    page_icon="👋",
)
st.text("Si vous êtes un nouvel utilisateur : ")
options = st.selectbox("Séléctionnez le numéro de l'utilisateur test"
,['1','2','3','4','5'])
first = st.button('Valider')

st.text("Si vous êtes un utilisateur existant :")
number = st.number_input("Saisissez votre numéro d'utilisateur",1,943)
usertable = np.array(pivot.iloc[int(number)-1,:],copy=True)
testUser = usertable.reshape(1,usertable.shape[0])
results = model.predict(testUser)
results = np.argsort(results.reshape(testUser.shape[1]))[::-1] 
numberec = st.slider("Séléectionnez le nombre de recommandations à afficher",1,96)
temp =results[:int(numberec)]
for i in temp:
    st.text(movies.loc[movies['movieId']==list_movieids[i]]['Title'])
n=96
i=1
recalls = []
precisions = []
rev = ListRelevant(pivot,pivot.shape[1],int(number)-1)
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
st.line_chart(data=recalls)
st.line_chart(data=precisions)

