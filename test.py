
from msilib.schema import Directory
import random
import pandas as pd
import numpy as np
from SPARQLWrapper import SPARQLWrapper, CSV
"""Chargement du Dataset (le préfiltrage se fera dans cette partie) et Transformation en relevant et non-relevant"""

"""Template SPARQL Request : 
 PREFIX dbpedia:<http://dbpedia.org/>
 PREFIX dbo:<http://dbpedia.org/ontology/>
 PREFIX dbp:<http://dbpedia.org/property/>
select ?name ?gross ?country ?director
where {
values ?input {<http://dbpedia.org/resource/Pulp_Fiction>}
?input dbp:name ?name.
?input dbp:gross ?gross.
 ?input dbp:country ?country.
 ?input dbp:director ?director
 }"""
def ChargerDataset(path,th):
    ratings = pd.read_csv(path,parse_dates=['timestamp'])
    for i in range(ratings.shape[0]):
        if ratings['rating'][i] >= float(th):
            ratings.loc[i,'rating']=float(1)
        else: ratings.loc[i,'rating']=float(0)
    return ratings
def GenTrainTest(nb_users,per):
    nbgen = int(nb_users*per)
    train = random.sample(range(1,nb_users),nbgen)
    test =  list()
    i=0
    while(i< (nb_users-nbgen)):
        x = random.randrange(1,nb_users)
        if train.count(x) == 0:
            test.append(x)
        else: i+=1     
    return train,test
def ListRelevant(matrix,n_items,ind):
    relevants = []
    for i in range(n_items):
        if(matrix.iloc[ind,i]==1):
            relevants.append(i)
    return relevants   
def GenInputTargetUser(pivot,n_items,ind):
    i=0
    Input = np.zeros((nbrel,n_items))
    Target = np.zeros((nbrel))
    for nb in train:
     for j in  ListRelevant(pivot,n_items,nb):
        Input[i] = np.array(pivot.iloc[nb,:],copy=True)
        Input[i,j]=0
        Target[i]=j
        i+=1 
    return Input,Target
"""Création des inputs et targets du RDN"""
sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.setReturnFormat(CSV)
movieslist = pd.read_csv("ml-100K/copy.csv",delimiter=";")
values = movieslist['URI'].unique()
f = open("resultssparql.txt",'w')
for i in range(len(values)):
    sparql.setQuery("""
      PREFIX dbpedia:<http://dbpedia.org/>
 PREFIX dbo:<http://dbpedia.org/ontology/>
 PREFIX dbp:<http://dbpedia.org/property/>
select ?name ?release
where {
values ?input {<"""+values[i]+""">}
?input dbo:name ?name.
 OPTIONAL { 
 ?input dbp:released ?release. }
}
""")
    try:
       print(values[i])
       ret= sparql.queryAndConvert()
       f.write(ret.decode())
       print(i)
    except Exception as e:
       print(e)
f.close()
"""
train = GenTrainTest(n_users,0.8)[0]
test = GenTrainTest(n_users,0.8)[1]
i=0
nbrel=0
for nb in train:
    nbrel = nbrel + len(ListRelevant(pivot,n_items,nb))
    
InputTr = np.zeros((nbrel,n_items))
TargetTr = np.zeros((nbrel))
for nb in train:
  for j in  ListRelevant(pivot,n_items,nb):
        InputTr[i] = np.array(pivot.iloc[nb,:],copy=True)
        InputTr[i,j]=0
        TargetTr[i]=j
        i+=1


model = Sequential()
model.add(Input(shape=(nbrel,n_items)))
model.add(Dense(600, activation='elu'))
model.add(Dropout(rate=0.2))
model.add(Dense(300,activation='elu'))
model.add(Dropout(rate=0.2))
model.add(Dense(n_items,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(InputTr,TargetTr,validation_split=0.1,epochs=150,batch_size=250)

nbrel=0
for nb in test:
    nbrel = nbrel + len(ListRelevant(pivot,n_items,nb))
InputTe = np.zeros((nbrel,n_items))
TargetTe = np.zeros((nbrel))
i=0
for nb in test:
  for j in  ListRelevant(pivot,n_items,nb):
        InputTe[i] = np.array(pivot.iloc[nb,:],copy=True)
        InputTe[i,j]=0
        TargetTe[i]=j
        i+=1
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

print("Evaluate on test data")
results = model.evaluate(InputTe, TargetTe, batch_size=128)
print("test loss, test acc:", results)"""
