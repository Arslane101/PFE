import pandas as pd
import numpy as np

def ChargerDataset(path,th):
    ratings = pd.read_csv(path,delimiter=";",parse_dates=['timestamp'])
    """rand_movies = np.random.choice(ratings['movieId'].unique(), 
                                size=int(len(ratings['movieId'].unique())*per), 
                                replace=False)

    ratings = ratings.loc[ratings['movieId'].isin(rand_movies)]
    ls = []
    ls.extend(ratings.index[(ratings['rating']>=0)])"""
    for i in range(ratings.shape[0]):
        if ratings['rating'][i] >= float(th):
            ratings.loc[i,'rating']=float(1)
        else: ratings.loc[i,'rating']=float(0) 
    ratings.to_csv("filteredratings.csv")
    return ratings
def CheckValues():
    ratings = pd.read_csv("ml-100k/ratings.csv",delimiter=";")
    pivot = ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating',fill_value=0)
    list_items = pivot.columns.unique()
    movies = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
    list_movies = movies['movieId'].unique().tolist()
    for i in range(len(list_items)):
        if(list_movies.count(list_items[i]) ==0 ):
            ratings.drop(ratings[ratings['movieId']==list_items[i]].index,inplace=True)
    ratings.to_csv("specificratings.csv")
    return ratings
def ContextualisationDataset(ratings,th,genrelist,country):
    items = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
    movies = pd.read_csv("ml-100k/dbpediamovies.csv",delimiter=";")
    specificmovies = movies[movies['country'].isin(country)]
    uniques = specificmovies['name'].unique()
    for i in range(ratings.shape[0]):
        listid = GenresSpecificMovie(ratings['movieId'][i])
        temp = items.loc[items['movieId']==ratings['movieId'][i]]
        title = temp['SPARQLTitle'][temp.index[0]]
        if(len(list(set(genrelist).intersection(listid)))!=0 ):
                ratings.loc[i,'rating']=float(1)
        else: ratings.loc[i,'rating']=float(0)
    return ratings
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
def GenTrainTest(nb_users,per):
    nbgen = int(nb_users*per)
    train = random.sample(range(1,nb_users),nbgen)
    test =  list()
    i=0
    while(i< (nb_users-nbgen)):
        x = random.randrange(1,nb_users)
        if train.count(x) == 0:
            test.append(x)
            i+=1     
    return train,test
def ListRelevant(matrix,n_items,ind):
    relevants = []
    for i in range(n_items):
        if(matrix.iloc[ind,i]==1):
            relevants.append(matrix.columns.unique()[i])
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
              relevants.append(matrix.columns.unique()[j])
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
def BuildProfile(ratings,ind):
   with open("ml-100k/genres.txt") as f:
    genres = f.readlines()
   movies = pd.read_csv("ml-100k/item.csv",delimiter=";")
   size = len(genres)
   profile = np.zeros(size)
   userratings= ratings.loc[ratings['userId']==ind]
   ratedmovies = userratings['movieId'].unique()
   for i in ratedmovies:
       for j in range(len(genres)):
           genre = genres[j].strip()
           if(movies.loc[i,genre]==1):
               profile[j]=1
   return profile
def UserMostMoviesbyCountry(pivot,country):
    items = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
    movies = pd.read_csv("ml-100k/dbpediamovies.csv",delimiter=";")
    specificmovies = movies[movies['country'].isin(country)]['name'].unique()
    print(specificmovies)
    uniqueids = items[items['SPARQLTitle'].isin(specificmovies)]['movieId'].unique()
    userids = pivot.index.unique()
    maxmovies= len(set(uniqueids).intersection(ListRelevant(pivot,pivot.shape[1],userids[0])))  
    maxuser = 0
    i=0
    while(i !=len(userids)):
        relevant = ListRelevant(pivot,pivot.shape[1],i)
        if(len(set(uniqueids).intersection(relevant))>maxmovies):
            maxmovies = len(set(uniqueids).intersection(relevant))
            maxuser = i
        i+=1

    return maxuser

ratings = pd.read_csv("ml-100k/filteredratings.csv",delimiter=";",parse_dates=['timestamp'])
movie = pd.read_csv("ml-100k/dbpediamovies.csv",delimiter=";")
pivot = ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating',fill_value=0)
items = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
user = UserMostMoviesbyCountry(pivot,['Spain'])
movielist = ListRelevant(pivot,pivot.shape[1],user)
for i in movielist:
    #print(movie[movie['name']==items.loc[movielist[i],'SPARQLTitle']]) 
    print(i)
    print(items[items['movieId']==i]['SPARQLTitle'])  
print(len(movielist))
print(user)
