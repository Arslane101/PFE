import pandas as pd
import matplotlib as plt
import numpy as np
def ListRel(array):
    relevants = []
    for i in range(len(array)):
        if(array[i]>=4):
            relevants.append(i)
    return relevants 

ratings = pd.read_csv("specificratings.csv",delimiter=";",parse_dates=["timestamp"])
movies = pd.read_csv("ml-100k/filmsenrichis.csv",delimiter=";")
ratings = pd.merge(ratings,movies,on="movieId")
ratings_df = pd.DataFrame(ratings.groupby('Title')['rating'].mean())
ratings_df.rename(columns={'rating': 'average_rating'}, inplace=True)
ratings_df['num_of_ratings'] = pd.DataFrame(ratings.groupby('Title')['rating'].count())
pivot = ratings.pivot_table(index=['userId'],columns=['Title'],values='rating')
list_movies = movies['Title'].values
def SimilarMovies(film, min_num_reviews):
    list_ratings = pivot[film]
    similar_to_film =pivot.corrwith(list_ratings)
    corr_film_x = pd.DataFrame(similar_to_film, columns=['Correlation'])
    corr_film_x.dropna(inplace=True)
    corr_film_x = corr_film_x.join(ratings_df['num_of_ratings'])
    new_corr_film_x = corr_film_x[corr_film_x['num_of_ratings'] >= min_num_reviews]
    return new_corr_film_x.sort_values('Correlation',ascending=False)
def GenAllRecommendations(userratings,minreviews):
    list_all_movies = ListRel(userratings)
    moviestitles = list()
    for id in list_all_movies:
        temp = SimilarMovies(list_movies[id],minreviews)
        if(len(temp.index)>=5):
          for i in range(5):
            moviestitles.append(temp.index[i+1])
    return moviestitles


Users = np.loadtxt("RandomUsers.txt")
print(GenAllRecommendations(Users[0],50))