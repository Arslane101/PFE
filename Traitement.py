
import pandas as pd
import numpy as np
ratings = pd.read_csv("reviews.csv",delimiter=";",parse_dates=['review_date'])
movies = pd.read_csv("movies.csv",delimiter=";")
print(movies.head(10))
print(ratings.info())
"""ratings = ratings.drop(['top_critic','publisher_name','review_type'],axis=1)
rand_movies = np.random.choice(ratings['rotten_tomatoes_link'].unique(), 
                                size=8000, 
                                replace=False)

ratings = ratings.loc[ratings['rotten_tomatoes_link'].isin(rand_movies)]
movies = movies.loc[movies['rotten_tomatoes_link'].isin(rand_movies)] 
ratings.to_csv("reviews.csv")
movies.to_csv("movies.csv")"""