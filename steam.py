import numpy as np
import pandas as pd
"""
anime= pd.read_csv("animereviews.csv",delimiter=";")
anime = anime.drop(columns=['peopleFoundUseful','storyRating','animationRating','soundRating','characterRating','enjoymentRating','episodesSeen'])
users = anime['author'].unique().tolist()
for user in users:
    print(anime[anime['author']==user].count())
reviews = pd.read_csv("rotten_tomatoes_critic_reviews.csv",delimiter=",")
rand_userIds = np.random.choice(reviews['rotten_tomatoes_link'].unique(), 
                                size=2000, 
                                replace=False)

reviews = reviews.loc[reviews['rotten_tomatoes_link'].isin(rand_userIds)]
authors = reviews['critic_name'].unique().tolist()
for author in authors:
    if(reviews[reviews['critic_name']==author].shape[0] < 20 ):
        reviews = reviews[reviews['critic_name']!=author]
print(reviews.shape)
authors = reviews['critic_name'].unique().tolist()
for author in authors:
    print(reviews[reviews['critic_name']==author].shape[0])"""
query = pd.read_csv("query.csv",delimiter=",")
query = query[query["rating"].str.contains("%")==False]
query.to_csv("query.csv")