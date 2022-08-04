
from math import nan
import pandas as pd
import numpy as np
import fractions as fr
def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split('/')
        except ValueError:
            return None
        try:
            leading, num = num.split(' ')
        except ValueError:
            return float(num) / float(denom)        
        if float(leading) < 0:
            sign_mult = -1
        else:
            sign_mult = 1
        return float(leading) + sign_mult * (float(num) / float(denom))
ratings = pd.read_csv("normalizedreviews.csv",delimiter=";",parse_dates=['review_date'],infer_datetime_format=True)
"""movies =  pd.read_csv("rotten_tomatoes_movies.csv",delimiter=",")
ratings = ratings.drop(['top_critic','publisher_name','review_type'],axis=1)
rand_movies = np.random.choice(ratings['rotten_tomatoes_link'].unique(), 
                                size=8000, 
                                replace=False)

ratings = ratings.loc[ratings['rotten_tomatoes_link'].isin(rand_movies)]
movies = movies.loc[movies['rotten_tomatoes_link'].isin(rand_movies)] 
ratings.to_csv("reviews.csv")
movies.to_csv("movies.csv")"""

val = ['A','B','C','D','F','A+','A-','B+','B-','C+','C-','D+','D-','F+','F-']
"""for i in range(ratings.shape[0]):
    if(str(ratings.loc[i,'review_score']) not in val):
     print(ratings.loc[i,'review_score'])
     value = convert_to_float(ratings.loc[i,'review_score'])
     fraction = fr.Fraction(str(value))
     ratings.loc[i,'review_score'] = int(fraction.numerator*(4/fraction.denominator))+1
     print(ratings.loc[i,'review_score'])
ratings.to_csv("normalizedreviews.csv")"""
def ChargerDataset(ratings,th):
    for i in range(ratings.shape[0]):
     if not  any ( x in val for x in str(ratings.loc[i,'review_score'])):
        if float(ratings['review_score'][i]) == float(5000):
            ratings.loc[i,'review_score'] = float(0)
        if float(ratings['review_score'][i]) >= float(th):
            ratings.loc[i,'review_score']=float(1)
        else: ratings.loc[i,'review_score']=float(0) 

for i in range(ratings.shape[0]):
    if("A" in list(str(ratings.loc[i,'review_score']))):
        print(str(ratings.loc[i,'review_score']))
        if("+" in list(str(ratings.loc[i,'review_score']))):
            ratings.loc[i,'review_score']=int(5)
        elif("-" in list(str(ratings.loc[i,'review_score']))):
            ratings.loc[i,'review_score']=int(4.75)
        else: ratings.loc[i,'review_score']=int(5)
    if("B" in list(str(ratings.loc[i,'review_score']))):
        print(str(ratings.loc[i,'review_score']))
        if("+" in list(str(ratings.loc[i,'review_score']))):
            ratings.loc[i,'review_score']=int(4.5)
        elif("-" in list(str(ratings.loc[i,'review_score']))):
            ratings.loc[i,'review_score']=int(4.25)
        else: ratings.loc[i,'review_score']=int(4)
    if("C" in list(str(ratings.loc[i,'review_score']))):
        print(str(ratings.loc[i,'review_score']))
        if("+" in list(str(ratings.loc[i,'review_score']))):
            ratings.loc[i,'review_score']=int(3.5)
        elif("-" in list(str(ratings.loc[i,'review_score']))):
            ratings.loc[i,'review_score']=int(3.25)
        else: ratings.loc[i,'review_score']=int(3)
    if("D" in list(str(ratings.loc[i,'review_score']))):
        print(str(ratings.loc[i,'review_score']))
        if("+" in list(str(ratings.loc[i,'review_score']))):
            ratings.loc[i,'review_score']=int(2.5)
        elif("-" in list(str(ratings.loc[i,'review_score']))):
            ratings.loc[i,'review_score']=int(2.25)
        else : ratings.loc[i,'review_score']=int(2)
    if("F" in list(str(ratings.loc[i,'review_score']))):
        print(str(ratings.loc[i,'review_score']))
        if("+" in list(str(ratings.loc[i,'review_score']))):
            ratings.loc[i,'review_score']=int(1.5)
        elif("-" in list(str(ratings.loc[i,'review_score']))):
            ratings.loc[i,'review_score']=int(1.25)
        else: ratings.loc[i,'review_score']=int(1)
ratings.to_csv("normalizedreviews.csv")
"""
ChargerDataset(ratings,4)
ratings.to_csv("filteredratings.csv")
pivot = ratings.pivot_table(index=['username'],columns=['movieId'],values='review_score',fill_value=0)
print(pivot.columns.unique())
print(pivot.index.unique())
print(pivot)"""