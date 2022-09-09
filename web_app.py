import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

links = pd.read_csv('https://raw.githubusercontent.com/sherwan-m/WBSFLIX_Recommender_System/main/ml-latest-small/links.csv')
movies = pd.read_csv('https://raw.githubusercontent.com/sherwan-m/WBSFLIX_Recommender_System/main/ml-latest-small/movies.csv')
ratings = pd.read_csv('https://raw.githubusercontent.com/sherwan-m/WBSFLIX_Recommender_System/main/ml-latest-small/ratings.csv')
tags = pd.read_csv('https://raw.githubusercontent.com/sherwan-m/WBSFLIX_Recommender_System/main/ml-latest-small/tags.csv')

ratings['datetime'] = ratings['timestamp'].apply(datetime.fromtimestamp)
# assign new column for year of movies, and exratc the yeatr from title 
movies= movies.assign(year = lambda df_ : df_['title'].replace(r'(.*)\((\d{4})\)', r'\2', regex= True) )

# movies= movies.assign(title = lambda df_ : df_['title'].replace(r'(.*)\((\d{4})\)', r'\1', regex= True).str.strip())

#there sre some movies have no year in their title, i fill the year by 0
movies= movies.assign(year = lambda df_ : np.where(df_['year'].str.len() <=5 , df_['year'], 1900))
#convert the year column to int
movies['year']= movies['year'].astype(int)
# def popular_n_movies(n):
#     popular_n = (
#     ratings
#             .groupby(by='movieId')
#             .agg(rating_mean=('rating', 'mean'), rating_count=('movieId', 'count'), datetime=('datetime','mean'))
#             .sort_values(['rating_mean','datetime','rating_count'], ascending= False)
#             .loc[lambda df_ :df_['rating_count'] >= (df_['rating_count'].mean()+df_['rating_count'].median())/2]
#             .head(n)
#             .reset_index()
#     )['movieId'].to_list()
#     return movies.loc[lambda df_ : df_['movieId'].isin(popular_n)]

def popular_n_movies(n, genre):
    popular_n = (
    ratings
            .groupby(by='movieId')
            .agg(rating_mean=('rating', 'mean'), rating_count=('movieId', 'count'), datetime=('datetime','mean'))
            .sort_values(['rating_mean','datetime','rating_count'], ascending= False)
            .loc[lambda df_ :df_['rating_count'] >= (df_['rating_count'].mean()+df_['rating_count'].median())/2]
            .reset_index()
    )['movieId'].to_list()
    result = movies.loc[lambda df_ : df_['movieId'].isin(popular_n)]
    if genre != 'all_genres':
            result = result.loc[lambda df_ : df_['genres'].str.contains(genre)]
    return result.head(n)

from sklearn.metrics.pairwise import cosine_similarity

def top_n_user_based(user_id , n , genres, time_period):
    if user_id not in ratings["userId"]: 
        return pd.DataFrame(columns= ['movieId', 'title', 'genres', 'year'])
    
    users_items = pd.pivot_table(data=ratings, 
                                 values='rating', 
                                 index='userId', 
                                 columns='movieId')
    users_items.fillna(0, inplace=True)
    user_similarities = pd.DataFrame(cosine_similarity(users_items),
                                 columns=users_items.index, 
                                 index=users_items.index)
    weights = (
    user_similarities.query("userId!=@user_id")[user_id] / sum(user_similarities.query("userId!=@user_id")[user_id])
          )
    
    
    new_userids = weights.sort_values(ascending=False).head(100).index.tolist()
    new_userids.append(user_id)
    new_ratings = ratings.loc[lambda df_ : df_['userId'].isin(new_userids)]
    new_users_items = pd.pivot_table(data=new_ratings, 
                                 values='rating', 
                                 index='userId', 
                                 columns='movieId')
    new_users_items.fillna(0, inplace=True)
    new_user_similarities = pd.DataFrame(cosine_similarity(new_users_items),
                                 columns=new_users_items.index, 
                                 index=new_users_items.index)
    new_weights = (
    new_user_similarities.query("userId!=@user_id")[user_id] / sum(new_user_similarities.query("userId!=@user_id")[user_id])
          )
    
    not_watched_movies = new_users_items.loc[new_users_items.index!=user_id, new_users_items.loc[user_id,:]==0]
    weighted_averages = pd.DataFrame(not_watched_movies.T.dot(new_weights), columns=["predicted_rating"])
    recommendations = weighted_averages.merge(movies, left_index=True, right_on="movieId").sort_values("predicted_rating", ascending=False)
    recommendations = recommendations.loc[lambda df_ : ((df_['year'] >= time_period[0]) & ( df_['year'] <= time_period[1]))]
    if len(genres)>0:
            result = pd.DataFrame(columns=['predicted_rating', 'movieId', 'title', 'genres', 'year'])
            for genre in genres:
                result= pd.concat([result, recommendations.loc[lambda df_ : df_['genres'].str.contains(genre)]])
            
            result.drop_duplicates(inplace=True)
            result = result.sort_values("predicted_rating", ascending=False)
            result.reset_index(inplace=True, drop= True)
            return result.drop(columns=['predicted_rating']).head(n)
                 
    return recommendations.reset_index(drop=True).drop(columns=['predicted_rating']).head(n)


all_genres = set()

all_genres.add("all_genres")
for genres in movies["genres"].unique():
    genres_2 = genres.split(r"|")
    for genre in genres_2:
        all_genres.add(genre)

st.title("Welcome to WBSFLIX")
 
# st.write("""
# ### Recommender System
# Pleas Enter the number of movies do you like to see:.
# """)
n_movies = st.number_input("Number of movies: ", value= 10)
user_ID = st.number_input("userID: ", value= 10)
# genre_movies = st.selectbox("Select your favorite Genre", sorted(all_genres), index= 20)
genre_movies = st.multiselect("Select your favorite Genre", sorted(all_genres))
time_period = st.slider('years:', min_value=1900, max_value=2018, value=(2000,2018), step=1)
# st.select_slider("Select your favorite Genre", options=sorted(all_genres))
# if genre_movies == 'all_genres':
#     st.write("Our sujestion is:")
# st.dataframe(popular_n_movies(int(n_movies)).reset_index(drop=True))
x= top_n_user_based(user_ID,int(n_movies), genre_movies, time_period)
x
