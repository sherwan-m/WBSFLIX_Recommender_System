#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 14:09:23 2022
@author: ilkayisik
Streamlit app for user based movie recommendations
Changes to the first version:
1. put all the widgets to the sidebar
2. add the time period option in the user id based recommendation
"""
# imports
from turtle import title
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from datetime import datetime
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
# %% load data
movie_df = pd.read_csv('https://raw.githubusercontent.com/sherwan-m/WBSFLIX_Recommender_System/main/ml-latest-small/movies.csv')
rating_df = pd.read_csv('https://raw.githubusercontent.com/sherwan-m/WBSFLIX_Recommender_System/main/ml-latest-small/ratings.csv')
links_df = pd.read_csv('https://raw.githubusercontent.com/sherwan-m/WBSFLIX_Recommender_System/main/ml-latest-small/links.csv')
tags_df = pd.read_csv('https://raw.githubusercontent.com/sherwan-m/WBSFLIX_Recommender_System/main/ml-latest-small/tags.csv')
# %% format dataframes
# MOVIE DF:
movie_df = (
    movie_df
        .assign(year=lambda df_ : df_['title'].replace(r'(.*)\((\d{4})\)', r'\2', regex= True))
        # replace with 0 if there is no year
        .assign(year=lambda df_ : np.where(df_['year'].str.len() <=5 , df_['year'], 0)))
# convert the year column to int
movie_df['year'] = movie_df['year'].astype(int)
movie_df['title']= movie_df['title'].str.replace(r'\'','', regex=True)
# create a genre list
genre_list = []
for i in movie_df['genres']:
    if "|" in i:
        genre_list.extend(i.rsplit("|"))
    else:
        genre_list.append(i)
genre_list = list(set(genre_list))

i = genre_list.index("(no genres listed)")
del genre_list[i]
genre_list.sort()
genre_list.insert(0, 'Any')

year_list = list(set(list(movie_df['year'])))[1:]

# create a list of movies
movie_list = list(set(list(movie_df['title'])))

# %% RATING DF
# convert timestamp to datetime format
rating_df['datetime'] = rating_df['timestamp'].apply(datetime.fromtimestamp)
# drop the timestamp column
rating_df.drop(columns=['timestamp'], inplace=True)
# %% DEFINE FUNCTIONS


def transform_genre_to_regex(genres):
    regex = ""
    for genre in genres:
        regex += f"(?=.*{genre})"
    return regex

# to make the the dataframe look nicer
def make_pretty(styler):
    styler.set_caption("Top movie recommendations for you")
    # styler.background_gradient(cmap="YlGnBu")
    return styler

import requests
from bs4 import BeautifulSoup
def add_image_link(movies):
    cover_pic=[]
    imdb_links =[]
    for index,movie in movies.iterrows():
        imdb_url = "https://www.imdb.com"
        imdb_search_url = f"/find?q={movie.title}"
        imdb_r = requests.get(imdb_url + imdb_search_url)
        imdb_soup = BeautifulSoup(imdb_r.content, "html.parser") #convert the response to BeautifulSoup variable
        try: movie_page = imdb_soup.select("div.article table.findList tr.findResult.odd td.primary_photo a")[0]['href']
        except: movie_page = 'Unknown' # managing error, when ther is no mayor name
        imdb_pic_r = requests.get(imdb_url+movie_page)
        imdb_pic_soup =  BeautifulSoup(imdb_pic_r.content, "html.parser") #convert the response to BeautifulSoup variable
        try: pic_page=imdb_pic_soup.select("#__next > main > div > section.ipc-page-background.ipc-page-background--base.sc-ca85a21c-0.efoFqn > section > div:nth-child(4) > section > section > div.sc-2a827f80-2.kqTacj > div.sc-2a827f80-3.dhWlsy > div > div.sc-77a2c808-2.mcnrT > div > div > a")[0]['href']
        except :
            cover_pic.append('https://i.stack.imgur.com/6M513.png')
            imdb_links.append(imdb_url+movie_page)
            continue
        pic_href_r = requests.get(imdb_url+pic_page)
        pic_href_soup =  BeautifulSoup(pic_href_r.content, "html.parser")
        pic_link = pic_href_soup.select("div.sc-7c0a9e7c-2.bkptFa img")[0]['src']
        cover_pic.append(pic_link)
        imdb_links.append(imdb_url+movie_page)
    movies['cover_pic'] = cover_pic
    movies['imdb_link'] = imdb_links
    return movies 

def test(movie = "Toy Story (1995)"):
    imdb_url = "https://www.imdb.com"
    imdb_search_url = f"/find?q={movie}"
    imdb_r = requests.get(imdb_url + imdb_search_url)
    imdb_soup = BeautifulSoup(imdb_r.content, "html.parser") #convert the response to BeautifulSoup variable
    try: movie_page = imdb_soup.select("div.article table.findList tr.findResult.odd td.primary_photo a")[0]['href']
    except: movie_page = 'Unknown' # managing error, when ther is no mayor name
    imdb_pic_r = requests.get(imdb_url+movie_page)
    imdb_pic_soup =  BeautifulSoup(imdb_pic_r.content, "html.parser") #convert the response to BeautifulSoup variable
    pic_page=imdb_pic_soup.select("div.sc-77a2c808-2.mcnrT div div a")#[1]['href']
    # pic_href_r = requests.get(imdb_url+pic_page)
    # pic_href_soup =  BeautifulSoup(pic_href_r.content, "html.parser")
    # pic_link = pic_href_soup.select("div.sc-7c0a9e7c-2.bkptFa img")[0]['src']
    return pic_page 

# population based
def popular_top_n(n, genres,time_period):
    popular_n = (
    rating_df
            .groupby(by='movieId')
            .agg(rating_mean=('rating', 'mean'), rating_count=('movieId', 'count'), datetime=('datetime','mean'))
        #     .sort_values(['rating_mean','rating_count','datetime'], ascending= False)
        #     .loc[lambda df_ :df_['rating_count'] >= (df_['rating_count'].mean()+df_['rating_count'].median())/2]
            .assign(overall_rating = lambda df_ : (df_['rating_mean']+df_['rating_count'] * 5* 10 / df_['rating_count'].max()) )
            .sort_values('overall_rating', ascending= False)
            .reset_index(drop= True)
    )
    top_n = popular_n.merge(movie_df,how='right', left_index=True, right_on="movieId")
    top_n = top_n.loc[lambda df_ : ((df_['year'] >= time_period[0]) & ( df_['year'] <= time_period[1]))]
    if 'Any' in genres: genres.remove('Any')
    genres_regex = transform_genre_to_regex(genres)
    top_n = top_n.loc[lambda df_ : df_['genres'].str.contains(genres_regex)]
    top_n.sort_values('overall_rating', ascending=False)
    top_n = top_n.drop(columns=['rating_count', 'overall_rating', 'datetime']).reset_index( drop= True).head(n)
    result_size = top_n.shape[0]
    new_index = ['movie-{}'.format(i+1) for i in range(result_size)]
    top_n.index = new_index
    pretty_rec = top_n.style.pipe(make_pretty)
    return top_n

# movie/item based
def item_n_movies(target_name , n , genres, time_period):
    #check the movie input
    target_Id = movie_df.loc[lambda df_ : df_['title'].str.lower() == target_name.lower(), 'movieId']
    if target_Id.empty: 
        return pd.DataFrame(columns= ['movieId', 'title', 'genres', 'year'])
    target_Id = int(target_Id)
    
    movie_user_matrix = (
                rating_df
                    .pivot_table(index='movieId', columns='userId', values='rating')
                    .fillna(0)
                )
    similarities_movies = pd.DataFrame(cosine_similarity(movie_user_matrix),
                                  index=movie_user_matrix.index,
                                  columns=movie_user_matrix.index)
    similarities = pd.DataFrame(
        (
        similarities_movies
                    .query("index != @target_Id")[target_Id] / sum(similarities_movies.query("index != @target_Id")[target_Id]))
                    .sort_values(ascending= False)
                    
        )
    recommendations = similarities.merge(movie_df, how= 'left', left_index = True, right_on = 'movieId')
    rating_n =(
                rating_df
                    .groupby(by='movieId')
                    .agg(rating_count=('userId', 'count'))
                    .reset_index()
        )
    recommendations = recommendations.join(rating_n[['rating_count']])
    recommendations = recommendations.loc[lambda df_ : df_['rating_count']>=3]
    recommendations = recommendations.loc[lambda df_ : ((df_['year'] >= time_period[0]) & ( df_['year'] <= time_period[1]))]
    if 'Any' in genres: genres.remove('Any')
    genres_regex = transform_genre_to_regex(genres)
    recommendations = recommendations.loc[lambda df_ : df_['genres'].str.contains(genres_regex)]
    top_n = recommendations.head(n)
    result_size = top_n.shape[0]
    new_index = ['movie-{}'.format(i+1) for i in range(result_size)]
    top_n.index = new_index
    pretty_rec = top_n.style.pipe(make_pretty)
    return pretty_rec

# user based
def user_n_movies(user_id , n , genres, time_period):
    if user_id not in rating_df["userId"]: 
        return pd.DataFrame(columns= ['movieId', 'title', 'genres', 'year'])
    
    users_items = pd.pivot_table(data=rating_df, 
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
    new_ratings = rating_df.loc[lambda df_ : df_['userId'].isin(new_userids)]
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
    recommendations = weighted_averages.merge(movie_df, left_index=True, right_on="movieId").sort_values("predicted_rating", ascending=False)
    recommendations = recommendations.loc[lambda df_ : ((df_['year'] >= time_period[0]) & ( df_['year'] <= time_period[1]))]
    if 'Any' in genres: genres.remove('Any')
    genres_regex = transform_genre_to_regex(genres)
    recommendations = recommendations.loc[lambda df_ : df_['genres'].str.contains(genres_regex)]             
    top_n = recommendations.reset_index(drop=True).drop(columns=['predicted_rating']).head(n)
    result_size = top_n.shape[0]
    new_index = ['movie-{}'.format(i+1) for i in range(result_size)]
    top_n.index = new_index
    pretty_rec = top_n.style.pipe(make_pretty)
    return pretty_rec


# %% STREAMLIT
# Set configuration
st.set_page_config(page_title="WBSFLIX",
                   page_icon="🎬",
                   initial_sidebar_state="expanded",
                   layout="wide"
                   )

# set colors: These has to be set on the setting menu online
    # primary color: #FF4B4B, background color:#0E1117
    # text color: #FAFAFA, secondary background color: #E50914

# Set the logo of app
st.sidebar.image("wbs_logo.png",
                 width=300, clamp=True)
welcome_img = Image.open('welcome_page_img01.png')
st.image(welcome_img)
st.sidebar.markdown("""
# 🎬 Welcome to the next generation movie recommendation app
""")

# %% APP WORKFLOW
st.sidebar.markdown("""
### How may we help you?
"""
)
# Popularity based recommender system
genre_default = None
pop_based_rec = st.sidebar.checkbox("Show me the all time favourites",
                            False,
                            help="Movies that are liked by many people")


if pop_based_rec:
    st.markdown("### Select the Genre and the Number of recommendations")
    genre_default, n_default = None, 5
    with st.form(key="pop_form"):
        genre_default = ['Any']
        genre = st.multiselect(
                "Genre",
                options=genre_list,
                help="Select the genre of the movie you would like to watch",
                default=genre_default)

        nr_rec = st.slider("Number of recommendations",
                        min_value=1,
                        max_value=20,
                        value=5,
                        step=1,
                        key="n",
                        help="How many movie recommendations would you like to receive?",
                        )
        time_period = st.slider('years:', min_value=1900,
                                max_value=2018,
                                value=(2010,2018),
                                step=1)


        submit_button_pop = st.form_submit_button(label="Submit")


    if submit_button_pop:
        popular_movie_recs = popular_top_n(nr_rec, genre, time_period)
        st.table(popular_movie_recs)
        for index, movie in add_image_link(popular_movie_recs.reset_index(drop=True)).iterrows():
            st.image(movie['cover_pic'], width=300)
            st.write(f"[imdb link for: {movie['title']}]({movie['imdb_link']})")
# to put some space in between options
st.write("")
st.write("")
st.write("")

item_based_rec = st.sidebar.checkbox("Show me a movie like this",
                             False,
                             help="Input some movies and we will show you similar ones")

if item_based_rec:
    st.markdown("### Tell us a movie you like:")
    with st.form(key="movie_form"):
        movie_name = st.multiselect(label="Movie name",
                                    # options=movie_list,
                                    options=pd.Series(movie_list),
                                    help="Select a movie you like",
                                    key='item_select',
                                    default= 'Toy Story 2 (1999)'
                                    )
        genre_default = ['Any']
        genre = st.multiselect(
                "Genre",
                options=genre_list,
                help="Select the genre of the movie you would like to watch",
                default=genre_default)

        nr_rec = st.slider("Number of recommendations",
                           min_value=1,
                           max_value=20,
                           value=5,
                           step=1,
                           key="nr_rec_movie",
                           help="How many movie recommendations would you like to receive?",
                           )
        time_period = st.slider('years:', min_value=1900,
                                max_value=2018,
                                value=(2010,2018),
                                step=1)

        submit_button_movie = st.form_submit_button(label="Submit")

    if submit_button_movie:
        st.write('Because you like {}:'.format(movie_name[0]))

        item_movie_recs = item_n_movies(movie_name[0], nr_rec, genre, time_period)
        st.table(item_movie_recs)

# to put some space in between options
st.write("")
st.write("")
st.write("")

user_based_rec = st.sidebar.checkbox("I want to get personalized recommendations",
                             False,
                             help="Login to get personalized recommendations")

if user_based_rec:
    st.markdown("### Please login to get customized recommendations just for you")
    genre_default, n_default = None, 5
    with st.form(key="user_form"):

        user_id = st.number_input("Please enter your user id", step=1,
                                  min_value=1)
        genre_default = ['Any']
        genre = st.multiselect(
                "Genre",
                options=genre_list,
                help="Select the genre of the movie you would like to watch",
                #default=genre_default
                )

        nr_rec = st.slider("Number of recommendations",
                           min_value=1,
                           max_value=20,
                           value=5,
                           step=1,
                           key="nr_rec",
                           help="How many movie recommendations would you like to receive?",
                           )

        time_period = st.slider('years:', min_value=1900,
                                max_value=2018,
                                value=(2010,2018),
                                step=1)

        submit_button_user = st.form_submit_button(label="Submit")

    if submit_button_user:
        # user_movie_recs = user_n_movies(user_id, nr_rec)
        user_movie_recs = user_n_movies(user_id, nr_rec, genre, time_period)

        # st.write(time_period)
        st.table(user_movie_recs)


