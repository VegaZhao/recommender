# coding=utf-8
import pandas as pd
import random
import numpy as np
import os
from sklearn.utils import shuffle

file_dir = '/root/vega/data/normalize_data/'

movie_set = set()
user_set = set()
for i in range(1, 6):
    print('read data_tmp_' + str(i) + '.csv')
    df = pd.read_csv(file_dir + 'data_tmp_' + str(i) + '.csv', usecols=[1, 2, 3, 4])

    if i == 1:
        df_data = df
    else:
        df_data = df_data.append(df)

del df
print('Shape User-Ratings raw data:\t{}'.format(df_data.shape))

print('original user length: ', len(df_data['User'].unique()))
print('original movie length: ', len(df_data['Movie'].unique()))

movie_list = df_data['Movie'].unique()
user_list = df_data['User'].unique()

save_user = set()
random.seed(5)
for i in range(len(user_list)):
    if random.randint(1,100) == 1:
        save_user.add(user_list[i])

print('len(save_user) = ', len(save_user))

df_filterd_user = df_data[df_data['User'].isin(save_user)]

print('Shape df_filterd_user data:\t{}'.format(df_filterd_user.shape))
print('df_filterd_user user length: ', len(df_filterd_user['User'].unique()))
print('df_filterd_user movie length: ', len(df_filterd_user['Movie'].unique()))

del df_data

# Filter sparse movies
min_movie_ratings = 400
filter_movies = (df_filterd_user['Movie'].value_counts()>min_movie_ratings)
filter_movies = filter_movies[filter_movies].index.tolist()

df_filterd_min_movies = df_filterd_user[df_filterd_user['Movie'].isin(filter_movies)]

# Filter sparse users
min_user_ratings = 20
filter_users_min = (df_filterd_min_movies['User'].value_counts()>min_user_ratings)
filter_users_min = filter_users_min[filter_users_min].index.tolist()

df_filterd_min_user = df_filterd_min_movies[df_filterd_min_movies['User'].isin(filter_users_min)]

# Filter sparse users
max_user_ratings = 200
filter_users_max = (df_filterd_min_user['User'].value_counts()<max_user_ratings)
filter_users_max = filter_users_max[filter_users_max].index.tolist()

df_filterd = df_filterd_min_user[df_filterd_min_user['User'].isin(filter_users_max)]
print('Shape User-Ratings filtered:\t{}'.format(df_filterd.shape))

del df_filterd_user, df_filterd_min_movies, df_filterd_min_user

print('user length: ', len(df_filterd['User'].unique()))
print('movie length: ', len(df_filterd['Movie'].unique()))


movieCount = df_filterd.groupby('Movie')['Rating'].count().sort_values(ascending = False).to_frame(name='count')
print(movieCount)
print('-'*100)
userCount = df_filterd.groupby('User')['Rating'].count().sort_values(ascending = False).to_frame(name='count')
print(userCount)

df_filterd.to_csv('/root/vega/data/normalize_data/filterd_u3065_m374.csv')