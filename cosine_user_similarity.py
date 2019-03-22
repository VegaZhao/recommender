import pandas as pd

import numpy as np

from sklearn.metrics import mean_squared_error

from sklearn.metrics.pairwise import cosine_similarity


# read data

file_dir = '/media/4T/zwj/zksg/temp/1000file_df_shuffle.csv'

df = pd.read_csv(file_dir, usecols=[1, 2, 3])


# set train-test

border_line = int(df.shape[0] / 5)

df_train = df.iloc[0:-border_line]

df_test = df.iloc[-border_line:]

print('df Shape: {}, trainset: {}, testset: {}'.format(df.shape, len(df_train), len(df_test)))


# Create a user-movie matrix with empty values

df_p = df.pivot_table(index='Cust_Id', columns='Movie_Id', values='Rating')

print('Shape User-Movie-Matrix:\t{}'.format(df_p.shape))


# User index for recommendation
user_index = 0

# Number of similar users for recommendation
n_recommendation = 6

# Fill in missing values
df_p_imputed = df_p.T.fillna(df_p.mean(axis=1)).T

# Compute similarity between all users
similarity = cosine_similarity(df_p_imputed.values)

# Remove self-similarity from similarity-matrix
similarity -= np.eye(similarity.shape[0])

# Sort similar users by index
similar_user_index = np.argsort(similarity[user_index])[::-1]

# Sort similar users by score
similar_user_score = np.sort(similarity[user_index])[::-1]

# Get unrated movies
unrated_movies = df_p.iloc[user_index][df_p.iloc[user_index].isna()].index

# Weight ratings of the top n most similar users with their rating and compute the mean for each movie
mean_movie_recommendations = (df_p_imputed.iloc[similar_user_index[:n_recommendation]].T * similar_user_score[:n_recommendation]).T.mean(axis=0)

# Filter for unrated movies and sort results
best_movie_recommendations = mean_movie_recommendations[unrated_movies].sort_values(ascending=False).to_frame()

# Create user-id mapping
user_id_mapping = {id:i for i, id in enumerate(df_p_imputed.index)}

prediction = []
# Iterate over all testset items
for user_id in df_test['Cust_Id'].unique():

    # Sort similar users by index
    similar_user_index = np.argsort(similarity[user_id_mapping[user_id]])[::-1]
    # Sort similar users by score
    similar_user_score = np.sort(similarity[user_id_mapping[user_id]])[::-1]
    
    for movie_id in df_test[df_test['Cust_Id']==user_id]['Movie_Id'].values:
        # Compute predicted score
        score = (df_p_imputed.iloc[similar_user_index[:n_recommendation]][movie_id] * similar_user_score[:n_recommendation]).values.sum() / similar_user_score[:n_recommendation].sum()
        prediction.append([user_id, movie_id, score])

# Create prediction DataFrame
df_pred = pd.DataFrame(prediction, columns=['Cust_Id', 'Movie_Id', 'Prediction']).set_index(['Cust_Id', 'Movie_Id'])
df_pred = df_test.set_index(['Cust_Id', 'Movie_Id']).join(df_pred)

# Get labels and predictions
y_true = df_pred['Rating'].values
y_pred = df_pred['Prediction'].values

# Compute RMSE
rmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))

print('\n\nTesting Result With Consine_uus: {:.4f} RMSE'.format(rmse))


