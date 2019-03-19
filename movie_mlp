import pandas as pd
import numpy as np
import os
from surprise import Reader, Dataset, SVD, model_selection
from sklearn.metrics import mean_squared_error
from math import sqrt
import time
from sklearn.utils import shuffle
from keras.layers import Input, Embedding, Reshape, Dot, Concatenate, Dense, Dropout
from keras.models import Model


def mlp(path):

    df = pd.read_csv(path, usecols=[1, 2, 3])
    
    border_line = int(df.shape[0] / 5)
    df_train = df.iloc[0:-border_line]
    df_test = df.iloc[-border_line:]

    print('df Shape: {}, trainset: {}, testset: {}'.format(df.shape, len(df_train), len(df_test)))
    # Create user- & movie-id mapping
    user_id_mapping = {id:i for i, id in enumerate(df['Cust_Id'].unique())}
    movie_id_mapping = {id:i for i, id in enumerate(df['Movie_Id'].unique())}


    # Create correctly mapped train- & testset
    train_user_data = df_train['Cust_Id'].map(user_id_mapping)
    train_movie_data = df_train['Movie_Id'].map(movie_id_mapping)

    test_user_data = df_test['Cust_Id'].map(user_id_mapping)
    test_movie_data = df_test['Movie_Id'].map(movie_id_mapping)

    # Get input variable-sizes
    users = len(user_id_mapping)
    movies = len(movie_id_mapping)

    # Setup variables
    user_embedding_size = 20
    movie_embedding_size = 10


    ##### Create model
    # Set input layers
    user_id_input = Input(shape=[1], name='user')
    movie_id_input = Input(shape=[1], name='movie')

    # Create embedding layers for users and movies
    user_embedding = Embedding(output_dim=user_embedding_size,
                               input_dim=users,
                               input_length=1,
                               name='user_embedding')(user_id_input)
    movie_embedding = Embedding(output_dim=movie_embedding_size,
                                input_dim=movies,
                                input_length=1,
                                name='item_embedding')(movie_id_input)

    # Reshape the embedding layers
    user_vector = Reshape([user_embedding_size])(user_embedding)
    movie_vector = Reshape([movie_embedding_size])(movie_embedding)

    # Concatenate the reshaped embedding layers
    concat = Concatenate()([user_vector, movie_vector])

    # Combine with dense layers
    layer_1 = Dense(256)(concat)
    layer_2 = Dense(512)(layer_1)
    layer_3 = Dropout(0.5)(layer_2)
    y = Dense(1)(layer_3)

    # Setup model
    model = Model(inputs=[user_id_input, movie_id_input], outputs=y)
    model.compile(loss='mse', optimizer='adam')


    # Fit model
    model.fit([train_user_data, train_movie_data],
              df_train['Rating'],
              batch_size=256,
              epochs=3,
              validation_split=0.2,
              shuffle=True)

    # Test model
    y_pred = model.predict([test_user_data, test_movie_data])
    y_true = df_test['Rating'].values

    #  Compute RMSE
    rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
    print('\n\nTesting Result With Keras Deep Learning: {:.4f} RMSE'.format(rmse))



if __name__ == '__main__':
    file_dir = '/root/vicky/1000file_df_shuffle.csv'

    mlp(file_dir)

