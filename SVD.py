# coding=utf-8
import pandas as pd
import numpy as np
import os
from surprise import Reader, Dataset
from surprise import NormalPredictor, BaselineOnly
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise import SVD, SVDpp, NMF
from sklearn.metrics import mean_squared_error


def algo_predict(algo, user_id, movie_id):
    return algo.predict(user_id, movie_id).est

def CF(df_train, df_test):

    ###################### train ######################
    reader = Reader()
    algo = BaselineOnly()
    data = Dataset.load_from_df(df_train[['User', 'Movie', 'Rating']], reader)

    # train 1: cross_validate
    # model_selection.cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

    # train 2: not cross_validate
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    ###################### test ######################
    df_test['Predict_Score'] = df_test.apply(lambda row: algo_predict(algo, row['User'], row['Movie']), axis=1)
    df_test.to_csv('df_test.csv')
    rmse = np.sqrt(mean_squared_error(df_test['Predict_Score'], df_test['Rating']))
    print('\n\nTesting Result: {:.4f} RMSE'.format(rmse))


if __name__ == '__main__':

    df = pd.read_csv('/home/zwj/Desktop/recommend/netflix_prize_data/netflix_data_4178032.csv', usecols=[1, 2, 4])
    m = 400000
    df_train = df[0:-m]
    df_test = df[-m:]

    print('df Shape: {}, trainset: {}, testset: {}'.format(df.shape, len(df_train), len(df_test)))
    print(df.sample(5))
    print('-'*50)

    CF(df_train, df_test)
