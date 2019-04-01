# coding=utf-8
import pandas as pd
import numpy as np
import os
from surprise import Reader, Dataset
from surprise import NormalPredictor, BaselineOnly
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise import SVD, SVDpp, NMF, model_selection
from sklearn.metrics import mean_squared_error


def algo_predict(algo, user_id, movie_id):
    return algo.predict(user_id, movie_id).est

# SVD训练和预测
def CF(df_train, df_test):

    ###################### train ######################
	# 读取数据
    reader = Reader()
    algo = SVD()
    data = Dataset.load_from_df(df_train[['User', 'Movie', 'Rating']], reader)

	# 训练模型
    # 方式 1: 交叉验证
	# (算法, 数据, loss计算方式， CV=交叉验证次数
    # model_selection.cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

    # 方式 2: 没有交叉验证
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    ###################### test ######################
	# 预测
    df_test['Predict_Score'] = df_test.apply(lambda row: algo_predict(algo, row['User'], row['Movie']), axis=1)
	# 计算RMSE
    rmse = np.sqrt(mean_squared_error(df_test['Predict_Score'], df_test['Rating']))
    print('\n\nTesting Result: {:.4f} RMSE'.format(rmse))


if __name__ == '__main__':

    df = pd.read_csv('/home/zwj/Desktop/recommend/netflix_prize_data/netflix_data_4178032.csv', usecols=[1, 2, 4])
    m = 400000
    df_train = df[0:-m]
    df_test = df[-m:]

    print('df Shape: {}, trainset: {}, testset: {}'.format(df.shape, len(df_train), len(df_test)))

    CF(df_train, df_test)
