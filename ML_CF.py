# coding=utf-8
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt
import time
from sklearn.utils import shuffle
from keras.layers import Input, Embedding, Reshape, Dot, Concatenate, Dense, Dropout
from keras.models import Model
from sklearn.linear_model import LinearRegression, LogisticRegression

# 逻辑回归
def CF_LogisticRegression(df_train, df_test):

	# 训练模型
    for i, C in enumerate((1, 0.1, 0.01)):
        model = LogisticRegression(multi_class="ovr",C=C,penalty="l2",solver="lbfgs",tol=0.01)
        model.fit(np.array(df_train.loc[:, ['User', 'Movie']], dtype=np.int64), np.array(df_train.loc[:, 'Rating'], dtype=np.int64))

	# 测试
    df_test['Predict_Score'] = model.predict(np.array(df_test.loc[:, ['User', 'Movie']], dtype=np.int64))
	
	# 计算误差
    rmse = np.sqrt(mean_squared_error(df_test['Predict_Score'], df_test['Rating']))

    print('\n\nTesting Result With LogisticRegression: {:.4f} RMSE'.format(rmse))

# 线性回归
def CF_LinearRegression(df_train, df_test):

	# 训练模型
    model = LinearRegression()
    model.fit(np.array(df_train.loc[:, ['User', 'Movie']], dtype=np.int64), np.array(df_train.loc[:, 'Rating'], dtype=np.float64))

	# 测试
    df_test['Predict_Score'] = model.predict(np.array(df_test.loc[:, ['User', 'Movie']], dtype=np.int64))
	
	#计算误差
    rmse = np.sqrt(mean_squared_error(df_test['Predict_Score'], df_test['Rating']))

    print('\n\nTesting Result With LinearRegression: {:.4f} RMSE'.format(rmse))

# 均值法：将测试集预测评分置为全局平均评分
def CF_mean(df_train, df_test):

    # 计算所有电影的平均评分
    ratings_mean = df_train.mean(axis=0).sort_values(ascending=False).rename('Rating-Mean').to_frame()

	# 给测试集预测评分赋值
    sample_num = len(df_test)
    df_test['Predict_Score'] = np.full((sample_num,), ratings_mean.loc['Rating'][0])

	#计算误差
    rmse = np.sqrt(mean_squared_error(df_test['Predict_Score'], df_test['Rating']))

    print('\n\nTesting Result With Mean-Rating: {:.4f} RMSE'.format(rmse))

# 神经网络算法
def MLP(df, df_train, df_test):

    # 创建 user- & movie-id 映射
    user_id_mapping = {id:i for i, id in enumerate(df['User'].unique())}
    movie_id_mapping = {id:i for i, id in enumerate(df['Movie'].unique())}

    # 将训练集和测试集的id设置为索引号
    train_user_data = df_train['User'].map(user_id_mapping)
    train_movie_data = df_train['Movie'].map(movie_id_mapping)

    test_user_data = df_test['User'].map(user_id_mapping)
    test_movie_data = df_test['Movie'].map(movie_id_mapping)

    # 获取输入参数的size
    users = len(user_id_mapping)
    movies = len(movie_id_mapping)

    user_embedding_size = 20
    movie_embedding_size = 10


    ##### 创建模型Create model
    # 设置input层
    user_id_input = Input(shape=[1], name='user')
    movie_id_input = Input(shape=[1], name='movie')

    # 为users和movies创建embedding层
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

	# 搭建网络
	# 方式1：一层网络
    # dense = Dense(256)(concat)
    # y = Dense(1)(dense)

    # 方式2：两层网络
    layer_1 = Dense(256)(concat)
    layer_2 = Dense(512)(layer_1)
    layer_3 = Dropout(0.5)(layer_2)
    y = Dense(1)(layer_3)

    # 设置模型参数
    model = Model(inputs=[user_id_input, movie_id_input], outputs=y)
	
	# 编译
    model.compile(loss='mse', optimizer='adam')


    # 训练模型
    model.fit([train_user_data, train_movie_data],
              df_train['Rating'],
              batch_size=256,
              epochs=10,
              validation_split=0.1,
              shuffle=True)

    # 测试
    y_pred = model.predict([test_user_data, test_movie_data])
    y_true = df_test['Rating'].values

    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
    print('\n\nTesting Result With Keras Deep Learning: {:.4f} RMSE'.format(rmse))

# 矩阵分解算法
def  Matrix_Factorisation(df, df_train, df_test):
    # 创建 user- & movie-id 映射
    user_id_mapping = {id:i for i, id in enumerate(df['User'].unique())}
    movie_id_mapping = {id:i for i, id in enumerate(df['Movie'].unique())}


    # 将训练集和测试集的id设置为索引号
    train_user_data = df_train['User'].map(user_id_mapping)
    train_movie_data = df_train['Movie'].map(movie_id_mapping)

    test_user_data = df_test['User'].map(user_id_mapping)
    test_movie_data = df_test['Movie'].map(movie_id_mapping)


    # 获取input参数size
    users = len(user_id_mapping)
    movies = len(movie_id_mapping)
    embedding_size = 10


    ##### 创建模型
    # 设置input层
    user_id_input = Input(shape=[1], name='user')
    movie_id_input = Input(shape=[1], name='movie')

    # 为users和movies创建embedding层
    user_embedding = Embedding(output_dim=embedding_size,
                               input_dim=users,
                               input_length=1,
                               name='user_embedding')(user_id_input)
    movie_embedding = Embedding(output_dim=embedding_size,
                                input_dim=movies,
                                input_length=1,
                                name='item_embedding')(movie_id_input)

    # Reshape the embedding layers
    user_vector = Reshape([embedding_size])(user_embedding)
    movie_vector = Reshape([embedding_size])(movie_embedding)

    # 计算reshaped embedding layers点乘结果作为预测结果
    y = Dot(1, normalize=False)([user_vector, movie_vector])

    # 设置模型参数
    model = Model(inputs=[user_id_input, movie_id_input], outputs=y)
	
	# 编译
    model.compile(loss='mse', optimizer='adam')


    # 训练模型
    model.fit([train_user_data, train_movie_data],
              df_train['Rating'],
              batch_size=256,
              epochs=1,
              validation_split=0.1,
              shuffle=True)

    # 测试模型
    y_pred = model.predict([test_user_data, test_movie_data])
    y_true = df_test['Rating'].values

    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
    print('\n\nTesting Result With Keras Matrix-Factorization: {:.4f} RMSE'.format(rmse))

if __name__ == '__main__':

	# 读取数据
    df = pd.read_csv('/home/zwj/Desktop/recommend/netflix_prize_data/netflix_data_4178032.csv', usecols = [1, 2, 4])
    
	# 设置测试集样本数，拆分训练集和测试集
	m = 400000
    df_train = df[0:-m]
    df_test = df[-m:]

    print('df Shape: {}, trainset: {}, testset: {}'.format(df.shape, len(df_train), len(df_test)))

    # CF_mean(df_train, df_test)
    # MLP(df, df_train, df_test)
    # Matrix_Factorisation(df, df_train, df_test)
    # CF_LinearRegression(df_train, df_test)
    # CF_LogisticRegression(df_train, df_test)
