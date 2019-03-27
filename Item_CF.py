#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

# 读取demo数据集
# 格式：
#     Movie  User  Rating
# 0       1   101       4
# 1       1   102       3
# 2       1   103       4
df_data = pd.read_csv('/home/zwj/Desktop/recommend/uus_CF_demo.txt', header=None, names=['Movie', 'User', 'Rating'])

# 将数据集拆分为训练集和测试集，后三条为测试数据，其余的为训练数据
df_train = df_data[0:-3]
df_test = df_data[-3:]

# 将训练集数据转换成透视表
# 格式：
# Shape User-Movie-Matrix:	(6, 7)
# User   101  102  103  104  105  106  107
# Movie
# 1      4.0  3.0  4.0  NaN  4.0  3.0  NaN
# 2      3.0  2.0  4.0  3.0  NaN  3.0  NaN
# 3      4.0  3.0  3.0  NaN  5.0  4.0  4.0
# 4      5.0  NaN  4.0  5.0  5.0  NaN  5.0
# 5      NaN  4.0  NaN  4.0  NaN  NaN  4.0
# 6      NaN  3.0  NaN  4.0  NaN  NaN  NaN
df_p = df_train.pivot_table(index='Movie', columns='User', values='Rating')
print('Shape User-Movie-Matrix:\t{}'.format(df_p.shape))

# 设置用于推荐的相似电影数
n_recommendation = 3

# 补充透视表中缺失的值，补充的值为电影已有评价的均值
# User   101  102  103       104  105  106  107
# Movie
# 1      4.0  3.0  4.0  3.600000  4.0  3.0  3.6
# 2      3.0  2.0  4.0  3.000000  3.0  3.0  3.0
# 3      4.0  3.0  3.0  3.833333  5.0  4.0  4.0
# 4      5.0  4.8  4.0  5.000000  5.0  4.8  5.0
# 5      4.0  4.0  4.0  4.000000  4.0  4.0  4.0
# 6      3.5  3.0  3.5  4.000000  3.5  3.5  3.5
df_p_imputed = df_p.T.fillna(df_p.mean(axis=1)).T

# 创建 movie-id 映射
# {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
movie_id_mapping = {id:i for i, id in enumerate(df_p_imputed.index)}


# 计算所有电影之间的相似度
# [[1.    0.991 0.986 0.99  0.993 0.994]
#  [0.991 1.    0.971 0.974 0.984 0.988]
#  [0.986 0.971 1.    0.991 0.987 0.988]
#  [0.99  0.974 0.991 1.    0.998 0.995]
#  [0.993 0.984 0.987 0.998 1.    0.997]
#  [0.994 0.988 0.988 0.995 0.997 1.   ]]
similarity = cosine_similarity(df_p_imputed.values)

# 在相似度矩阵中去掉自身相似度
# [[ 0.     0.991  0.986  0.99   0.993  0.994]
#  [ 0.991 -0.     0.971  0.974  0.984  0.988]
#  [ 0.986  0.971  0.     0.991  0.987  0.988]
#  [ 0.99   0.974  0.991  0.     0.998  0.995]
#  [ 0.993  0.984  0.987  0.998 -0.     0.997]
#  [ 0.994  0.988  0.988  0.995  0.997  0.   ]]
similarity -= np.eye(similarity.shape[0])

###################################推荐单个用户####################################
# 设置推荐用户在数据集中的索引，本例中设置为训练集透视表中的第一位用户User：101
user_index = 0

# 找到该用户（101）没有打分的电影
# Int64Index([5, 6], dtype='int64', name=u'Movie')
unrated_movies = df_p.T.iloc[user_index][df_p.T.iloc[user_index].isna()].index


# 创建prediction列表，存储预测结果
prediction = []

# 遍历该用户所有没有打分的电影
for item_id in unrated_movies:
    # 已 item_id = 5 为例
    # 电影相似度排序（索引）
    # [3 5 0 2 1 4]
    similar_item_index = np.argsort(similarity[movie_id_mapping[item_id]])[::-1]

    # 电影相似度排序（值）
    # [ 0.998  0.997  0.993  0.987  0.984 -0.   ]
    similar_item_score = np.sort(similarity[movie_id_mapping[item_id]])[::-1]

    # 计算前n部最相似电影的带权重的评分
    # 1.前n部最相似电影的所有用户评分
    # User   101  102  103  104  105  106  107
    # Movie
    # 4      5.0  4.8  4.0  5.0  5.0  4.8  5.0
    # 6      3.5  3.0  3.5  4.0  3.5  3.5  3.5
    # 1      4.0  3.0  4.0  3.6  4.0  3.0  3.6
    # ==============================
    # 2.找到改用户对应评分
    # Movie
    # 4    5.0
    # 6    3.5
    # 1    4.0
    # Name: 101, dtype: float64
    # ==============================
    # 3.前n部最相似电影的相似度 对应电影为 4 6 1
    # [0.998 0.997 0.993]
    # ==============================
    # 4.前n部最相似电影的相似度之和
    # 2.9880774074123364
    # ==============================
    # 计算预测评分：计算前n部最相似电影的带权重的评分
    # 4.166990457044285
    score = (df_p_imputed.iloc[similar_item_index[:n_recommendation]][df_p_imputed.columns[user_index]] * similar_item_score[:n_recommendation]).values.sum() / similar_item_score[:n_recommendation].sum()
    # 合并每次的预测结果
    # [[101, 5, 4.166990457044285], [101, 6, 4.333324910043814]]
    prediction.append([df_p_imputed.columns[user_index], item_id, score])

movie_recommendations = pd.DataFrame(prediction, columns=['User', 'Movie', 'Rating'])

# 推荐电影按分值由高到低排列
#    Movie    Rating
# 1      6  4.333325
# 0      5  4.166990
best_movie_recommendations = movie_recommendations.sort_values(by='Rating', ascending=False).loc[:, ['Movie', 'Rating']]

###################################批量预测用户电影评分####################################

# 创建prediction列表，存储预测结果
prediction = []
# 遍历测试集中的所有电影
for movie_id in df_test['Movie'].unique():

    # 电影相似度排序（索引）
    similar_item_index = np.argsort(similarity[movie_id_mapping[movie_id]])[::-1]

    # 电影相似度排序（值）
    similar_item_score = np.sort(similarity[movie_id_mapping[movie_id]])[::-1]

    # 遍历测试集中该电影对应的所有用户
    for user_id in df_test[df_test['Movie']==movie_id]['User'].values:

        # 计算预测评分：计算前n部最相似电影的带权重的评分
        score = (df_p_imputed.iloc[similar_item_index[:n_recommendation]][user_id] * similar_item_score[:n_recommendation]).values.sum() / similar_item_score[:n_recommendation].sum()

        # 合并每次的预测结果
		# [[105, 6, 4.333324910043814], [106, 6, 3.9338515239279848], [107, 6, 4.200201548400988]]
        prediction.append([user_id, movie_id, score])


# 将prediction列表转成DataFrame
#             Prediction
# User Movie
# 105  6        4.333325
# 106  6        3.933852
# 107  6        4.200202
df_pred = pd.DataFrame(prediction, columns=['User', 'Movie', 'Prediction']).set_index(['User', 'Movie'])

# 将预测结果表df_pred与测试表df_test根据列['User', 'Movie']自然连接
#             Rating  Prediction
# User Movie
# 105  6           5    4.333325
# 106  6           4    3.933852
# 107  6           4    4.200202
df_pred = df_test.set_index(['User', 'Movie']).join(df_pred)

# 获取labels 与 predictions
y_true = df_pred['Rating'].values
y_pred = df_pred['Prediction'].values

# 计算 RMSE
rmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))

print('\n\nTesting Result With Cosine User-User Similarity: {:.4f} RMSE'.format(rmse))
