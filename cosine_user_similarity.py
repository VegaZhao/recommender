#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

# 读取demo数据集
# 格式：
#     Movie  User  Rating
# 0          1      101       4
# 1          1      102       3
# 2          1      103       4
df_data = pd.read_csv('/home/zwj/Desktop/recommend/uus_CF_demo.txt', header=None, names=['Movie', 'User', 'Rating'])

# 将数据集拆分为训练集和测试集，后三条为测试数据，其余的为训练数据
df_train = df_data[0:-3]
df_test = df_data[-3:]

# 将训练集数据转换成透视表
# 格式：
# Shape User-Movie-Matrix:	(7, 6)
# Movie    1    2    3    4    5    6
# User
# 101       4.0  3.0  4.0  5.0  NaN  NaN
# 102       3.0  2.0  3.0  NaN  4.0  3.0
# 103       4.0  4.0  3.0  4.0  NaN  NaN
# 104       NaN  3.0  NaN  5.0  4.0  4.0
# 105       4.0  NaN  5.0  5.0  NaN  NaN
# 106       3.0  3.0  4.0  NaN  NaN  NaN
# 107       NaN  NaN  4.0  5.0  4.0  NaN
df_p = df_train.pivot_table(index='User', columns='Movie', values='Rating')
print('Shape User-Movie-Matrix:\t{}'.format(df_p.shape))

###################################推荐单个用户####################################
# 设置推荐用户在数据集中的索引，本例中设置为训练集透视表中的第一位用户User：101
user_index = 0

# 设置用于推荐的相似用户数
n_recommendation = 3

# 补充透视表中缺失的值，补充的值为用户的已评价电影的均值
# Movie         1         2    3         4         5         6
# User
# 101       4.000000  3.000000  4.0  5.000000  4.000000  4.000000
# 102       3.000000  2.000000  3.0  3.000000  4.000000  3.000000
# 103       4.000000  4.000000  3.0  4.000000  3.750000  3.750000
# 104       4.000000  3.000000  4.0  5.000000  4.000000  4.000000
# 105       4.000000  4.666667  5.0  5.000000  4.666667  4.666667
# 106       3.000000  3.000000  4.0  3.333333  3.333333  3.333333
# 107       4.333333  4.333333  4.0  5.000000  4.000000  4.333333
df_p_imputed = df_p.T.fillna(df_p.mean(axis=1)).T

# 计算所有用户之间的相似度
# [[1.    0.985 0.985 1.    0.99  0.989 0.993]
#  [0.985 1.    0.974 0.985 0.979 0.983 0.975]
#  [0.985 0.974 1.    0.985 0.99  0.982 0.997]
#  [1.    0.985 0.985 1.    0.99  0.989 0.993]
#  [0.99  0.979 0.99  0.99  1.    0.997 0.995]
#  [0.989 0.983 0.982 0.989 0.997 1.    0.99 ]
#  [0.993 0.975 0.997 0.993 0.995 0.99  1.   ]]
similarity = cosine_similarity(df_p_imputed.values)

# 在相似度矩阵中去掉自身相似度
# [[-0.    0.99  0.99  1.    0.99  0.99  0.99]
#  [ 0.99  0.    0.97  0.99  0.98  0.98  0.97]
#  [ 0.99  0.97  0.    0.99  0.99  0.98  1.  ]
#  [ 1.    0.99  0.99 -0.    0.99  0.99  0.99]
#  [ 0.99  0.98  0.99  0.99 -0.    1.    1.  ]
#  [ 0.99  0.98  0.98  0.99  1.   -0.    0.99]
#  [ 0.99  0.97  1.    0.99  1.    0.99  0.  ]]
similarity -= np.eye(similarity.shape[0])

# 用户相似度排序（索引）
# [3 6 4 5 1 2 0]
similar_user_index = np.argsort(similarity[user_index])[::-1]
# 用户相似度排序（值）
# [ 1.     0.993  0.99   0.989  0.985  0.985 -0.   ]
similar_user_score = np.sort(similarity[user_index])[::-1]

# 找到该用户（101）没有打分的电影
# Int64Index([5, 6], dtype='int64', name=u'Movie')
unrated_movies = df_p.iloc[user_index][df_p.iloc[user_index].isna()].index

# 计算前n位最相似用户的带权重的评分，并计算每部电影的平均得分
# 1.前n位最相似用户的所有电影评分
# User   104       107       105
# Movie
# 1         4.0  4.333333  4.000000
# 2         3.0  4.333333  4.666667
# 3         4.0  4.000000  5.000000
# 4         5.0  5.000000  5.000000
# 5         4.0  4.000000  4.666667
# 6         4.0  4.333333  4.666667
# ==============================
# 2.前n位最相似用户的相似度 对应用户为 104 107 105
# [1.    0.993 0.99 ]
# ==============================
# 3.每部电影前n位最相似用户评分的加权求和
# Movie
# 1    12.264332
# 2    11.924443
# 3    12.923447
# 4    14.916600
# 5    12.593391
# 6    12.924443
#
# ==============================
# 4.前n位最相似用户的相似度之和
# 2.983320098495457
# ==============================
# 5.每部电影前n位最相似用户评分的加权求和 / 前n位最相似用户的相似度之和
# Movie
# 1    4.110967
# 2    3.997038
# 3    4.331901
# 4    5.000000
# 5    4.221267
# 6    4.332235

mean_movie_recommendations = (df_p_imputed.iloc[similar_user_index[:n_recommendation]].T * similar_user_score[:n_recommendation]).sum(axis=1) / similar_user_score[:n_recommendation].sum()

# 过滤出没有评价过的电影，并且按评分降序排序
# Movie
# 6         4.332235
# 5         4.221267
best_movie_recommendations = mean_movie_recommendations[unrated_movies].sort_values(ascending=False).to_frame()

###################################批量预测用户电影评分####################################
# 创建 user-id 映射
# {101: 0, 102: 1, 103: 2, 104: 3, 105: 4, 106: 5, 107: 6}
user_id_mapping = {id:i for i, id in enumerate(df_p_imputed.index)}

# 创建prediction列表，存储预测结果
prediction = []
# 遍历测试集中的所有用户
for user_id in df_test['User'].unique():

    # 用户相似度排序（索引）
    similar_user_index = np.argsort(similarity[user_id_mapping[user_id]])[::-1]
    # 用户相似度排序（值）
    similar_user_score = np.sort(similarity[user_id_mapping[user_id]])[::-1]

	# 遍历测试集中该用户对应的所有电影
    for movie_id in df_test[df_test['User']==user_id]['Movie'].values:

        # 计算预测评分：计算前n位于该用户最相似用户的带权重的评分，并计算每部电影的平均得分
        score = (df_p_imputed.iloc[similar_user_index[:n_recommendation]][movie_id] * similar_user_score[:n_recommendation]).values.sum() / similar_user_score[:n_recommendation].sum()
		# 合并每次的预测结果
		# [[105, 6, 3.88835688945032], [106, 6, 4.334263060639755], [107, 6, 4.138804253587084]]
        prediction.append([user_id, movie_id, score])

# 将prediction列表转成DataFrame
#                   Prediction
# User Movie
# 105     6           3.888357
# 106     6           4.334263
# 107     6           4.138804
df_pred = pd.DataFrame(prediction, columns=['User', 'Movie', 'Prediction']).set_index(['User', 'Movie'])

# 将预测结果表df_pred与测试表df_test根据列['User', 'Movie']自然连接
#                   Rating  Prediction
# User Movie
# 105     6              5    3.888357
# 106     6              4    4.334263
# 107     6              4    4.138804
df_pred = df_test.set_index(['User', 'Movie']).join(df_pred)

# 获取labels 与 predictions
y_true = df_pred['Rating'].values
y_pred = df_pred['Prediction'].values

# 计算 RMSE
rmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))

print('\n\nTesting Result With Cosine User-User Similarity: {:.4f} RMSE'.format(rmse))

