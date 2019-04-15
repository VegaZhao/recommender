# coding=utf-8
import pandas as pd
import numpy as np
import os
from surprise import Reader, Dataset
from surprise import NormalPredictor, BaselineOnly
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise import SVD, SVDpp, NMF, model_selection
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import math
import operator


# 将列表转成user-item字典
def UserItemDict(data):
    user_item = {}
    for user, item, rate in data:
        if user not in user_item:
            user_item[user] = {}
        user_item[user].update({item : rate})
    return user_item

def predict(algo, user_id, movie_id):
    return algo.predict(user_id, movie_id).est

# 模型训练
def trainModel(df_train):

    ###################### train ######################
    # 读取数据
    reader = Reader()
    algo = SVDpp()
    data = Dataset.load_from_df(df_train[['User', 'Movie', 'Rating']], reader)

    # 训练模型
    # 方式 1: 交叉验证
    # (算法, 数据, loss计算方式， CV=交叉验证次数
    model_selection.cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

    # 方式 2: 没有交叉验证
    # trainset = data.build_full_trainset()
    # algo.fit(trainset)

    # return trained model
    return algo

# 获取全局热门电影
def getHotItem(df_train, N=5):
    # 输入：
    #	df_train: 训练数据集
    #	N：推荐的电影数
    # 输出：
    #	rank：字典，该用户的推荐电影列表 {user_id: {item_t:rate1, item_k:rate2}}
    item_count = df_train.groupby('Movie')['Rating'].count().sort_values(ascending=False)

    hot_rank = {}

    r = 0
    for item_id in item_count[0:N].index:
        hot_rank[item_id] = 1 - 0.01*r
        r += 1
    return hot_rank

def recommendation(model, user_item, user_id, item_set, R):
    # 输入：
    #   model: 训练的模型
    #	user_item: 训练集中user-item字典 {user1 : {item1 : rate1, item2 : rate2}, ...}}
    #	user_id：推荐的用户id
    #   item_set: 训练集中的电影集合
    #   R：推荐列表中电影个数
    # 输出：
    #	rank：字典，该用户的推荐电影列表 {item_t:ratet, item_k:ratek}

    # 存储用户推荐电影
    rank = {}
    # 开辟用户空子字典 ('rank: ', {user_id: {}})
    rank.setdefault(user_id, {})

    # 如果该用户不在训练集中，则推荐热门电影
    if user_id not in user_item:
        print('user {} not in trainset, give hot rank list'.format(user_id))
        rank[user_id] = hot_rank
    else:
        # 用户已观看的电影集合
        item_watched_list = user_item[user_id]

        for item_id in item_set:
            if item_id in item_watched_list:
                continue
            rank[user_id].setdefault(item_id, 0)

            # 将模型预测评分结果赋给rank[user_id][item_id]
            rank[user_id][item_id] = model.predict(user_id, item_id).est

    # 推荐列表按评分由高到低排序
    rank_sorted = {}
    rank_sorted[user_id] = sorted(rank[user_id].items(), key=operator.itemgetter(1), reverse=True)[0:R]

    return rank_sorted

# 准确度/召回评价
def precisionRecall(test, recommend):
    # 推荐列表命中数
    hit = 0
    # 召回率分母（测试集中用户观看电影数）
    n_recall = 0
    # 准确率分母（推荐列表中电影数）
    n_precision = 0

    user_num = 0
    for user, items in test.items():
        # 推荐电影列表
        reco_item = [item for item, score in recommend[user]]
        # 推荐电影命中列表
        hit_list = [item for item in reco_item if item in test[user]]

        user_num += 1
        hit += len(hit_list)
        n_recall += len(test[user])
        n_precision += len(reco_item)
    print('user_num: {}, hit: {}, n_recall: {}, n_precision: {} '.format(user_num, hit, n_recall, n_precision))

    return [hit / (1.0 * n_recall), hit / (1.0 * n_precision)]

# RMSE评价
def rmse(df_test, model):
    # 预测
    df_test['Predict_Score'] = df_test.apply(lambda row: predict(model, row['User'], row['Movie']), axis=1)
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(df_test['Predict_Score'], df_test['Rating']))

    return rmse

if __name__ == '__main__':

    # 读取数据,这是没有shuffle的数据
    df_data = pd.read_csv('/home/zwj/Desktop/RSAlgorithms-master/data/ft_ratings.txt', \
                          sep=' ', header=None, names=['User', 'Movie', 'Rating'])

    df = shuffle(df_data)
    print(df.sample(5))

    m = 30000
    df_train = df[0:m]
    df_test = df[m:]

    print('df Shape: {}, trainset: {}, testset: {}'.format(df.shape, len(df_train), len(df_test)))

    # 推荐电影数
    reco_num = 5
    hot_rank = getHotItem(df_train, reco_num)

    algo = trainModel(df_train)

    # 生成user-tiem排列表
    user_item = UserItemDict(df_train.values)
    test_user_item = UserItemDict(df_test.values)

    # 定义test集的推荐字典
    test_reco_list = {}
    for test_user in df_test['User'].unique():

        # 生成单用户推荐列表
        rank_list = recommendation(algo, user_item, test_user, df_train['Movie'].unique(), reco_num)
        # 合并到总的推荐字典中
        test_reco_list.update(rank_list)


    recall, precision = precisionRecall(test_user_item, test_reco_list)

    rmse = rmse(df_test, algo)
    print('\n\nTesting Result: {:.4f} RMSE \t{:.4f} Precision \t{:.4f} Recall'.format(rmse, precision, recall))