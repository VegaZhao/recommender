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
import time
import operator

# 将列表转成user-item字典
def userItemDict(data):
    """
    param：
        data: lsit [user, item, rating]
    return：
        user_item: 用户-电影排列表 type:dict, key:user, value:dict, key:item, value: rate
    """
    user_item = {}
    for user, item, rate in data:
        if user not in user_item:
            user_item[user] = {}
        user_item[user].update({item : rate})
    return user_item


# 模型训练
def trainModel(df_train):
    """
    param：
        df_train: 训练数据dataframe格式 包含字段 ('userId', 'movieId', 'rating')
    return:
        algo: 训练好的模型
    """
    # 读取数据
    reader = Reader()
    algo = SVD()
    data = Dataset.load_from_df(df_train[['userId', 'movieId', 'rating']], reader)
    ###################### train ######################
    # 训练模型
    # 方式 1: 交叉验证
    # (算法, 数据, loss计算方式， CV=交叉验证次数
    model_selection.cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

    # 方式 2: 没有交叉验证
    # trainset = data.build_full_trainset()
    # algo.fit(trainset)
    ###################################################

    # 返回训练好的模型
    return algo


# 获取全局热门电影
def getHotItem(df_train, N=5):
    """
    param：
        df_train: 训练数据集
        N：推荐的电影数
    return：
        rank：字典，该用户的推荐电影列表 {user_id: {item_t:rate1, item_k:rate2}}
    """
    item_count = df_train.groupby('movieId')['rating'].count().sort_values(ascending=False)

    hot_rank = {}

    r = 0
    for item_id in item_count[0:N].index:
        hot_rank[item_id] = 1 - 0.01 * r
        r += 1
    return hot_rank


def recommendation(model, user_item, user_id, item_set, hot_rank, R):
    """
    param：
        model: 训练的模型
        user_item: 训练集中user-item字典 {user1 : {item1 : rate1, item2 : rate2}, ...}}
        user_id：推荐的用户id
        item_set: 训练集中的电影集合
        hot_rank: 热门电影列表
        R：推荐列表中电影个数
    return：
        rank_sorted：该用户的推荐电影列表 type:dict, key:user, value:dict, key:item, value:score
    """
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

            # 将模型预测结果赋给rank[user_id][item_id]
            rank[user_id][item_id] = model.predict(user_id, item_id).est

    # 推荐列表按评分由高到低排序
    rank_sorted = {}
    rank_sorted[user_id] = sorted(rank[user_id].items(), key=operator.itemgetter(1), reverse=True)[0:R]

    return rank_sorted


# 准确度/召回评价
def precisionRecall(test, recommend):
    """
    param：
        test: 测试集用户-电影排列表 type:dict, key:user, value:dict, key:item, value: rate
        recommend: 用户推荐电影字典 type:dict, key:user, value:dict, key:item, value:score
    return:
        [召回率, 准确率]
    """
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

# 数据采样
def sampleData(data, thres_rate):
    """
    param:
        data: 二维矩阵 [item, user, rate]
        thres_rate: 评分转成0/1分类的阈值
    return:
        train_data 调整正负样本数之后的训练数据 type:list [(user, item, class)]
    """
    # 定义数据集
    train_data = []
    # 正样本字典 key:user value:tuple (item, rate)
    pos_dict = {}
    # 负样本字典 key:user value:tuple (item, rate)
    neg_dict = {}

    for user, item, rate in data:
        if user not in pos_dict:
            pos_dict[user] = []
        if user not in neg_dict:
            neg_dict[user] = []
        if rate >= thres_rate:
            pos_dict[user].append((item, rate))
        else:
            neg_dict[user].append((item, rate))
    for user in pos_dict:
        # 获取每个用户的正负样本数目，去原本正样本或者负样本的最小值，样本多余的截取
        data_num = min(len(pos_dict.get(user, [])), len(neg_dict.get(user, [])))
        if data_num > 0:
            # 按分值从大到小排序，保留data_num个样本数
            sorted_pos_list = sorted(pos_dict[user], key=lambda element: element[1], reverse=True)[:data_num]
            train_data += [(user, item, 1) for item, rate in sorted_pos_list]
            sorted_neg_list = sorted(neg_dict[user], key=lambda element: element[1], reverse=True)[:data_num]
            train_data += [(user, item, 0) for item, rate in sorted_neg_list]

    return train_data


if __name__ == '__main__':

    start = time.time()
    # 读取数据
    df_train = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v1_train.csv', \
                           usecols=[0, 1, 2])

    df_test = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v1_test.csv', \
                          usecols=[0, 1, 2])
    # sample num: 35497
    print(len(df_train), len(df_test))

    # 推荐电影数
    reco_num = 30
    # 评分转成0/1分类的阈值
    thres_rate = 4.0
    hot_rank = getHotItem(df_train, reco_num)

    ######################两种训练数据######################
    # 未调整正负样本数
    # df_train['rating'] = df_train['rating'].apply(lambda x: 1.0 if x>=thres_rate else 0.0)
    algo = trainModel(df_train)

    # 调整正负样本数 1：1
    # train_data = sampleData(df_train.values, thres_rate)
    # df_sample = pd.DataFrame(train_data, columns=['userId', 'movieId', 'rating'])
    # algo = trainModel(df_sample)
    ###################################################

    # 生成user-tiem排列表
    user_item = userItemDict(df_train.values)
    test_user_item = userItemDict(df_test.values)

    # 定义test集的推荐字典
    test_reco_list = {}
    for test_user in df_test['userId'].unique():
        # 生成单用户推荐列表
        rank_list = recommendation(algo, user_item, test_user, df_train['movieId'].unique(), hot_rank, reco_num)
        # 合并到总的推荐字典中
        test_reco_list.update(rank_list)

    recall, precision = precisionRecall(test_user_item, test_reco_list)

    print('\n\nTesting Result:Precision={:.4f}\t Recall={:.4f}'.format(precision, recall))
    print(recall, precision)
    print('time: ', time.time() - start)