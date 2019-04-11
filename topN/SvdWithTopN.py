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

def PrecisionRecall(test, recommend):
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

    # return trained model
    return algo

def recommendation(model, user_item, user_id, item_set, R):
    # 输入：
    #	user_item: 字典 {user1 : {item1 : rate1, item2 : rate2}, ...}}
    #	user_id：推荐的用户id
    #   R：推荐列表中电影个数
    # 输出：
    #	rank：字典，该用户的推荐电影列表 {item_t:sim1, item_k:sim2}

    # 存储用户推荐电影
    rank = {}
    # 开辟用户空子字典 ('rank: ', {user_id: {}})
    rank.setdefault(user_id, {})

    # 如果该用户不在训练集中，则推荐空序列
    if user_id not in user_item:
        print('unvalid user_id(user_id not in user_item): ', user_id)
        return rank

    # 用户已观看的电影集合
    item_watched_list = user_item[user_id]

    for item_id in item_set:
        if item_id in item_watched_list:
            continue
        rank[user_id].setdefault(item_id, 0)

        #
        rank[user_id][item_id] = model.predict(user_id, item_id).est
    rank_sorted = {}
    rank_sorted[user_id] = sorted(rank[user_id].items(), key=operator.itemgetter(1), reverse=True)[0:R]

    return rank_sorted


    # # 预测
    # df_test['Predict_Score'] = df_test.apply(lambda row: algo_predict(model, row['User'], row['Movie']), axis=1)
    # # 计算RMSE
    # rmse = np.sqrt(mean_squared_error(df_test['Predict_Score'], df_test['Rating']))
    # print('\n\nTesting Result: {:.4f} RMSE'.format(rmse))

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

    algo = CF(df_train, df_test)

    # 生成user-tiem排列表
    user_item = UserItemDict(df_train.values)
    test_user_item = UserItemDict(df_test.values)

    # 定义test集的推荐字典
    test_reco_list = {}
    for test_user in df_test['User'].unique():

        # 生成单用户推荐列表
        rank_list = recommendation(algo, user_item, test_user, df_train['Movie'].unique(), 5)
        # 合并到总的推荐字典中
        test_reco_list.update(rank_list)


    recall, precision = PrecisionRecall(test_user_item, test_reco_list)

    print(recall, precision)