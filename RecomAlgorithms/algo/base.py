# coding:utf-8
import sys
import pandas as pd
import random
import math
import operator
import time

sys.path.append("..")
from reader.read_data import readRawData
from util.util import precisionRecall


# 将列表转成user-item字典
def userItemDict(data):
    # 输入：
    #	data: lsit [user, item, rating]
    # 输出：
    #	user_item: 用户-电影排列表 type:dict, key:user, value:dict, key:item, value: rate
    user_item = {}
    for user, item, rate in data:
        if user not in user_item:
            user_item[user] = {}
        user_item[user].update({item : rate})
    return user_item


# 获取全局热门电影
def getHotItem(df_train, N=5):
    # 输入：
    #	df_train: 训练数据集
    #	N：推荐的电影数
    # 输出：
    #	rank：该用户的推荐电影列表 type:dict, key:user, value:dict, key:item, value:sim
    item_count = df_train.groupby('Movie')['Rating'].count().sort_values(ascending=False)

    hot_rank = {}

    r = 0
    for item_id in item_count[0:N].index:
        hot_rank[item_id] = 1 - 0.01*r
        r += 1
    return hot_rank

# 推荐电影
def recommendation(user_id, hot_rank):
    # 输入：
    #	user_id：推荐的用户id
    #   hot_rank: 热门电影列表
    # 输出：
    #	rank_sorted：该用户的推荐电影列表 type:dict, key:user, value:dict, key:item, value:sim

    # 存储用户推荐电影
    rank = {}
    # 开辟用户空子字典 ('rank: ', {user_id: {}})
    rank.setdefault(user_id, {})
    rank[user_id] = hot_rank

    rank_sorted = {}
    rank_sorted[user_id] = sorted(rank[user_id].items(), key=operator.itemgetter(1), reverse=True)[0:R]

    return rank_sorted


if __name__ == '__main__':

    start = time.time()
	
	df_train, df_test = readRawData()
    # 推荐电影数
    reco_num = 5
    hot_rank = getHotItem(df_train, reco_num)


    # 定义test集的推荐字典
    test_reco_list = {}
    for test_user in df_test['User'].unique():

        # 生成单用户推荐列表
        rank_list = recommendation(int(test_user), hot_rank)
        # 合并到总的推荐字典中
        test_reco_list.update(rank_list)

    test_user_item = userItemDict(df_test.values)
    recall, precision = precisionRecall(test_user_item, test_reco_list)

    print(recall, precision)
    print('time: ', time.time() - start)
