# coding:utf-8
# 说明：评分作为兴趣度的itemCF算法
import numpy as np
import pandas as pd
import random
import math
import operator
import time
from sklearn.utils import shuffle


# 相似矩阵相似度最大值归一化（一种优化方案）
def normalizeSimilarity(dict):
    """
    param:
        dict: 相似度字典 type: key=item_i value=dict, key=item_j value=similarity
    return:
        dict: 归一化的相似度字典 type: key=item_i value=dict, key=item_j value=similarity
    """
    maxW = 0

    # 找到相似矩阵中相似度的最大值
    for k, item in dict.items():
        tmp = max(item.values())
        if maxW < tmp:
            maxW = tmp

    # 将所有相似度除以最大值
    for idx, item in dict.items():
        for i in item:
            item[i] /= maxW

    return dict


# 按比例随机分配训练数据和测试数据
def splitData(data, M, k, seed):
    """
    param:
        data: 二维矩阵 [user, item, rating]
        M: 测试集占比，训练集:测试集 = M:1
        k: 选取前k个相似item
        seed: 随机种子
    return:
        train: 训练数据 二维列表 [user, item, rating]
        test: 测试数据 二维列表 [user, item, rating]
    """
    test = []
    train = []
    random.seed(seed)
    for user, item, rate in data:
        if random.randint(0, M) == k:
            test.append([user, item, rate])
        else:
            train.append([user, item, rate])
    return train, test


# 将列表转成user-item字典
def userItemDict(data):
    """
    param:
        data: lsit [user, item, rating]
    return:
        user_item: 用户-电影排列表 type:dict, key=user, value=dict, key=item, value=rate
    """
    user_item = {}
    for user, item, rate in data:
        if user not in user_item:
            user_item[user] = {}
        user_item[user].update({item: rate})
    return user_item


# 计算item之间的相似度
def itemSimilarity(user_item):
    """
    param:
        user_item: 用户-电影排列表 type:dict, key:user, value:dict, key:item, value: rate
    return: 
        W：物品相似度矩阵，type:dict, key:item_i, value:dict, key:item_j, value:similarity
    """
    # C[i][j]存储观看电影i和j的用户数
    C = {}
    # 统计item的观看量 N[i]记录观看电影i的用户数
    N = {}
    for u, items in user_item.items():
        for item_i in items:
            N.setdefault(item_i, 0)
            N[item_i] += 1
            for item_j in items:
                if item_i == item_j:
                    continue
                C.setdefault(item_i, {})
                C[item_i].setdefault(item_j, 0)
                # 统计观看了电影i和电影j的用户数
                # 1.传统方法
                # C[item_i][item_j] += 1
                # 2.优化方法，削弱了活跃用户的贡献度，用户观看电影越多其影响越弱
                C[item_i][item_j] += 1 / math.log(1 + len(items) * 1.0)
    # 电影相似矩阵
    W = {}
    # W_sorted = {}

    #  item1, {item2: num, item3: num}
    for item_i, related_items in C.items():

        for item_j, cij in related_items.items():
            W.setdefault(item_i, {})
            W[item_i].setdefault(item_j, 0)
            # 计算相似度
            W[item_i][item_j] = cij / math.sqrt(N[item_i] * N[item_j])

            # 矩阵相似度从大到小排序
            # for item_i in W:
            #     W_sorted[item_i] = sorted(W[item_i].iteritems(), key = \
            # 				operator.itemgetter(1), reverse=True)

    # 优化方式（可选）：将相似矩阵相似度最大值归一化
    # W = NormalizeSimilarity(W)
    return W


# 获取全局热门电影
def getHotItem(df_train, N=5):
    """
    param:
        df_train: 训练数据集 type:dataframe
        N: 推荐的电影数
    return: 
        hot_rank: 该用户的推荐热门电影列表 type:dict, key:user, value:dict, key:item, value:sim
    """

    item_count = df_train.groupby('movieId')['rating'].count().sort_values(ascending=False)

    hot_rank = {}

    r = 0
    for item_id in item_count[0:N].index:
        hot_rank[item_id] = 1 - 0.01 * r
        r += 1
    return hot_rank


# 推荐电影
def recommendation(user_item, user_id, W, hot_rank, K, R):
    """
    param:
        user_item: 用户-电影排列表 type:dict, key:user, value:dict, key:item, value:rate
        user_id: 推荐的用户id
        W: 电影相似矩阵, type:dict, key:item_i, value:dict, key:item_j, value:similarity
        hot_rank: 热门电影列表, type:dict, key:user, value:dict, key:item, value:sim
        K: 前K个最相似电影
        R: 推荐列表中电影个数
    return: 
        rank_sorted：该用户的推荐电影列表 type:dict, key:user, value:dict, key:item, value:sim
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
        watched_item_list = user_item[user_id]

        # item_i:项目号， ri:对应的评分（兴趣度）
        for item_i, ri in watched_item_list.items():

            # 如果该item不在相似度矩阵中，则推荐空序列
            if item_i not in W:
                print('unvalid item_id(item_id not in W): ', item_i)
                continue

            # 在遍历电影i与相似矩阵中前K个电影j的相似度
            for item_j, wj in sorted(W[item_i].items(), key=operator.itemgetter(1), reverse=True)[0:K]:

                # 如果电影j在该用户的电影观看列表中则跳过
                if item_j in watched_item_list:
                    continue

                rank[user_id].setdefault(item_j, 0)
                # 电影推荐度 = 用户评分（或者兴趣度）* 电影相似度
                # 此例中用户观看过电影则兴趣度为1
                rank[user_id][item_j] += ri * wj

    rank_sorted = {}
    rank_sorted[user_id] = sorted(rank[user_id].items(), key=operator.itemgetter(1), reverse=True)[0:R]

    return rank_sorted


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


if __name__ == '__main__':

    start = time.time()
    # 读取数据
    df_train = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v1_train.csv', \
                           usecols=[0, 1, 2])

    df_test = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v1_test.csv', \
                          usecols=[0, 1, 2])
    print(len(df_train), len(df_test))

    # 推荐电影数
    reco_num = 30
    # 加权求和计算的相似项个数
    sim_num = 10
    hot_rank = getHotItem(df_train, reco_num)

    # data = df_data.values
    # train, test = splitData(data, 6, 1, 1)

    # 生成user-tiem排列表
    user_item = userItemDict(df_train.values)

    # 生成电影相似度字典
    item_sim = itemSimilarity(user_item)

    # 定义test集的推荐字典
    test_reco_list = {}
    for test_user in df_test['userId'].unique():
        # 生成单用户推荐列表
        rank_list = recommendation(user_item, int(test_user), item_sim, hot_rank, sim_num, reco_num)
        # 合并到总的推荐字典中
        test_reco_list.update(rank_list)

    test_user_item = userItemDict(df_test.values)
    recall, precision = precisionRecall(test_user_item, test_reco_list)

    print(recall, precision)
    print('time: ', time.time() - start)
