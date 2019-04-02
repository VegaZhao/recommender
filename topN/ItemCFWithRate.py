# coding:utf-8
# 说明：评分作为兴趣度的itemCF算法
import numpy as np
import pandas as pd
import random
import math
import operator

# 相似矩阵相似度最大值归一化（一种优化方案）
def NormalizeSimilarity(dict):
	# 输入：
	#	二维字典 {item_i: {item_k: similarity}}
	# 输出：
	#	二维字典 {item_i: {item_k: similarity}}
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
def SplitData(data, M, k, seed):
	# 输入：
	#	data: 二维矩阵 [item, user, rating]
	#	M：测试集占比，训练集:测试集 = M:1
	#	k：选取前k个相似item
	#	seed：随机种子
	# 输出：
	#	train, test 二维列表 [item, user, rating]
    test = []
    train = []
    random.seed(seed)
    for item, user, rate in data:
        if random.randint(0, M) == k:
            test.append([user,item, rate])
        else:
            train.append([user,item, rate])
    return train, test

# 将列表转成user-item字典
def UserItemDict(data):
	# 输入：
	#	二维列表 [item, user, rating]
	# 输出：
	#	user_item字典 {user1 : {item1 : rate1, item2 : rate2}, ...}}
	user_item = {}
	for item, user, rate in data:
		if user not in user_item:
			user_item[user] = {}
		user_item[user].update({item : rate})
	return user_item

# 计算item之间的相似度
def ItemSimilarity(user_item):
	# 输入：
	#	user-item字典 {user1 : {item1 : rate1, item2 : rate2}, ...}}
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
                C[item_i][item_j] += 1
                # 2.优化方法，削弱了活跃用户的贡献度，用户观看电影越多其影响越弱
                # C[item_i][item_j] = 1 / math.log(1 + len(items) * 1.0)
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

# 推荐电影
def Recommendation(user_item, user_id, W, K):
    # 输入：
	#	user_item: 字典 {user1 : {item1 : rate1, item2 : rate2}, ...}}
	#	user_id：推荐的用户id
	#	W：电影相似矩阵
	#	K：前K个最相似电影
	# 输出：
	#	rank：字典，该用户的推荐电影列表 {item_t:sim1, item_k:sim2}

	# 存储用户推荐电影
    rank = {}
	# 用户已观看的电影集合
    item_list = user_item[user_id]

	# 开辟用户空子字典 ('rank: ', {user_id: {}})
    rank.setdefault(user_id, {})
	# item_i:项目号， ri:对应的评分（兴趣度）
    for item_i, ri in item_list.items():

        # 在遍历电影i与相似矩阵中前K个电影j的相似度
        for item_j, wj in sorted(W[item_i].items(), key=operator.itemgetter(1), reverse=True)[0:K]:

			# 如果电影j在该用户的电影观看列表中则跳过
            if item_j in item_list:
                continue
            rank[user_id].setdefault(item_j, 0)
			# 电影推荐度 = 用户评分（或者兴趣度）* 电影相似度
			# 此例中用户观看过电影则兴趣度为1
            rank[user_id][item_j] += ri * wj
    return rank

if __name__ == '__main__':

    # 读取数据
    df_data = pd.read_csv('/home/zwj/Desktop/recommend/topNCF_demo.txt', header=None, names=['Movie', 'User', 'Rating'])
    data = df_data.values

    # 拆分训练集和测试集
    train, test = SplitData(data, 2, 1, 1)

    # 生成user-tiem排列表
    user_item = UserItemDict(data)

    # 生成电影相似度字典
    item_sim = ItemSimilarity(user_item)

    # 定义test集的推荐字典
    test_reco_list = {}
    for test_user in test:

        # 生成单用户推荐列表
        rank_list = Recommendation(user_item, test_user[0], item_sim, 6)
        # 合并到总的推荐字典中
        test_reco_list.update(rank_list)

    print test_reco_list

