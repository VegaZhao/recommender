# coding:utf-8
import numpy as np
import pandas as pd
import random
import math
import operator

df_data = pd.read_csv('/home/zwj/Desktop/recommend/uus_CF_demo.txt', header=None, usecols = [0, 1], names=['Movie', 'User'])

data = df_data.values

# item相似矩阵相似度最大值归一化（一种优化方案）
def NormalizeSimilarity(dict):
	# 输入：一个二维字典
	# 输出：一个二维字典
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
	#	data: 二维矩阵 [item, user]
	#	M：测试集占比，训练集:测试集 = M:1
	#	k：选取前k个相似item
	#	seed：随机种子
	# 输出：
	#	train, test 二维列表
    test = []
    train = []
    random.seed(seed)
    for item, user in data:
        if random.randint(0, M) == k:
            test.append([user,item])
        else:
            train.append([user,item])
    return train, test

train, test = SplitData(data, 2, 1, 1)

# 将列表转成user-item字典
def UserItemDict(data):
	# 输入：
	#	二维列表 [item, user]
	# 输出：
	#	user_item字典 {userid:[item1, item2, ...]}
	user_item = {}
	for item, user in data:
		if user not in user_item:
			user_item[user] = []
		user_item[user].append(item)
	return user_item

# 计算item之间的相似度
def ItemSimilarity(train_dict):
	# 输入：
	#	user-item字典 {user1: [item1, item2], user2: [item1, item3], ...}
	# 倒排表，C[i][j]存储观看电影i和j的用户数
    C = {}
	# 统计item的观看量 N[i]记录观看电影i的用户数
    N = {}
    for u, items in train_dict.items():
        for id_i in items:
            N.setdefault(id_i, 0)
            N[id_i] += 1
            for id_j in items:
                if id_i == id_j:
                    continue
                C.setdefault(id_i, {})
                C[id_i].setdefault(id_j, 0)
				# 统计观看了电影i和电影j的用户数
                # 1.传统方法
                C[id_i][id_j] += 1
                # 2.优化方法，削弱了活跃用户的贡献度，用户观看电影越多其影响越弱
                # C[id_i][id_j] = 1 / math.log(1 + len(items) * 1.0)
	# 电影相似矩阵
    W = {}
    # W_sorted = {}
    for id_i, related_items in C.items():
		#  item1, {item2: num, item3: num}
        for id_j, cij in related_items.items():
            W.setdefault(id_i, {})
            W[id_i].setdefault(id_j, 0)
			# 计算相似度
            W[id_i][id_j] = cij / math.sqrt(N[id_i] * N[id_j])

	# 矩阵相似度从大到小排序
    # for id_i in W:
    #     W_sorted[id_i] = sorted(W[id_i].iteritems(), key = \
		# 				operator.itemgetter(1), reverse=True)

	# 优化方式（可选）：将相似矩阵相似度最大值归一化
    # W = NormalizeSimilarity(W)
    return W

# 推荐电影
def Recommendation(user_item, user_id, W, K):
	# 输入：
	#	user_item: 字典 {userid:[item1, item2, ...]}
	#	user_id：推荐的用户id
	#	W：电影相似矩阵
	#	K：前K个最相似电影
	# 输出：
	#	rank：字典，该用户的推荐电影列表

	# 存储用户推荐电影
    rank = {}
	# 用户观看的电影集合
    item_list = user_item[user_id]

	# 开辟用户空子字典 ('rank: ', {user_id: {}})
    rank.setdefault(user_id, {})
    for id_i in item_list:
		# 在遍历电影i与相似矩阵中前K个电影j的相似度
        for id_j, wj in sorted(W[id_i].items(), key=operator.itemgetter(1), reverse=True)[0:K]:
			# 如果电影j在该用户的电影观看列表中则跳过
            if id_j in item_list:
                continue
            rank[user_id].setdefault(id_j, 0)
			# 电影推荐度 = 用户兴趣度（或者评分）* 电影相似度
			# 此例中用户观看过电影则兴趣度为1
            rank[user_id][id_j] += 1 * wj
    return rank

user_item = UserItemDict(data)
item_sim = ItemSimilarity(user_item)
rank_list = Recommendation(user_item, 104, item_sim, 2)

print rank_list
