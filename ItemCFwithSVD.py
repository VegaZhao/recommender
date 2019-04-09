#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import time
from numpy import linalg as la
from sklearn.metrics import mean_squared_error


def ecludSim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))


def pearsSim(inA, inB):
    if len(inA) < 3: return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)


# 一般的评估方法
def standEst(dataMat, user, simMeas, item):
    # 作用：评估对应用户和电影的得分
    # 输入：
    # 	dataMat 二维的矩阵matrix
    # 	user 用户编号，此处为索引号
    # 	simMeas 相似度计算方法
    # 	item 电影索引号
    # 输出：
    # 	pred 评估得分

    # 获取电影数
    n = np.shape(dataMat)[1]

    # 初始化所有相似度之和
    simTotal = 0.0

    # 初始化所有相似度和得分加权之和
    ratSimTotal = 0.0

    # 遍历所有电影
    for j in range(n):

        # 获取user用户对电影的评分
        userRating = dataMat[user, j]

        # 如果评分为0，证明没有观看，忽略；如果是同一部电影，也忽略
        if userRating == 0 or j == item: continue

        # 获取同时观看（给出评分）电影j和电影item的user索引号
        overLap = np.nonzero(np.logical_and(dataMat[:, item] > 0, \
                                      dataMat[:, j] > 0))[0]
        # 如果没有同时观看这两部电影的用户，则这两部电影相似度为0；
        # 反之计算两部电影的相似度
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], \
                                 dataMat[overLap, j])
        print 'the %d and %d similarity is: %f' % (item, j, similarity)

        # 将所有与item电影相似的电影相似度叠加，同时相似度加权评分叠加
        simTotal += similarity
        ratSimTotal += similarity * userRating

    # 如果相似度之和为0，则评分为0；反之，返回评估得分
    if simTotal == 0:
        pred = 0
    else:
        pred = ratSimTotal / simTotal
    return pred


# svd优化方法
def svdEst(dataMat, user, simMeas, item):
    # 作用：评估对应用户和电影的得分
    # 输入：
    # 	dataMat 二维的矩阵matrix
    # 	user 用户编号，此处为索引号
    # 	simMeas 相似度计算方法
    # 	item 电影索引号
    # 输出：
    # 	pred 评估得分

    # 获取电影数
    n = np.shape(dataMat)[1]

    # 初始化所有相似度之和
    simTotal = 0.0

    # 初始化所有相似度和得分加权之和
    ratSimTotal = 0.0

    # 使用svd分解矩阵
    U, Sigma, VT = la.svd(dataMat)

    # 取前n（此处n=4）个主要特征，既对角阵中的前n位
    Sig4 = np.mat(np.eye(4) * Sigma[:4])  # arrange Sig4 into a diagonal matrix

    # 获取压缩后的电影因子矩阵
    xformedItems = dataMat.T * U[:, :4] * Sig4.I  # create transformed items

    # 遍历所有电影
    for j in range(n):
        # 获取user用户对电影的评分
        userRating = dataMat[user, j]

        # 如果评分为0，证明没有观看，忽略；如果是同一部电影，也忽略
        if userRating == 0 or j == item: continue

        # 比较电影j与电影item的相似度
        similarity = simMeas(xformedItems[item, :].T, \
                             xformedItems[j, :].T)
        print 'the %d and %d similarity is: %f' % (item, j, similarity)
        # 将所有与item电影相似的电影相似度叠加，同时相似度加权评分叠加
        simTotal += similarity
        ratSimTotal += similarity * userRating

    # 如果相似度之和为0，则评分为0；反之，返回评估得分
    if simTotal == 0:
        pred = 0
    else:
        pred = ratSimTotal / simTotal
    return pred

# 推荐系统
def recommend(dataMat, user, user_idx_mapping, idx_item_mapping, \
              forTest=False, N=3, simMeas=cosSim, estMethod=svdEst):
    # 作用：给出user推荐电影
    # 输入：
    #   dataMat 用户电影矩阵
    #   user 需要推荐电影的用户
    #   user_idx_mapping 用户号映射到矩阵中的行索引号
    #   idx_item_mapping 矩阵的列索引号映射到电影号
    #   forTest 标识符：Ture则是单个用户推荐，返回该用户推荐的电影；
    #					False则是用于批量用户推荐，在测试集中评估结果
    #   N 推荐电影数
    #   simMeas 计算相似度方法
    #   estMethod 评估方法
    # 输出：
    #   推荐电影列表

    # 找出用户电影矩阵中没有评分的电影
    unratedItems = np.nonzero(dataMat[user_idx_mapping[user], :] == 0)[1]
    if len(unratedItems) == 0: return 'you rated everything'

    # 当forTest为False时，进入单个用户的推荐
    if forTest == False:

        # 定义未评分的电影及估计得分
        itemScores = []

        # 遍历未观看电影
        for item in unratedItems:
            # 获得该用户中为观看电影的估计得分
            estimatedScore = estMethod(dataMat, user_idx_mapping[user], simMeas, item)
            itemScores.append((idx_item_mapping[item], estimatedScore))

        # 推荐列表
        reco_list = sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]
        return reco_list

    # 当forTest为True时，用于评估算法，计算测试集中的RMSE结果
    else:
        userItemScores = []
        for item in unratedItems:
            estimatedScore = estMethod(dataMat, user_idx_mapping[user], simMeas, item)

            # 此处将user也放入列表中
            userItemScores.append((user, idx_item_mapping[item], estimatedScore))
        return userItemScores

def main(df_train, df_test):

    # 将训练集数据转换成透视表
    df_p = df_train.pivot_table(index='User', columns='Movie', values='Rating')
    print('Shape User-Movie-Matrix:\t{}'.format(df_p.shape))

    # 补充透视表中缺失的值，补充的值为电影已有评价的均值
    df_p_imputed = df_p.fillna(0)
    df_p_arr = np.mat(df_p_imputed.values)

    # 创建 user-id 映射
    user_id_mapping = {id: i for i, id in enumerate(df_p_imputed.index)}

    # 创建 id_movie 映射
    id_movie_mapping = {i: id for i, id in enumerate(df_p_imputed.columns)}

    ###################################推荐单个用户####################################
    # user_id = 101

    # reco_list = recommend(df_p_arr, user_id, user_id_mapping, id_movie_mapping)
    #
    # print('reco_list: ', reco_list)

    ###################################批量预测用户电影评分####################################

    # 创建prediction列表，存储预测结果
    prediction = []
    # 遍历测试集中的所有电影
    for user_id in df_test['User'].unique():
	
		# 如果是训练集中不存在的新用户，暂时跳过
		if user_id not in user_id_mapping:
			continue
        pred = recommend(df_p_arr, user_id, user_id_mapping, id_movie_mapping, True, estMethod=standEst)
        prediction.extend(pred)

    df_pred = pd.DataFrame(prediction, columns=['User', 'Movie', 'Prediction']).set_index(['User', 'Movie'])

    # 将预测结果与测试集数据自然连接
    df_pred = df_test.set_index(['User', 'Movie']).join(df_pred)

    # 获取labels 与 predictions
    y_true = df_pred['Rating'].values
    y_pred = df_pred['Prediction'].values

    # 计算 RMSE
    rmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))

    print('\n\nTesting Result With Cosine User-User Similarity: {:.4f} RMSE'.format(rmse))


if __name__ == '__main__':
    df_data = pd.read_csv('/home/zwj/Desktop/recommend/uus_CF_demo.txt', header=None, names=['Movie', 'User', 'Rating'])

    # 将数据集拆分为训练集和测试集
    m = 3
    df_train = df_data[0:-m]
    df_test = df_data[-m:]
'''	
	df_data = pd.read_csv('/root/vega/netflix-prize-data/ft_ratings.txt', \
							  sep=' ', header=None, names=['User', 'Movie', 'Rating'])

	# sample:35497						  
	# 将数据集拆分为训练集和测试集，后m条为测试数据，其余的为训练数据
	m = 30000

	df_train = df_data[0:m]
	df_test = df_data[m:]
'''
    # 执行主程序
    main(df_train, df_test)