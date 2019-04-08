# coding=utf-8
from numpy import *
from numpy import linalg as la


def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]


def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


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
    n = shape(dataMat)[1]

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
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, \
                                      dataMat[:, j].A > 0))[0]
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
    n = shape(dataMat)[1]

    # 初始化所有相似度之和
    simTotal = 0.0

    # 初始化所有相似度和得分加权之和
    ratSimTotal = 0.0

    # 使用svd分解矩阵
    U, Sigma, VT = la.svd(dataMat)

    # 取前n（此处n=4）个主要特征，既对角阵中的前n位
    Sig4 = mat(eye(4) * Sigma[:4])  # arrange Sig4 into a diagonal matrix

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
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    # 作用：给出user推荐电影
    # 输入：
    #   dataMat 用户电影矩阵
    #   user 需要推荐电影的用户
    #   N 推荐电影数
    #   simMeas 计算相似度方法
    #   estMethod 评估方法
    # 输出：
    #   推荐电影列表

    # 找出用户电影矩阵中没有评分的电影
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0: return 'you rated everything'

    # 定义未评分的电影及估计得分
    itemScores = []

    # 遍历未观看电影
    for item in unratedItems:

        # 获得该用户中为观看电影的估计得分
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))

    # 返回推荐列表
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]
