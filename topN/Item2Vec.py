# coding=utf-8
import os
import pandas as pd
import numpy as np
import operator
import time
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity

# 将列表转成user-item字典
def userItemDict(data):
    # 输入：
    #   data: type:ndarray [[user, item, rating],[...]]
    # 输出：
    #   user_item 用户商品排列表 type:dict {user_id: {item_t:rate1, item_k:rate2}}
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
    #	rank：该用户的推荐电影列表 type:dict {user_id: {item_t:score1, item_k:score2}}

    # 电影观看量统计，并且降序排列
    item_count = df_train.groupby('Movie')['Rating'].count().sort_values(ascending=False)

    # 定义热门电影字典
    hot_rank = {}

    # 给热门电影计算score，公式没有意义，只是为了保持热门电影的得分排序，越热门得分越高
    r = 0
    for item_id in item_count[0:N].index:
        hot_rank[item_id] = 1 - 0.01*r
        r += 1
    return hot_rank

# 准确度/召回评价
def precisionRecall(test, recommend):
    # 推荐列表命中数
    hit = 0
    # 召回率分母（测试集中用户观看电影数）
    n_recall = 0
    # 准确率分母（推荐列表中电影数）
    n_precision = 0
    # 统计测试集中用户数
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

# 将列表转成itemWord字典
def itemWord(data, save_file):
    # 输入：
    #   data: type:ndarray [[user, item, rating],[...]]
    # 将data转成如下字典格式，注意itemid存储的是字符串
    # {userid1: ['itemid1', 'itemid2'], userid2: [...], ...}

    user_itemWord = {}
    for user, item, rate in data:
        if user not in user_itemWord:
            # int()是为了最后去掉小数点.0的部分
            user_itemWord[int(user)] = []
        # 判断，电影评分大于3的，归到感兴趣的影片集中
        if rate > 3.0:
            # str()将itemid转成字符串，因为join的对象必须是字符串
            user_itemWord[user].append(str(int(item)))
    # 保存user_itemWord中values部分['itemid1', 'itemid2']，每个用户的item集合为一行，看作一个句子

    fw = open(save_file, 'w+')
    for user in user_itemWord:
        fw.write(" ".join(user_itemWord[user]) + "\n")
    fw.close()
    # 后续使用word2vec工具将此txt文件转成词向量文件，文件中一个向量表示一个词（本例中为itemid）

# 从word2vec工具生成的文件中加载item_vec
def loadItemVec(input_file):
    # 输入：
    #   input_file: 词向量文件路径
    # 输出：
    #   item_vec：type:dict key:userid values:词向量
    if not os.path.exists(input_file):
        print('input file not found!')
        return {}

    # 第一行标识，第一行是要舍去的
    linenum = 0
    item_vec = {}
    fp = open(input_file)
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split()
        # 因为本例中词向量是128维，加上userid应该有129个值
        if len(item) < 129:
            continue
        itemid = item[0]
        # "</s>"去掉文档中此符号的词向量
        if itemid == "</s>":
            continue
        item_vec[int(itemid)] = np.array([float(x) for x in item[1:]])
    fp.close()
    return item_vec

# 计算item_vec的相似度
def calItemSim(item_vec, item_id, K=10):
    # 输入：
    #   item_vec: type:dict key:userid values:词向量
    #   item_id: 物品id
    #   K： 考虑相似item的个数，默认为10
    # 输出：
    #   score：返回与item_id相似的id及相似度，type:list [(itemid1, sim1), (itemid2, sim2)]

    # 如果item_id不在item向量字典中，返回空
    if item_id not in item_vec:
        return

    score = {}
    fix_item_vec = item_vec[item_id]

    # 遍历向量字典中的所有itemid，与该item_id计算相似度
    for tmp_itemid in item_vec:
        # 如果是自己，则跳过
        if tmp_itemid == item_id:
            continue
        tmp_itemvec = item_vec[tmp_itemid]

        # 计算cosine相似度方法1，使用函数
        tmp_sim = np.concatenate((fix_item_vec.reshape(1, -1), tmp_itemvec.reshape(1, -1)), axis=0)
        # cosine_similarity返回列表中[0, 1]位置就是需要的相似度的值
        score[tmp_itemid] = cosine_similarity(tmp_sim)[0, 1]

        # 计算cosine相似度方法2，直接计算
        # fenmu = np.linalg.norm(fix_item_vec) * np.linalg.norm(tmp_itemvec)
        # if fenmu == 0:
        #     score[tmp_itemid] = 0
        # else:
        #     score[tmp_itemid] = round(np.dot(fix_item_vec, tmp_itemvec)/fenmu, 3)

    # 对item向量相似度字典进行排序，返回前K个向此向量
    score = sorted(score.items(), key=operator.itemgetter(1), reverse=True)[0:K]
    return score

# 推荐电影
def recommendation(user_item, user_id, hot_rank, item_vec, K, R):
    # 输入：
    #	user_item: 字典 {user1 : {item1 : rate1, item2 : rate2}, ...}}
    #	user_id：推荐的用户id
    #   hot_rank: 热门电影列表
    #   item_vec: item词向量
    #	K：前K个最相似电影
    #   R：推荐列表中电影个数
    # 输出：
    #	rank：字典，该用户的推荐电影列表 {user_id: {item_t:sim1, item_k:sim2}}

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
        # 该指标用于判断，如果userid用户中观看的电影itemid均不再item词向量中，则没有相似度判断依据，返回热门电影
        item_in_vec = False
        # item_i:项目号， ri:对应的评分（兴趣度）
        for item_i, ri in watched_item_list.items():
            if item_i not in item_vec:
                continue
            # 只要存在一个及以上的item在item词向量中，则表示为True，不用返回热门电影
            item_in_vec = True
            # 获得itemid的相似item集
            simMovie = calItemSim(item_vec, item_i, K)
            # 遍历该item集合，推荐的电影在这个集合中选择
            for item_j, simj in simMovie:
                # 如果item是已观看过的，则跳过
                if item_j in watched_item_list:
                    continue

                rank[user_id].setdefault(item_j, 0)
                # 电影推荐度 = 用户评分（或者兴趣度）* 电影相似度
                rank[user_id][item_j] += ri * simj
        # userid用户中观看的电影itemid均不再item词向量中，则没有相似度判断依据，返回热门电影
        if item_in_vec == False:
            rank[user_id] = hot_rank

    # 对推荐的电影排序，返回前R部电影
    rank_sorted = {}
    rank_sorted[user_id] = sorted(rank[user_id].items(), key=operator.itemgetter(1), reverse=True)[0:R]

    return rank_sorted

if __name__ == '__main__':
    start = time.time()
    # 读取数据
    df_train = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v1/v1_train.csv', \
                          usecols=[0, 1, 2])

    df_test = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v1/v1_test.csv', \
                          usecols=[0, 1, 2])
    print(len(df_train), len(df_test))

    # item转词向量（执行一次即可，也可单独执行）
    # item_word_file = '/home/zwj/Desktop/recommend/movielens/moive_database/v1/train_itemWord.txt'
    # itemWord(df_train.values, item_word_file)

    # 加载item词向量
    item_vec_file = '/home/zwj/Desktop/recommend/movielens/moive_database/v1/train_itemVec.txt'
    item_vec = loadItemVec(item_vec_file)

    # 推荐电影数
    reco_num = 30
    # 加权求和计算的相似项个数
    sim_num = 20

    # 热门电影列表
    hot_rank = getHotItem(df_train, reco_num)

    # 生成user-tiem排列表
    user_item = userItemDict(df_train.values)

    # 定义test集的推荐字典
    test_reco_list = {}
    # 遍历test集中的所有用户
    for test_user in df_test['User'].unique():
        print('user {} recommend'.format(test_user))
        # 生成单用户推荐列表
        rank_list = recommendation(user_item, int(test_user), hot_rank, item_vec, sim_num, reco_num)
        # 合并到总的推荐字典中
        test_reco_list.update(rank_list)
    # test集中user实际观看的电影集合
    test_user_item = userItemDict(df_test.values)
    # 计算召回率和准确率
    recall, precision = precisionRecall(test_user_item, test_reco_list)

    print(recall, precision)
    print('time: ', time.time() - start)