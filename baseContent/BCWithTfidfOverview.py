# coding=utf-8
import sys
import pandas as pd
import numpy as np
import operator
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle


# 解决ascii 编码字符串转换成"中间编码" unicode 时由于超出了其范围的问题
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# 将列表转成user-item字典
def UserItemDict(data):
    user_item = {}
    for user, item, rate in data:
        if user not in user_item:
            user_item[user] = {}
        user_item[user].update({item : rate})
    return user_item

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
        hot_rank[item_id] = 1 - 0.01*r
        r += 1
    return hot_rank

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

def overviewSimReco(movie_id, n):
    """
    param:
        movie_id: 电影ID号
        n: 前n部相似电影
    return: 
        reco_list:  推荐电影列表 type:dict key:movieid value:sim_score
    """
    # title:Men in Black id:12918
    # 加载电影信息
    train_movie_set = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v1_train.csv')['movieId'].unique()
    movie_metadata_raw = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v1_movie_info.csv',\
            usecols = [0, 1, 3])  #id title overview

    # 将电影名称设置为dataframe索引
    movie_metadata = movie_metadata_raw[movie_metadata_raw['id'].isin(train_movie_set)].set_index('id')

    # 创建tf-idf矩阵，用于比较电影简介的相似度
    # stop_words='english'使用英语内建的停用词列表
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movie_metadata['overview'].dropna())

    # 返回该电影在数据集中的位置索引
    index = movie_metadata.reset_index(drop=True)[movie_metadata.index==movie_id].index[0]

    # 存储该电影简介与其他电影的相似度
    sim_movie = []

    # 遍历所有的电影
    for idx in range(np.shape(tfidf_matrix)[0]):

        # 如果是剔除同一部电影，相似度置为0
        if idx == index:
            sim_movie.append(0.0)
        else:
            tmp = np.concatenate((tfidf_matrix[index].toarray(), tfidf_matrix[idx].toarray()), axis=0)
            similarity = cosine_similarity(tmp)[0, 1]
            sim_movie.append(similarity)


    # 获取前n部最相似电影的索引和相似度
    similar_movies_index = np.argsort(sim_movie)[::-1][:n]
    similar_movies_score = np.sort(sim_movie)[::-1][:n]

    # 获得相似电影的名称
    similar_movies_id = movie_metadata.iloc[similar_movies_index].index

    reco_list = {}
    for id, score in zip(similar_movies_id, similar_movies_score):
        reco_list[id] = score

    return reco_list

# 推荐电影
def recommendation(user_item, user_id, hot_rank, K, R):
    """
    param：
        user_item: 训练集中user-item字典 {user1 : {item1 : rate1, item2 : rate2}, ...}}
        user_id：推荐的用户id
        hot_rank: 热门电影列表
        K：前K个最相似电影
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
        watched_item_list = user_item[user_id]

        # item_i:项目号， ri:对应的评分（兴趣度）
        for item_i, ri in watched_item_list.items():
            simMovie = overviewSimReco(item_i, K)
            for item_j, simj in simMovie.items():
                if item_j in watched_item_list:
                    continue

                rank[user_id].setdefault(item_j, 0)
                # 电影推荐度 = 用户评分（或者兴趣度）* 电影相似度
                # 此例中用户观看过电影则兴趣度为1
                rank[user_id][item_j] += ri * simj

    rank_sorted = {}
    rank_sorted[user_id] = sorted(rank[user_id].items(), key=operator.itemgetter(1), reverse=True)[0:R]

    return rank_sorted

if __name__ == '__main__':

    start = time.time()
    # 读取数据
    df_train = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v1_train.csv', \
                           usecols=[0, 1, 2])

    df_test = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v1_test.csv', \
                          usecols=[0, 1, 2])

    # 推荐电影数
    reco_num = 30
    # 加权求和计算的相似项个数
    sim_num = 20
    hot_rank = getHotItem(df_train, reco_num)

    # 生成user-tiem排列表
    user_item = UserItemDict(df_train.values)

    # 定义test集的推荐字典
    test_reco_list = {}
    for test_user in df_test['userId'].unique():
        print('user {} recommend'.format(test_user))
        # 生成单用户推荐列表
        rank_list = recommendation(user_item, int(test_user), hot_rank, sim_num, reco_num)
        # 合并到总的推荐字典中
        test_reco_list.update(rank_list)

    test_user_item = UserItemDict(df_test.values)
    recall, precision = precisionRecall(test_user_item, test_reco_list)

    print(recall, precision)
    print('time: ', time.time() - start)