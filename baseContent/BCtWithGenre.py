# coding=utf-8
import sys
import pandas as pd
import numpy as np
import operator
import time

# 解决ascii 编码字符串转换成"中间编码" unicode 时由于超出了其范围的问题
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)


# 将日期字符串转换成时间戳
def str2timestamp(time_str):
    """
    param:
        time_str: 时间字符串
    return: 
        timestamp: 时间戳
    """
    model = "%Y/%m/%d"
    time_array = time.strptime(time_str, model)
    timestamp = int(time.mktime(time_array))
    return timestamp


# 计算时间得分
def getTimeScore(timestamp):
    """
    param:
        timestamp: 时间戳
    return: 
        时间权重得分
    """
    # 训练集中最近的日期，根据实际情况需要更改
    fix_time_stamp = 893286638
    total_sec = 24 * 60 * 60 * 100
    delta = (fix_time_stamp - timestamp) / total_sec
    # 返回时间得分，日期最近权重越大
    return round(1.0 / (1 + delta), 3)


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
        reco_item = [item for item in recommend[user]]
        # 推荐电影命中列表
        hit_list = [item for item in reco_item if item in test[user]]

        user_num += 1
        hit += len(hit_list)
        n_recall += len(test[user])
        n_precision += len(reco_item)
    print('user_num: {}, hit: {}, n_recall: {}, n_precision: {} '.format(user_num, hit, n_recall, n_precision))

    return [hit / (1.0 * n_recall), hit / (1.0 * n_precision)]


# 获得item的全局平均得分
def getItemAveScore(data):
    """
    param:
        data: type ndarray [[user, item, rating]]
    return: 
        ave_score: item的全局平均得分字典 type dict, key:item, value: ratio
    """

    # 中间字典 key: item, value: list [rate_sum, rate_count]
    record_tmp = {}
    ave_score = {}

    for user, item, rate in data:
        if item not in record_tmp:
            record_tmp[item] = [0, 0]
        record_tmp[item][0] += rate
        record_tmp[item][1] += 1
    for item in record_tmp:
        # 保存item的平均得分
        ave_score[item] = round(record_tmp[item][0] / record_tmp[item][1], 3)
    return ave_score

def getGenre_old(genre_str):
    """
    param:
        genre_str: 字符串 如"[{'id': 18, 'name': 'Drama'}, {'id': 35, 'name': 'Comedy'}, {'id': 10749, 'name': 'Romance'}]"
    return:
        genre_list: 风格列表 ['Drama', 'Comedy', 'Romance']
    """
    genre_list = []
    empty = 0
    # 当genres字段为"[]"时，跳过
    if genre_str == "[]":
        empty = 1
        return genre_list, empty
    # 字符串处理，先按"'}"分割字符串
    raw_list = genre_str.strip().split("'}")
    # 通过查找genre_str中最后一个"'"符号的位置，截取风格单词
    for genre_str in raw_list:
        idx = genre_str.rfind("'")
        # 分割后最后一个元素是"]"，此时跳过
        if idx == -1:
            continue
        # 获取风格元素
        genre_list.append(genre_str[idx + 1:])
    return genre_list, empty

def getGenre(genre_str):
    """
    param:
        genre_str: 字符串 如"Drama|Mystery|Romance|"
    return:
        genre_list: 风格列表 ['Drama', 'Mystery', 'Romance']
    """
    genre_list = []
    empty = 0
    # 当genres字段为空时，跳过
    if genre_str == "" or genre_str == "unknown":
        empty = 1
        return genre_list, empty
    # 字符串处理，按"|"分割字符串,list最后一个元素为空，所以去掉
    genre_list = genre_str.strip().split("|")

    return genre_list, empty

# 获得item-genre排列表和genre-item倒排表
def getItemgenre(movie_info_path, ave_score, K=50):
    """
    param:
        movie_info_path: 电影信息文件
        ave_score: item的全局平均得分字典 type dict, key:item, value: ratio
        K: genre_item排列表中记录前K个得分最高的item
    return: 
        item_genre： item-genre排列表，记录每部电影的风格，以及风格的权重比， \
                     type dict, key: item, value:dict, key:genre, value:ratio
        genre_item： genre-item倒排表，记录每种风格中评分高的电影，\
                     type dict, key: genre, value:list [item1, item2, ...]
    """

    item_genre = {}
    genre_item = {}
    # 中间记录字典 key:genre, value: dict, key:item, value:ave_score
    record_tmp = {}
    # 读取电影信息
    movie_metadata = pd.read_csv(movie_info_path, usecols=[0, 6])

    for item, genres in movie_metadata.values:
        # 存储每部电影的风格
        print(item, genres)
        genre_list, f_void = getGenre(genres)
        if f_void:
            continue
        # 电影中每种风格的比重是=1/总的风格个数
        ratio = round(1.0 / len(genre_list), 3)

        if item not in item_genre:
            item_genre[item] = {}
        for genre in genre_list:
            item_genre[item][genre] = ratio

    # 遍历item-genre排列表，生成genre-item的中间表，记录每种风格下的电影item以及得分
    for item in item_genre:
        for genre in item_genre[item]:
            if genre not in record_tmp:
                record_tmp[genre] = {}
            # 赋值item的平均得分，如果在item_ave_score字典中没有该item，赋值为0
            record_tmp[genre][item] = ave_score.get(item, 0)

    # 遍历genre-item中间表，生成genre-item排列表，每种风格记录前K部评分最高电影
    for genre in record_tmp:
        if genre not in genre_item:
            genre_item[genre] = []
        for item, score in sorted(record_tmp[genre].iteritems(), key=operator.itemgetter(1), reverse=True)[:K]:
            genre_item[genre].append(item)
    return item_genre, genre_item


# 获得用户画像，即用户最喜爱的电影类型
def getUserProfile(data, item_genre, K=3):
    """
    param:
        data: type ndarray [[user, item, rating]]
        item_genre: item-genre排列表，记录每部电影的风格，以及风格的权重比， \
                    type dict, key: item, value:dict, key:genre, value:ratio
        K: 用户最喜爱的电影类型个数 
    return: 
        user_profile: 记录用户喜爱的风格和兴趣度，type:dict, key:user, value:list [genre, score]
    """

    # 得分阈值，大于等于阈值设置为喜欢
    score_thr = 4.0
    # user-genre中间表，key:user, value:dict, key:genre, value:score
    record_tmp = {}
    user_profile = {}
    for user, item, rate, ts in data:
        # 小于阈值的数据忽略
        if rate < score_thr:
            continue
        # 如果item不在item-genre排列表中，也跳过
        if item not in item_genre:
            continue

        if user not in record_tmp:
            record_tmp[user] = {}
        for genre in item_genre[item]:
            if genre not in record_tmp[user]:
                # 初始化用户对每种风格的电影兴趣度
                record_tmp[user][genre] = 0
            # 用户风格兴趣度=电影评分*电影中该类型的权重比*时间权重
            record_tmp[user][genre] += rate * item_genre[item][genre] * getTimeScore(int(ts))

    for user in record_tmp:
        if user not in user_profile:
            user_profile[user] = []
        total_score = 0
        for genre, score in sorted(record_tmp[user].iteritems(), key=operator.itemgetter(1), reverse=True)[:K]:
            user_profile[user].append((genre, score))
            total_score += score
        # 遍历用户喜欢的风格
        for index in range(len(user_profile[user])):
            # 将每种风格的得分归一化
            user_profile[user][index] = (user_profile[user][index][0], \
                                         round(user_profile[user][index][1] / total_score, 3))
    return user_profile


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


# 推荐系统
def recommendation(genre_item, user_profile, user, user_item, hot_rank, R=30):
    """
    param:
        genre_item: genre-item排列表，记录每种风格中评分高的电影，\
                   type dict, key: genre, value:list [item1, item2, ...]
        user_profile: 记录用户喜爱的风格和兴趣度，type:dict, key:user, value:list [genre, score]
        user: 用户id
        user_item: 用户-电影排列表 type:dict, key=user, value=dict, key=item, value=rate
        hot_rank: 热门电影列表, type:dict, key:user, value:dict, key:item, value:sim
        R: 推荐列表中电影个数
    return: 
        recom_result: 推荐列表 type:dict, key:user, value:list [item1, item2, ...]
    """

    # 用户已观看的电影集合
    watched_item_list = user_item[user]

    recom_result = {}
    if user not in recom_result:
        recom_result[user] = []
    # 如果没有该用户的用户画像，推荐热门电影
    if user not in user_profile:
        recom_result[user] = hot_rank.keys()
    else:
        for genre, ratio in user_profile[user]:
            # 按照用户对电影风格的喜爱比重计算该风格推荐个数，向上取整
            num = int(R * ratio) + 1
            if genre not in genre_item:
                continue
            candidate = [item for item in genre_item[genre] if item not in watched_item_list]
            recom_list = candidate[:num]
            recom_result[user].extend(recom_list)
    return recom_result


if __name__ == '__main__':
    start = time.time()
    # 读取数据
    df_train = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v1/v1_train.csv')

    df_test = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v1/v1_test.csv', \
                          usecols=[0, 1, 2])


    # print(df_train.sample(5))
    # 推荐电影数
    reco_num = 30
    # 计算item平均得分
    ave_score = getItemAveScore(df_train[['userId', 'movieId', 'rating']].values)

    movie_info_path = '/home/zwj/Desktop/recommend/movielens/moive_database/v1/v1_movie_info.csv'
    # 获取item-genre和genre-item排列表
    item_genre, genre_item = getItemgenre(movie_info_path, ave_score)
    # 用户画像
    user_profile = getUserProfile(df_train.values, item_genre)
    # 热门电影列表
    hot_movie = getHotItem(df_train[['userId', 'movieId', 'rating']], reco_num)
    # 生成user-tiem排列表
    user_item = userItemDict(df_train[['userId', 'movieId', 'rating']].values)

    # 定义test集的推荐字典
    test_reco_list = {}
    for test_user in df_test['userId'].unique():
        recom_result = recommendation(genre_item, user_profile, test_user, user_item, hot_movie, reco_num)
        # 合并到总的推荐字典中
        test_reco_list.update(recom_result)
    # user-item排列表
    test_user_item = userItemDict(df_test.values)
    # 计算召回率和准确率
    recall, precision = precisionRecall(test_user_item, test_reco_list)

    print(recall, precision)
    print('time: ', time.time() - start)
