# coding=utf-8
import pandas as pd
import time
import operator
import tensorflow as tf
import numpy as np

# 将列表转成user-item字典
def userItemDict(data):
    """
    param：
        data: lsit [user, item, rating]
    return：
        user_item: 用户-电影排列表 type:dict, key:user, value:dict, key:item, value: rate
    """
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
        hot_rank[item_id] = 1 - 0.01 * r
        r += 1
    return hot_rank

def predict(input_x):
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('/home/zwj/Desktop/recommend/movielens/use_data/model/model.ckpt-5.meta')
        new_saver.restore(sess, '/home/zwj/Desktop/recommend/movielens/use_data/model/model.ckpt-5')
        # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
        prediction = tf.get_collection('pred_network')[0]
        graph = tf.get_default_graph()

        x = graph.get_operation_by_name('input_x').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0] # 使用y进行预测
        output= prediction.eval(feed_dict={x:input_x, keep_prob:1.0})
        return output[0,0]

def recommendation(user_item, user_id, item_set, hot_rank, R):
    """
    param：
        model: 训练的模型
        user_item: 训练集中user-item字典 {user1 : {item1 : rate1, item2 : rate2}, ...}}
        user_id：推荐的用户id
        item_set: 训练集中的电影集合
        hot_rank: 热门电影列表
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
        item_watched_list = user_item[user_id]

        for item_id in item_set:
            if item_id in item_watched_list:
                continue
            rank[user_id].setdefault(item_id, 0)

            # 将模型预测结果赋给rank[user_id][item_id]
            input = np.array([[user_id, item_id]])
            rank[user_id][item_id] = predict(input)

    # 推荐列表按评分由高到低排序
    rank_sorted = {}
    rank_sorted[user_id] = sorted(rank[user_id].items(), key=operator.itemgetter(1), reverse=True)[0:R]

    return rank_sorted


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



if __name__ == '__main__':

    start = time.time()
    # 读取数据
    df_train = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v1/v1_train.csv', \
                           usecols=[0, 1, 2])

    df_test = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v1/v1_test.csv', \
                          usecols=[0, 1, 2])
    df_test = df_test.iloc[0:1]
    # sample num: 35497
    print(len(df_train), len(df_test))

    # 推荐电影数
    reco_num = 30

    hot_rank = getHotItem(df_train, reco_num)

    # 生成user-tiem排列表
    user_item = userItemDict(df_train.values)
    test_user_item = userItemDict(df_test.values)

    # 定义test集的推荐字典
    test_reco_list = {}
    for test_user in df_test['userId'].unique():
        # 生成单用户推荐列表
        rank_list = recommendation(user_item, test_user, df_train['movieId'].unique(), hot_rank, reco_num)
        # 合并到总的推荐字典中
        test_reco_list.update(rank_list)

    recall, precision = precisionRecall(test_user_item, test_reco_list)

    print('\n\nTesting Result:Precision={:.4f}\t Recall={:.4f}'.format(precision, recall))
    print(recall, precision)
    print('time: ', time.time() - start)