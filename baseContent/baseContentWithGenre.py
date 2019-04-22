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


def getItemAveScore(data):
    record = {}
    ave_score = {}

    for user, item, rate in data:
        if item not in record:
            record[item] = [0, 0]
        record[item][0] += rate
        record[item][1] += 1
    for item in record:
        ave_score[item] = round(record[item][0] / record[item][1], 3)
    return ave_score

def getItemgenre(movie_info_path, ave_score, K=5):
    item_genre = {}
    record = {}
    genre_item = {}
    movie_metadata_raw = pd.read_csv(movie_info_path, usecols=[1, 3, 7]).dropna()
    # print(movie_metadata_raw.sample(5))
    for item, title, genres in movie_metadata_raw.values:
        genre_list = []
        if genres == "[]":
            continue
        raw_list = genres.strip().split("'}")

        for genre_str in raw_list:
            idx = genre_str.rfind("'")
            if idx == -1:
                continue
            genre_list.append(genre_str[idx+1:])

        ratio = round(1.0/len(genre_list), 3)

        if item not in item_genre:
            item_genre[item] = {}
        for genre in genre_list:
            item_genre[item][genre] = ratio

    for item in item_genre:
        for genre in item_genre[item]:
            if genre not in record:
                record[genre] = {}
            record[genre][item] = ave_score.get(item, 0)

    for genre in record:
        if genre not in genre_item:
            genre_item[genre] = []
        for item, score in sorted(record[genre].iteritems(), key=operator.itemgetter(1), reverse=True)[:K]:
            genre_item[genre].append(item)
    return item_genre, genre_item

def getUserProfile(data, item_genre, K=2):
    score_thr = 4
    record = {}
    user_profile = {}
    for user, item, rate in data:
        if rate < score_thr:
            continue
        if item not in item_genre:
            continue
        if user not in record:
            record[user] = {}
        for genre in item_genre[item]:
            if genre not in record[user]:
                record[user][genre] = 0
            record[user][genre] += rate*item_genre[item][genre]
    for user in record:
        if user not in user_profile:
            user_profile[user] = []
        total_score = 0
        for genre, score in sorted(record[user].iteritems(), key = operator.itemgetter(1), reverse = True)[:K]:
            user_profile[user].append((genre, score))
            total_score += score

        for index in range(len(user_profile[user])):
            user_profile[user][index] = (user_profile[user][index][0], round(user_profile[user][index][1] / total_score,3))
    return user_profile

def recommendation(genre_item, user_profile, user, K=5):
    recom_result = {}
    if user not in user_profile:
        return {}
    if user not in recom_result:
        recom_result[user] = []
    for genre, ratio in user_profile[user]:
        num = int(K*ratio) + 1
        if genre not in genre_item:
            continue
        recom_list = genre_item[genre][:num]
        recom_result[user].extend(recom_list)
    return recom_result

def getTimeScore(timestamp):
    fix_time_stamp = 1476086345
    total_sec = 24*60*60
    delta = (fix_time_stamp - timestamp) / total_sec
    return round(1/(1+delta), 3)


if __name__ == '__main__':

    start = time.time()
    # 读取数据,这是没有shuffle的数据
    df_train = pd.read_csv('/home/zwj/Desktop/recommend/small_data/movie_train_s.csv', \
                          usecols=[1, 2, 3])

    df_test = pd.read_csv('/home/zwj/Desktop/recommend/small_data/movie_test_s.csv', \
                          usecols=[1, 2, 3])
    # print(df_train.sample(5))
    # 推荐电影数
    reco_num = 5
    ave_score = getItemAveScore(df_train.values)

    movie_info_path = '/home/zwj/Desktop/recommend/small_data/movie_info_s.csv'
    item_genre, genre_item = getItemgenre(movie_info_path, ave_score)

    user_profile = getUserProfile(df_train.values, item_genre)
    recom_result = recommendation(genre_item, user_profile, 622594)
    print(recom_result)
