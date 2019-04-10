# coding=utf-8
import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# 解决ascii 编码字符串转换成"中间编码" unicode 时由于超出了其范围的问题
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)


# 加载电影信息，将电影名称设置为dataframe索引
movie_metadata = pd.read_csv('/home/zwj/Desktop/recommend/netflix_prize_data/movies_metadata.csv',\
        low_memory=False)[['title', 'overview']].set_index('title').dropna()

# 测试的时候减少数据量，只取前100部电影信息
# movie_metadata = movie_metadata_ori.iloc[:100]

print('Shape Movie-Metadata:\t{}'.format(movie_metadata.shape))
# print(movie_metadata.iloc[0:10])

# 创建tf-idf矩阵，用于比较电影简介的相似度
# stop_words='english'使用英语内建的停用词列表
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movie_metadata['overview'].dropna())


# # 比较所有电影简介之间的相似度
# similarity = cosine_similarity(tfidf_matrix)
#
# # 剔除自身的相似度
# similarity -= np.eye(similarity.shape[0])

# 设置需要查找相似电影的电影名
movie = 'Sabrina'

# 返回该电影在数据集中的位置索引
index = movie_metadata.reset_index(drop=True)[movie_metadata.index==movie].index[0]

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

# 设置相似电影的个数
n = 10

# 获取前n部最相似电影的索引和相似度
similar_movies_index = np.argsort(sim_movie)[::-1][:n]
similar_movies_score = np.sort(sim_movie)[::-1][:n]

# 获得相似电影的名称
similar_movie_titles = movie_metadata.iloc[similar_movies_index].index

for movie in similar_movie_titles:
    print(movie)