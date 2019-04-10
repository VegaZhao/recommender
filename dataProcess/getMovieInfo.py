# coding=utf-8
import sys
import pandas as pd

# 解决ascii 编码字符串转换成"中间编码" unicode 时由于超出了其范围的问题
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# 加载电影数据
movie_titles = pd.read_csv('/home/zwj/Desktop/recommend/netflix_prize_data/movie_titles.csv',
                           encoding = 'ISO-8859-1',
                           header = None,
                           names = ['id', 'year', 'title'])

# 剔除1990年之前的电影
movie_titles = movie_titles[movie_titles['year'] > 1990.0]

print('Shape Movie-Titles:\t{}'.format(movie_titles.shape))
# print(movie_titles.sample(5))

# 加载电影描述数据，包括名字，简介，投票总数，时长，风格
movie_metadata = pd.read_csv('/home/zwj/Desktop/recommend/netflix_prize_data/movies_metadata.csv',\
                low_memory=False)[['title', 'overview', 'vote_count', 'runtime', 'genres']].dropna()

print('Shape Movie-Metadata:\t{}'.format(movie_metadata.shape))
# print(movie_metadata.sample(5))

# 将两个电影数据合并
movie_info = pd.merge(movie_titles, movie_metadata, on='title')

print('Shape movie_info:\t{}'.format(movie_info.shape))
# print(movie_info.sample(5))

movie_info.to_csv('/home/zwj/Desktop/recommend/netflix_prize_data/normalize_data/movie_info.csv')
