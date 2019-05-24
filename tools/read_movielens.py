# coding=utf-8
import pandas as pd

### user_data
file = '/home/zwj/Desktop/recommend/movielens/ml-1m/users.dat'

occupation = {0:"other", 1:  "academic/educator", 2:  "artist", 3:  "clerical/admin", 4:  "college/grad student",\
              5:  "customer service", 6:  "doctor/health care", 7:  "executive/managerial", 8:  "farmer", 9:  "homemaker",\
              10:  "K-12 student", 11:  "lawyer", 12:  "programmer", 13:  "retired", 14:  "sales/marketing", 15:  "scientist",\
              16:  "self-employed", 17:  "technician/engineer", 18:  "tradesman/craftsman", 19:  "unemployed", 20:  "writer"}

f = open(file, 'r')
data_dict = {'id':[], 'age':[], 'gender':[], 'occupation':[]}
for line in f.readlines():
    (id, gender, age, ioc) = line.strip().split('::')[:-1]
    print(id, gender, age, occupation[int(ioc)])
    data_dict['id'].append(int(id))
    data_dict['age'].append(int(age))
    data_dict['gender'].append(gender)
    data_dict['occupation'].append(occupation[int(ioc)])
f.close()

print(len(data_dict['id']))
df = pd.DataFrame(data=data_dict)
print(len(df))
print(df.iloc[0:5])
# # 转换列的顺序
cols = list(df)
print(cols)
cols.insert(0, cols.pop(cols.index('id')))
print(cols)
df_data = df.loc[:, cols]
print(df_data.iloc[0:5])
df_data.to_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v2/v2_user_info.csv')

### rating data
"""
# coding=utf-8
import pandas as pd

### u_data
file = '/home/zwj/Desktop/recommend/movielens/ml-10M100K/ratings.dat'


f = open(file, 'r')
data_dict = {'User':[], 'Movie':[], 'Rating':[], 'Timestamp':[]}
count = 0
for line in f.readlines():
    count += 1
    (u, m, r, t) = line.strip().split('::')
    # print(u, m, r, t)
    data_dict['User'].append(int(u))
    data_dict['Movie'].append(int(m))
    data_dict['Rating'].append(float(r))
    data_dict['Timestamp'].append(int(t))
f.close()
print('count=',count)
print(len(data_dict['User']))
df = pd.DataFrame(data=data_dict)
print(len(df))
print(df.iloc[0:5])
# # # 转换列的顺序
cols = list(df)
print(cols)
cols.insert(0, cols.pop(cols.index('User')))
print(cols)
df_data = df.loc[:, cols]
print(df_data.iloc[0:5])
df_data.to_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v3/v3_ratings.csv')
"""


### movie.csv
"""
# coding=utf-8
import pandas as pd

### u_data
file = '/home/zwj/Desktop/recommend/movielens/ml-1m/movies.dat'


f = open(file, 'r')

item_dict = {'id':[], 'title':[], 'year':[], 'genres':[]}
count = 0
for line in f.readlines():
    count += 1
    iteminfo = line.strip().split('::')
    # print(iteminfo)
    # if iteminfo[2] == '':
    #     continue
    tmp_end_idx1 = iteminfo[1].rfind('(1')  # 1000 to 1999
    tmp_end_idx2 = iteminfo[1].rfind('(2')  # 2000 to 2999
    title_end_idx = tmp_end_idx1 if tmp_end_idx1 > -1 else tmp_end_idx2

    item_dict['id'].append(int(iteminfo[0]))
    item_dict['title'].append(iteminfo[1][:title_end_idx].strip())
    item_dict['year'].append(int(iteminfo[1][title_end_idx+1:title_end_idx+5]))
    item_dict['genres'].append(iteminfo[2])


f.close()

df = pd.DataFrame(data=item_dict)

print(count, len(df))
print(df.iloc[0:5])
# 转换列的顺序
cols = list(df)
print(cols)
cols.insert(3, cols.pop(cols.index('genres')))
print(cols)
df_data = df.loc[:, cols]
print(df_data.iloc[0:5])
df.to_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v2/v2_movies.csv')
"""


### item/user word
"""
# coding=utf-8
import pandas as pd

# 将列表转成itemWord字典
def itemWord(data):
    # 输入：
    #   data: type:ndarray [[user, item, rating],[...]]
    # 将data转成如下字典格式，注意itemid存储的是字符串
    # {userid1: ['itemid1', 'itemid2'], userid2: [...], ...}

    user_itemWord = {}
    itemset = []
    for user, item, rate in data:
        if item not in itemset:
            itemset.append(item)
        if user not in user_itemWord:
            # int()是为了最后去掉小数点.0的部分
            user_itemWord[int(user)] = []
        # str()将itemid转成字符串，因为join的对象必须是字符串
        user_itemWord[user].append(str(int(item)))
    # 保存user_itemWord中values部分['itemid1', 'itemid2']，每个用户的item集合为一行，看作一个句子

    # fw = open('/home/zwj/Desktop/recommend/movielens/data_process/u1_train_filterd_itemWord.txt', 'w+')
    # for user in user_itemWord:
    #     fw.write(" ".join(user_itemWord[user]) + "\n")
    # fw.close()
    # 后续使用word2vec工具将此txt文件转成词向量文件，文件中一个向量表示一个词（本例中为itemid）
    print len(itemset)
    print len(set(itemset))


def userWord(data):

    item_userWord = {}
    userset = []
    for user, item, rate in data:
        if user not in userset:
            userset.append(user)
        if item not in item_userWord:
            # int()是为了最后去掉小数点.0的部分
            item_userWord[int(item)] = []
        # str()将itemid转成字符串，因为join的对象必须是字符串
        item_userWord[item].append(str(int(user)))
    # 保存user_itemWord中values部分['itemid1', 'itemid2']，每个用户的item集合为一行，看作一个句子

    # fw = open('/home/zwj/Desktop/recommend/movielens/data_process/u1_train_filterd_userWord.txt', 'w+')
    # for item in item_userWord:
    #     fw.write(" ".join(item_userWord[item]) + "\n")
    # fw.close()
    # 后续使用word2vec工具将此txt文件转成词向量文件，文件中一个向量表示一个词（本例中为itemid）
    print(len(userset))
    print(len(set(userset)))

df = pd.read_csv('/home/zwj/Desktop/recommend/movielens/u1_train_filterd.csv', usecols=[1,2,3])

itemWord(df.values)
userWord(df.values)
"""
### normalize_genres
"""
# coding=utf-8
import pandas as pd

df = pd.read_csv('/home/zwj/Desktop/recommend/movielens/u_item_info.csv', usecols=[1,2,3,4,5,6,7])

# print(df.sample(5))
print(df.columns)

def normalize_genres(genres):
    genres_str = ''
    if genres == "[]":
        genres_str += 'unknown'
    else:
        # 字符串处理，先按"'}"分割字符串
        raw_list = genres.strip().split("'}")
        # 通过查找genre_str中最后一个"'"符号的位置，截取风格单词
        for cate in raw_list[:-1]:
            idx = cate.rfind("'")
            genres_str += cate[idx + 1:]
            genres_str += '|'
    return genres_str

df['genres'] = df['genres'].apply(lambda x:normalize_genres(x))

# title = df['title'].values
# genres = df['genres_norm'].values
# overview = df['overview'].values
#
# with open('/home/zwj/Desktop/NLP/doc2vec/train_data/genres.txt', 'w') as f:
#     f.write('\n'.join(genres))
df.to_csv('/home/zwj/Desktop/recommend/movielens/u_item_info_new.csv')
"""



## split train&test
"""
# coding:utf-8
import pandas as pd
from sklearn.utils import shuffle

df_raw = pd.read_csv('/home/zwj/Desktop/recommend/movielens/ml-latest-small/ratings.csv')

print df_raw.sample(5)

df_shuffle=shuffle(df_raw)
del df_raw

thres = int(len(df_shuffle)/5.0*4.0)
print(len(df_shuffle), thres)
df_train = df_shuffle.iloc[0:thres]
df_test = df_shuffle.iloc[thres:]

print(len(df_train), len(df_test))

df_train.to_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v4/v4_train.csv', index=False)
df_test.to_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v4/v4_test.csv', index=False)
"""

### statistics
"""
# coding:utf-8
import pandas as pd
import time
import datetime
####1
# df_raw = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v4/v4_movie_info.csv')
#
# print df_raw.sample(5)
#
# print('movies num:', len(df_raw))
#
# y = df_raw['year'].unique()
# print y
# print(max(y), min(y))

###2
# df_raw = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v4/v4_train.csv')
#
# def statistics(df):
#
#     movie_statistics = df.groupby('movieId')['rating'].count().sort_values(ascending=False)
#     user_statistics = df.groupby('userId')['rating'].count().sort_values(ascending=False)
#     movie_num = len(df['movieId'].unique())
#     user_num = len(df['userId'].unique())
#
#     return movie_num, user_num, movie_statistics, user_statistics
# movie_num, user_num, movie_statistics, user_statistics = statistics(df_raw)
#
# print(len(df_raw), movie_num, user_num)
# print(movie_statistics)
# print(user_statistics)

###3
# df = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v4/v4_train.csv')
# df2 = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v4/v4_test.csv')
#
# df = pd.concat([df, df2])
#
# def statistics(df):
#
#     movie_statistics = df.groupby('movieId')['rating'].count().sort_values(ascending=False)
#     user_statistics = df.groupby('userId')['rating'].count().sort_values(ascending=False)
#     movie_num = len(df['movieId'].unique())
#     user_num = len(df['userId'].unique())
#
#     return movie_num, user_num, movie_statistics, user_statistics
# movie_num, user_num, movie_statistics, user_statistics = statistics(df)
#
# print(len(df), movie_num, user_num)
# print(max(df['timestamp']), min(df['timestamp']))

### timeStamp 2 date
timeStamp = 828124615
timeArray = time.localtime(timeStamp)
otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
print otherStyleTime   # 2013--10--10 23:40:00

timeStamp = 1537799250
timeArray = time.localtime(timeStamp)
otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
print otherStyleTime   # 2013--10--10 23:40:00
"""

### 1 cvs file movie
"""
# coding=utf-8
import pandas as pd

df = pd.read_csv('/home/zwj/Desktop/recommend/movielens/ml-latest-small/movies.csv')

item_dict = {'id':[], 'title':[], 'year':[], 'genres':[]}
count = 0
for iteminfo in df.values:
    count += 1
    # print(iteminfo)
    # if iteminfo[2] == '':
    #     continue
    tmp_end_idx1 = iteminfo[1].rfind('(1')  # 1000 to 1999
    tmp_end_idx2 = iteminfo[1].rfind('(2')  # 2000 to 2999
    title_end_idx = tmp_end_idx1 if tmp_end_idx1 > -1 else tmp_end_idx2
    # no year
    if title_end_idx == -1:
        continue
    item_dict['id'].append(int(iteminfo[0]))
    item_dict['title'].append(iteminfo[1][:title_end_idx].strip())
    item_dict['year'].append(int(iteminfo[1][title_end_idx+1:title_end_idx+5]))
    item_dict['genres'].append(iteminfo[2])

df = pd.DataFrame(data=item_dict)

print(count, len(df))
print(df.iloc[0:5])
# # 转换列的顺序
cols = list(df)
print(cols)
cols.insert(3, cols.pop(cols.index('genres')))
print(cols)
df_data = df.loc[:, cols]
print(df_data.iloc[0:5])
df.to_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v4/v4_movies.csv')
"""

## 2 movie merge
"""
# coding=utf-8
import sys
import time
import pandas as pd

# 解决ascii 编码字符串转换成"中间编码" unicode 时由于超出了其范围的问题
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# 加载电影数据
movie_titles = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v3/v3_movies.csv',
                           usecols=[1,2,3,4])

print('Shape Movie-Titles:\t{}'.format(movie_titles.shape))
# print(movie_titles.sample(5))

# 加载电影描述数据，包括名字，简介，投票总数，时长，风格
movie_metadata = pd.read_csv('/home/zwj/Desktop/recommend/netflix_prize_data/movies_metadata.csv',\
                low_memory=False)[['title', 'overview', 'release_date', 'vote_count', 'runtime']].dropna()


movie_metadata['year'] = movie_metadata['release_date'].apply(lambda x:time.strptime(x, "%Y-%m-%d").tm_year)
movie_metadata = movie_metadata.drop('release_date', axis=1)
print('Shape Movie-Metadata:\t{}'.format(movie_metadata.shape))

# print(movie_metadata.sample(5))

# 将两个电影数据合并
movie_info = pd.merge(movie_titles, movie_metadata, on=['title', 'year'])

print('Shape movie_info:\t{}'.format(movie_info.shape))

movie_info= movie_info.drop_duplicates(['id'], keep='last')
print('Shape movie_info drop duplicates:\t{}'.format(movie_info.shape))
print(movie_info.sample(5))


cols = list(movie_info)
print(cols)
cols.insert(6, cols.pop(cols.index('genres')))
print(cols)
df_data = movie_info.loc[:, cols]
print(df_data.iloc[0:5])
df_data.to_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v3/v3_movie_info.csv')

"""

### 3 u_data/ratings filter
"""
# coding=utf-8

import sys
import time
import pandas as pd

# 解决ascii 编码字符串转换成"中间编码" unicode 时由于超出了其范围的问题
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# 加载电影数据
movie_titles = pd.read_csv('/home/zwj/Desktop/recommend/movielens/u_item_info.csv', usecols=[1, 2])
# print(movie_titles.sample(5))
print(len(movie_titles))
# movie_titles= movie_titles.drop_duplicates(['id'], keep='last')

df_raw = pd.read_csv('/home/zwj/Desktop/recommend/movielens/u1_train.csv', usecols=[1, 2, 3, 4])
print('Shape df_raw:\t{}'.format(df_raw.shape))
print(df_raw.sample(5))

itemid = movie_titles['id'].values
print('itemid len:', len(itemid))

df_filterd = df_raw[df_raw['Movie'].isin(itemid)]
print('Shape df_filterd:\t{}'.format(df_filterd.shape))
print(df_filterd.sample(5))
df_filterd.to_csv('/home/zwj/Desktop/recommend/movielens/u1_train_filterd.csv')

"""