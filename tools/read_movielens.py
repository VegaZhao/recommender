# coding=utf-8
import pandas as pd

### u_data
file = '/home/zwj/Desktop/recommend/movielens/ml-100k/u1.base'


f = open(file, 'r')

data_dict = {'User':[], 'Movie':[], 'Rating':[], 'Timestamp':[]}
for line in f.readlines():
    (userid, movieid, rating,ts) = line.strip().split('\t')
    data_dict['User'].append(int(userid))
    data_dict['Movie'].append(int(movieid))
    data_dict['Rating'].append(int(rating))
    data_dict['Timestamp'].append(int(ts))
f.close()
print(len(data_dict['User']))
df = pd.DataFrame(data=data_dict)
print(len(df))
print(df.iloc[0:5])
# 转换列的顺序
cols = list(df)
print(cols)
cols.insert(0, cols.pop(cols.index('User')))
print(cols)
df_data = df.loc[:, cols]
df_data.to_csv('/home/zwj/Desktop/recommend/movielens/u1_train.csv')

### item info
"""
# coding=utf-8
import pandas as pd

file = '/home/zwj/Desktop/recommend/movielens/ml-100k/u.item'


f = open(file, 'r')

item_dict = {'id':[], 'title':[], 'year':[]}
count = 0
for line in f.readlines():
    count += 1
    iteminfo = line.strip().split('|')
    if iteminfo[2] == '':
        continue
    tmp_end_idx1 = iteminfo[1].find('(1')  # 1000 to 1999
    tmp_end_idx2 = iteminfo[1].find('(2')  # 2000 to 2999
    title_end_idx = tmp_end_idx1 if tmp_end_idx1 > -1 else tmp_end_idx2

    item_dict['id'].append(int(iteminfo[0]))
    item_dict['title'].append(iteminfo[1][:title_end_idx].strip())
    item_dict['year'].append(int(iteminfo[1][title_end_idx+1:title_end_idx+5]))


f.close()

df = pd.DataFrame(data=item_dict)

print(count, len(df))
print(df.iloc[0:5])
# 转换列的顺序
# cols = list(df)
# print(cols)
# cols.insert(0, cols.pop(cols.index('User')))
# print(cols)
# df_data = df.loc[:, cols]
df.to_csv('/home/zwj/Desktop/recommend/movielens/u_item.csv')
"""

### u_data filter
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