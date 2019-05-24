# coding=utf-8
import pandas as pd

# 将列表转成itemWord字典
def itemWord(data, save_file, is_movie=True):
    """
    param:
        data: type:ndarray [[userid, movieid],[...]]
        save_file: 保存语料文件
        is_movie:  判断是movie语料还是user语料，默认是True即movie语料
    """

    # 将data转成如下字典格式，注意list中存储的是字符串
    # movie语料集 {userid1: ['movieid1', 'movieid2'], userid2: [...], ...}
    # user语料集 {movieid1: ['userid1', 'userid2'], movieid2: [...], ...}

    id_itemWord = {}
    for userid, movieid in data:
        if is_movie:
            if userid not in id_itemWord:
                # int()是为了最后去掉小数点.0的部分
                id_itemWord[int(userid)] = []
                # str()将movieid转成字符串，因为join的对象必须是字符串
                id_itemWord[int(userid)].append(str(int(movieid)))
        else:
            if movieid not in id_itemWord:
                # int()是为了最后去掉小数点.0的部分
                id_itemWord[int(movieid)] = []
                # str()将userid转成字符串，因为join的对象必须是字符串
                id_itemWord[int(movieid)].append(str(int(userid)))

    # 保存id_itemWord中values部分list的内容，每个用户的movieid集合或每部电影的userid集合为一行，看作一个句子
    fw = open(save_file, 'w+')
    for id in id_itemWord:
        fw.write(" ".join(id_itemWord[id]) + "\n")
    fw.close()

if __name__ == '__main__':
    # 生成movieid或者userid语料
    # 读取数据
    df_train = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v1/v1_train.csv', \
                          usecols=[0, 1])
    item_corpus_file = '/home/zwj/Desktop/recommend/movielens/moive_database/v1/train_itemWord.txt'
    itemWord(df_train.values, item_corpus_file, is_movie=True)
    
    # 生成title/genres/overview语料
    df = pd.read_csv('/home/zwj/Desktop/NLP/doc2vec/train_data/movies_metadata.csv', usecols=[3, 9, 20]).dropna()
    title = df['title'].values
    genres = df['genres'].values
    overview = df['overview'].values
    with open('/home/zwj/Desktop/NLP/doc2vec/train_data/title.txt', 'w') as f:
        f.write('\n'.join(title))
    
    with open('/home/zwj/Desktop/NLP/doc2vec/train_data/genres.txt', 'w') as f:
        f.write('\n'.join(genres))
    
    with open('/home/zwj/Desktop/NLP/doc2vec/train_data/overview.txt', 'w') as f:
        f.write('\n'.join(overview))

