# coding=utf-8
import os
import gensim
import h5py
import pandas as pd
import numpy as np

# 从word2vec工具生成的文件中加载item_vec
def loadItemVec(input_file, vec_len):
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
        # 因为本例中词向量是256维，加上userid应该有257个值
        if len(item) <= vec_len:
            continue
        itemid = item[0]
        # "</s>"去掉文档中此符号的词向量
        if itemid == "</s>":
            continue
        item_vec[int(itemid)] = np.array([float(x) for x in item[1:]])
    fp.close()
    return item_vec

def string2vec(df, model, match_col, match_value, select_col):
    string = df[df[match_col]==match_value][select_col].values
    print(string)
    str_words = gensim.utils.simple_preprocess(string[0])
    vector = model.infer_vector(str_words)
    return vector

if __name__ == '__main__':

    df = pd.read_csv('/home/zwj/Desktop/recommend/movielens/u1_test_filterd.csv', \
                 usecols = [1,2,3])
    df_test = df.iloc[0:5]
    print(df_test)

    item_vector = loadItemVec('/home/zwj/Desktop/recommend/movielens/data_process/u1_train_filterd_itemvec_144.txt', 144)
    user_vector = loadItemVec('/home/zwj/Desktop/recommend/movielens/data_process/u1_train_filterd_uservec_144.txt', 144)

    df_movie_info = pd.read_csv('/home/zwj/Desktop/recommend/movielens/u_item_info.csv', \
                                usecols=[1, 2, 4, 7], \
                                header=0, \
                                names=['movieid', 'title', 'overview', 'genres'])

    model_title = gensim.models.Word2Vec.load('/home/zwj/Desktop/NLP/doc2vec/train_data/model/title_144.model')
    model_overview = gensim.models.Word2Vec.load('/home/zwj/Desktop/NLP/doc2vec/train_data/model/overview_144.model')
    model_genre = gensim.models.Word2Vec.load('/home/zwj/Desktop/NLP/doc2vec/train_data/model/genres_144.model')

    datas = np.zeros((1, 432), dtype=np.float64)
    labels = []
    for raw in df_test.values:
        # label
        labels.append(raw[2])
        # data
        # user
        if raw[1] not in user_vector:
            user_vec = np.zeros((144,), dtype=np.float64)
        else:
            user_vec = user_vector[raw[1]]

        # item
        if raw[2] not in item_vector:
            item_vec = np.zeros((144,), dtype=np.float64)
        else:
            item_vec = item_vector[raw[2]]

        # item title
        title_vec = string2vec(df_movie_info, model_title, 'movieid', raw[1], 'title')

        # item overview
        overview_vec = string2vec(df_movie_info, model_overview, 'movieid', raw[1], 'overview')
        # item genres
        data = np.concatenate((user_vec, item_vec, title_vec, overview_vec), axis=0)
        data = np.reshape(data, (1, -1))

        datas = np.concatenate((datas, data), axis=0)

    datas = np.delete(datas, 0, 0)
    datas = np.reshape(datas, (-1, 3, 12, 12))
    labels = np.array(labels)
    print(datas.shape, labels.shape)

    # fhandle = h5py.File('/home/zwj/Desktop/recommend/movielens/use_data/test.hd5','w')
    # fhandle.create_dataset('data', data = datas, compression = 'gzip', compression_opts = 4)
    # fhandle.create_dataset('label', data = labels, compression = 'gzip', compression_opts = 4)
    # fhandle.close()



