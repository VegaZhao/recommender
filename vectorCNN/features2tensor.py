# coding=utf-8
import os
import gensim
import h5py
import pandas as pd
import numpy as np


# 将句子转换成词向量
def sen2vec(df, model, match_col, match_value, select_col):
    """
    param:
        df: 电影信息文件 dataframe格式，列包含 'movieid', 'title', 'overview', 'genres'字段
        model: gensim训练好的 Doc2Vec模型
        match_col: 条件筛选的列名
        match_value: 条件筛选的值
        select_col: 选择的列
    return:
        vector: 句子返回的向量
    """
    # 获取符合筛选条件的句子
    sentence = df[df[match_col]==match_value][select_col].values
    # 经过简单的句子处理，如去到符号，将句子拆解成词列表
    sen_words = gensim.utils.simple_preprocess(sentence[0])
    # 通过模型获取词向量
    vector = model.infer_vector(sen_words)
    return vector

if __name__ == '__main__':

    # 加载原始数据集
    df = pd.read_csv('/home/zwj/Desktop/recommend/movielens/u1_test_filterd.csv', \
                 usecols = [1,2,3])

    # 加载电影信息表
    df_movie_info = pd.read_csv('/home/zwj/Desktop/recommend/movielens/u_item_info.csv', \
                                usecols=[1, 2, 4, 7], \
                                header=0, \
                                names=['movieid', 'title', 'overview', 'genres'])
    # 加载model
    model_item = gensim.models.Word2Vec.load('/home/zwj/Desktop/recommend/movielens/data_process/model/u1_itemvec_144.model')
    model_user = gensim.models.Word2Vec.load('/home/zwj/Desktop/recommend/movielens/data_process/model/u1_uservec_144.model')
    model_title = gensim.models.Word2Vec.load('/home/zwj/Desktop/NLP/doc2vec/train_data/model/title_144.model')
    model_overview = gensim.models.Word2Vec.load('/home/zwj/Desktop/NLP/doc2vec/train_data/model/overview_144.model')
    model_genre = gensim.models.Word2Vec.load('/home/zwj/Desktop/NLP/doc2vec/train_data/model/genres_144.model')

    # 初始化数据，第一行会删除掉
    datas = np.zeros((1, 720), dtype=np.float64)
    # 初始化标签
    labels = []
    # 遍历原始数据集中的每一条记录
    count = 0
    for raw in df.values:
        count += 1
        print(count)
        # 获取label
        labels.append(raw[2])
        # 获取data
        # user
        # 如果模型中没有这个word，则返回全零
        try:
            user_vec = model_user.wv[str(raw[0])]
        except:
            user_vec = np.zeros((144,), dtype=np.float64)
        # item
        try:
            item_vec = model_item.wv[str(raw[1])]
        except:
            item_vec = np.zeros((144,), dtype=np.float64)

        # item title
        title_vec = sen2vec(df_movie_info, model_title, 'movieid', raw[1], 'title')

        # item overview
        overview_vec = sen2vec(df_movie_info, model_overview, 'movieid', raw[1], 'overview')

        # item genres
        genres_vec = sen2vec(df_movie_info, model_genre, 'movieid', raw[1], 'genres')

        # 将以上5个特征的向量数据拼接
        data = np.concatenate((user_vec, item_vec, title_vec, overview_vec, genres_vec), axis=0)
        data = np.reshape(data, (1, -1))
        # 将所有记录的特征数据拼接
        datas = np.concatenate((datas, data))

    # 删掉初始化是定义的全零首行
    datas = np.delete(datas, 0, 0)
    # reshape数据格式 四维数据分别是（样本数，特征数，长，宽）
    datas = np.reshape(datas, (-1, 5, 12, 12))
    labels = np.array(labels)
    print(datas.shape, labels.shape)
    # 保存成h5py格式
    fhandle = h5py.File('/home/zwj/Desktop/recommend/movielens/use_data/u1_test_features5_gensim.hd5','w')
    fhandle.create_dataset('data', data = datas, compression = 'gzip', compression_opts = 4)
    fhandle.create_dataset('label', data = labels, compression = 'gzip', compression_opts = 4)
    fhandle.close()