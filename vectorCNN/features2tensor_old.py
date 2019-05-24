# coding=utf-8
import os
import gensim
import h5py
import pandas as pd
import numpy as np

# 从word2vec工具生成的文件中加载item_vec
def loadItemVec(input_file, vec_len):
    """
    param:
        input_file: 词向量文件路径
        vec_len: 词向量维度（长度）
    return:
        item_vec: item_id转成的词向量 type:dict key:itemid values:词向量
    """
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
    df = pd.read_csv('/home/zwj/Desktop/recommend/movielens/u1_train_filterd.csv', \
                 usecols = [1,2,3])

    # 加载item/user词向量
    item_vector = loadItemVec('/home/zwj/Desktop/recommend/movielens/data_process/u1_train_filterd_itemvec_144.txt', 144)
    user_vector = loadItemVec('/home/zwj/Desktop/recommend/movielens/data_process/u1_train_filterd_uservec_144.txt', 144)

    # 加载电影信息表
    df_movie_info = pd.read_csv('/home/zwj/Desktop/recommend/movielens/u_item_info.csv', \
                                usecols=[1, 2, 4, 7], \
                                header=0, \
                                names=['movieid', 'title', 'overview', 'genres'])
    # 加载model
    model_title = gensim.models.Word2Vec.load('/home/zwj/Desktop/NLP/doc2vec/train_data/model/title_144.model')
    model_overview = gensim.models.Word2Vec.load('/home/zwj/Desktop/NLP/doc2vec/train_data/model/overview_144.model')
    model_genre = gensim.models.Word2Vec.load('/home/zwj/Desktop/NLP/doc2vec/train_data/model/genres_144.model')

    # 初始化数据，第一行会删除掉
    datas = np.zeros((1, 720), dtype=np.float64)
    # 初始化标签
    labels = []
    # 遍历原始数据集中的每一条记录
    for raw in df.values:
        # 获取label
        labels.append(raw[2])
        # 获取data
        # user
        # 如果user词向量集中没有该用户id，赋值为全零向量
        if raw[0] not in user_vector:
            # 144是词向量的维度
            user_vec = np.zeros((144,), dtype=np.float64)
        else:
            user_vec = user_vector[raw[0]]

        # item
        if raw[1] not in item_vector:
            item_vec = np.zeros((144,), dtype=np.float64)
        else:
            item_vec = item_vector[raw[1]]

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
    fhandle = h5py.File('/home/zwj/Desktop/recommend/movielens/use_data/u1_train_features5.hd5','w')
    fhandle.create_dataset('data', data = datas, compression = 'gzip', compression_opts = 4)
    fhandle.create_dataset('label', data = labels, compression = 'gzip', compression_opts = 4)
    fhandle.close()