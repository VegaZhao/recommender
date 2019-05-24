# coding=utf-8
# 修改DataFrame列的顺序
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

def changeDfCols(df):

    # 转换列的顺序
    cols = list(df)
    cols.insert(1, cols.pop(cols.index('Movie')))
    df_data = df.loc[:, cols]

    # shuffle数据
    df = shuffle(df_data)

    return df

def statistics(df):

    movie_statistics = df.groupby('Movie')['Rating'].count().sort_values(ascending=False)
    user_statistics = df.groupby('User')['Rating'].count().sort_values(ascending=False)
    movie_num = len(df['Movie'].unique())
    user_num = len(df['User'].unique())
    
    return movie_num, user_num, movie_statistics, user_statistics

def changeDfColsDataType(df):

    df['Rating'] = df['Rating'].astype(float)

    return df

if __name__ == '__main__':
    df = pd.read_csv('/home/zwj/Desktop/recommend/small_data/movie_train_s.csv', \
                      usecols=[1, 2, 3, 4])

    print(df.sample(5))



