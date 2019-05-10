# coding:utf-8
import pandas as pd

def readRawData():
    # 读取数据,这是没有shuffle的数据
    df_train = pd.read_csv('/home/zwj/Desktop/recommend/small_data/ft_ratings_train.csv', \
                          usecols=[1, 2, 3])

    df_test = pd.read_csv('/home/zwj/Desktop/recommend/small_data/ft_ratings_test.csv', \
                          usecols=[1, 2, 3])
    # sample num: 35497
    print(len(df_train), len(df_test))
	return df_train, df_test