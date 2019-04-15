# coding=utf-8
# 修改DataFrame列的顺序
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

df = pd.read_csv('/home/zwj/Desktop/recommend/download/normalize_data/filterd_u3065_m374.csv', \
                 usecols = [1, 2, 3, 4])

# 转换列的顺序
cols = list(df)
cols.insert(1, cols.pop(cols.index('Movie')))
df_data = df.loc[:, cols]

# shuffle数据
df = shuffle(df_data)

df.to_csv('/home/zwj/Desktop/recommend/download/normalize_data/shuffled_u3065_m374.csv')
