# coding=utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error

def normalizeRatings(rating, record):
    m, n =rating.shape
    #m代表电影数量，n代表用户数量
    rating_mean = np.zeros((m,1))
    #每部电影的平均得分
    rating_norm = np.zeros((m,n))
    #处理过的评分
    for i in range(m):
        idx = record[i,:] !=0
        #每部电影的评分，[i，:]表示每一行的所有列
        rating_mean[i] = np.mean(rating[i,idx])
        #第i行，评过份idx的用户的平均得分；
        #np.mean() 对所有元素求均值
        rating_norm[i,idx] = rating[i, idx] - rating_mean[i]
        #rating_norm = 原始得分-平均得分
    return rating_norm, rating_mean
	
df = pd.read_csv('/home/zwj/Desktop/recommend/netflix_prize_data/netflix_data_4178032.csv', usecols = [1, 2, 4])
m = 400000
df_train = df[0:-m]
df_test = df[-m:]
print('df Shape: {}, trainset: {}, testset: {}'.format(df.shape, len(df_train), len(df_test)))


df_p = df_train.pivot_table(index='Movie', columns='User', values='Rating')
movieNo, userNo = df_p.shape

df_p_imputed = df_p.fillna(0)
rating = df_p_imputed.values
record = rating > 0

record = np.array(record, dtype = int)
#更改数据类型，0表示用户没有对电影评分，1表示用户已经对电影评分

rating_norm, rating_mean = normalizeRatings(rating, record)

num_features = 10
X_parameters = tf.Variable(tf.random_normal([movieNo, num_features],stddev = 0.35))
Theta_parameters = tf.Variable(tf.random_normal([userNo, num_features],stddev = 0.35))
#tf.Variables()初始化变量
#tf.random_normal()函数用于从服从指定正太分布的数值中取出指定个数的值，mean: 正态分布的均值。stddev: 正态分布的标准差。dtype: 输出的类型

loss = 1/2 * tf.reduce_sum(((tf.matmul(X_parameters, Theta_parameters, transpose_b = True) - rating_norm) * record) ** 2) + 1/2 * (tf.reduce_sum(X_parameters ** 2) + tf.reduce_sum(Theta_parameters ** 2))
#基于内容的推荐算法模型

optimizer = tf.train.AdamOptimizer(1e-4)
train = optimizer.minimize(loss)
# Optimizer.minimize对一个损失变量基本上做两件事
# 它计算相对于模型参数的损失梯度。
# 然后应用计算出的梯度来更新变量。
tf.summary.scalar('loss',loss)
#用来显示标量信息

summaryMerged = tf.summary.merge_all()
#merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。
filename = './movie_tensorborad'
writer = tf.summary.FileWriter(filename)
#指定一个文件用来保存图。
sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)
#运行

for i in range(5000):
    _, movie_summary = sess.run([train, summaryMerged])
    # 把训练的结果summaryMerged存在movie里
    writer.add_summary(movie_summary, i)
    # 把训练的结果保存下来

Current_X_parameters, Current_Theta_parameters = sess.run([X_parameters, Theta_parameters])
# Current_X_parameters为用户内容矩阵，Current_Theta_parameters用户喜好矩阵
predicts = np.dot(Current_X_parameters,Current_Theta_parameters.T) + rating_mean
# dot函数是np中的矩阵乘法，np.dot(x,y) 等价于 x.dot(y)
valid = record > 0
errors = np.sqrt(np.mean((predicts[valid] - rating[valid])**2))
# sqrt(arr) ,计算各元素的平方根

print(errors)
rmse = np.sqrt(mean_squared_error(y_true=rating[valid], y_pred=predicts[valid]))
print(rmse)
