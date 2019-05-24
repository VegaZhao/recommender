# coding=utf-8
import tensorflow as tf
import numpy as np
import pandas as pd

sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def read_data(df_raw):
    data = df_raw[['userId', 'movieId']].values
    label = df_raw['rating'].values[:, np.newaxis]
    return data, label
# read data
df_train = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v1/v1_train.csv',\
                       usecols=[0, 1, 2])

df_test = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v1/v1_test.csv', \
                      usecols=[0, 1, 2])

train_data, train_label = read_data(df_train)
test_data, test_label = read_data(df_test)
# df_train = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v1/v1_train.csv',\
#                        usecols=[0, 1, 2])
#
# df_test = pd.read_csv('/home/zwj/Desktop/recommend/movielens/moive_database/v1/v1_test.csv', \
#                       usecols=[0, 1, 2])
# df = pd.concat([df_train, df_test])
# user_id_mapping = {id: i for i, id in enumerate(df['userId'].unique())}
# movie_id_mapping = {id: i for i, id in enumerate(df['movieId'].unique())}
#
# train_user_data = df_train['userId'].map(user_id_mapping).values[:, np.newaxis]
# train_movie_data = df_train['movieId'].map(movie_id_mapping).values[:, np.newaxis]
# train_data = np.concatenate((train_user_data, train_movie_data), axis=1)
# train_label = df_train['rating'].values[:, np.newaxis]
#
# test_user_data = df_test['userId'].map(user_id_mapping).values[:, np.newaxis]
# test_movie_data = df_test['movieId'].map(movie_id_mapping).values[:, np.newaxis]
# test_data = np.concatenate((test_user_data, test_movie_data), axis = 1)
# test_label = df_test['rating'].values[:, np.newaxis]

# set network-----------------------------------------------------------------------
# define placeholder for inputs to network
x = tf.placeholder(tf.float32, [None, 2])
y_ = tf.placeholder(tf.float32, [None, 1])
# layer 1
W_1 = weight_variable([2, 256])
bias_1 = bias_variable([256])
h_fc1 = tf.nn.relu(tf.matmul(x, W_1) + bias_1)

# layer 2
W_2 = weight_variable([256, 512])
bias_2 = bias_variable([512])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_2) + bias_2)

# layer 3
W_3 = weight_variable([512, 1024])
bias_3 = bias_variable([1024])
h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_3) + bias_3)

# drop
keep_prob = tf.placeholder("float")
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

# output layer
W_4 = weight_variable([1024, 1])
bias_4 = bias_variable([1])
prediction = tf.matmul(h_fc3_drop, W_4) + bias_4
# network --------------------------------------------------------------------------------------------------end
# the error between prediction and real data
loss = tf.sqrt(tf.reduce_mean(tf.square(y_ - prediction)))

train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)

sess.run(tf.global_variables_initializer())

# 设置训练迭代次数
epochs = 10

# 开启模型保存或者读取
saver = tf.train.Saver(max_to_keep=1)
# 加载已训练的模型，初始化网络参数
# saver.restore(sess, "/home/zwj/Desktop/recommend/movielens/use_data/model/model.ckpt-5")

# 输出初始网络预测结果的误差
print "original test loss %g" %(loss.eval(feed_dict = {
    x:test_data, y_:test_label, keep_prob:1.0}))

# 训练与预测
for e in range(epochs):
    for i in range(869): #1739
        batch = 64
        if i%100 == 0:
            train_loss = loss.eval(feed_dict={
                x:train_data[i*batch:(i+1)*batch], y_:train_label[i*batch:(i+1)*batch], keep_prob:1.0})
            print "epoch %d step %d, training batch loss %g" %(e+1, i, train_loss)
        train_step.run(feed_dict={x:train_data[i*batch:(i+1)*batch], y_:train_label[i*batch:(i+1)*batch], keep_prob:0.3})

    print "epoch %d test loss %g" %(e+1, loss.eval(feed_dict = {
        x:test_data, y_:test_label, keep_prob:1.0}))
saver.save(sess,'/home/zwj/Desktop/recommend/movielens/use_data/model/modelxu.ckpt', global_step=e+1)