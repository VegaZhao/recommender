# coding=utf-8
import numpy as np
import h5py
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from sklearn.metrics import mean_squared_error
import tensorflow as tf


# mnist = input_data.read_data_sets('mnist/', one_hot = True)

sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def batch_norm_layer(value,is_training=False,name='batch_norm'):
    '''
    批量归一化  返回批量归一化的结果

    args:
        value:代表输入，第一个维度为batch_size
        is_training:当它为True，代表是训练过程，这时会不断更新样本集的均值与方差。当测试时，要设置成False，这样就会使用训练样本集的均值和方差。
              默认测试模式
        name：名称。
    '''
    if is_training is True:
        #训练模式 使用指数加权函数不断更新均值和方差
        return tf.contrib.layers.batch_norm(inputs=value,decay=0.9,updates_collections=None,is_training = True)
    else:
        #测试模式 不更新均值和方差，直接使用
        return tf.contrib.layers.batch_norm(inputs=value,decay=0.9,updates_collections=None,is_training = False)
# set network-----------------------------------------------------------------------
x = tf.placeholder("float", shape = [None, 12, 12, 5])
y_ = tf.placeholder("float" , shape = [None, 5])
is_training = tf.placeholder(dtype=tf.bool)

# layer 1
W_conv1 = weight_variable([3, 3, 5, 32])
bias_conv1 = bias_variable([32])

# h_conv1 = tf.nn.relu(batch_norm_layer(conv2d(x,W_conv1) + bias_conv1,is_training=is_training))
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + bias_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# layer 2
W_conv2 = weight_variable([3, 3, 32, 64])
bias_conv2 = bias_variable([64])

# h_conv2 = tf.nn.relu(batch_norm_layer(conv2d(h_pool1,W_conv2) + bias_conv2,is_training=is_training))
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + bias_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# layer full connection
W_fc1 = weight_variable([3 * 3 * 64, 256])
bias_fc1 = bias_variable([256])

h_pool2_flat = tf.reshape(h_pool2, [-1, 3 * 3 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + bias_fc1)

# drop
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output y

W_fc2 = weight_variable([256, 5])
bias_fc2 = bias_variable([5])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)  + bias_fc2)
# y_conv = tf.matmul(h_fc1_drop, W_fc2)  + bias_fc2
# network --------------------------------------------------------------------------------------------------end

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
# AdamOptimizer
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())

# cy_ = tf.argmax(y_, 1).eval()
# cy_conv = tf.argmax(y_conv, 1).eval()
#
# loss = tf.sqrt(tf.reduce_mean(tf.square(cy_ - cy_conv)))



def read_data(file):
    f = h5py.File(file, 'r')
    x_data = f['data'][:]
    x_data = x_data.transpose((0, 3, 2, 1))
    y_data = f['label'][:]
    # y_data = np.reshape(y_data, (-1, 1))
    return x_data, y_data

train_file_name = '/home/zwj/Desktop/recommend/movielens/use_data/u1_train_features5_gensim256.hd5'
test_file_name = '/home/zwj/Desktop/recommend/movielens/use_data/u1_test_features5_gensim256.hd5'

x_train, y_train = read_data(train_file_name)
x_test, y_test = read_data(test_file_name)

y_train = y_train - 1
y_test = y_test - 1

y_train_oh = tf.one_hot(y_train, 5).eval()
y_test_oh = tf.one_hot(y_test, 5).eval()

epochs = 20

saver = tf.train.Saver(max_to_keep=1)
# # train samples:55648
# saver.restore(sess, "/home/zwj/Desktop/recommend/movielens/use_data/model/model520.ckpt-10")
print "original test loss %g" %(cross_entropy.eval(feed_dict = {
    x:x_test, y_:y_test_oh, keep_prob:1.0}))
for e in range(epochs):
    for i in range(869): #1739
        batch = 64
        if i%100 == 0:
            train_loss = cross_entropy.eval(feed_dict={
                x:x_train[i*batch:(i+1)*batch], y_:y_train_oh[i*batch:(i+1)*batch], keep_prob:1.0})
            print "epoch %d step %d, training batch loss %g" %(e+1, i, train_loss)
        train_step.run(feed_dict={x:x_train[i*batch:(i+1)*batch], y_:y_train_oh[i*batch:(i+1)*batch], keep_prob:0.3})

    print "epoch %d test loss %g" %(e+1, cross_entropy.eval(feed_dict = {
        x:x_test, y_:y_test_oh, keep_prob:1.0}))
    saver.save(sess,'/home/zwj/Desktop/recommend/movielens/use_data/model/model.ckpt', global_step=e+1)
    output_onehot = y_conv.eval(feed_dict={
       x:x_test, y_:y_test_oh, keep_prob:1.0})
    label_onehot = y_.eval(feed_dict={
       x:x_test, y_:y_test_oh, keep_prob:1.0})

    output = tf.argmax(output_onehot, 1).eval() + 1
    label = tf.argmax(label_onehot, 1).eval() + 1
    print('notice output:', output.shape)
    print('notice label:', label.shape)

    rmse = np.sqrt(mean_squared_error(output, label))
    print('\n\nTesting Result With Cosine User-User Similarity: {:.4f} RMSE'.format(rmse))
np.savetxt('/home/zwj/Desktop/recommend/movielens/use_data/y_conv_c.txt', output, fmt='%.2f')
np.savetxt('/home/zwj/Desktop/recommend/movielens/use_data/label_c.txt', label, fmt='%.2f')
