import numpy as np
import h5py
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

# mnist = input_data.read_data_sets('mnist/', one_hot = True)

sess = tf.InteractiveSession()

x = tf.placeholder("float", shape = [None, 12, 12, 5])
y_ = tf.placeholder("float" , shape = [None, 1])

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


# set network-----------------------------------------------------------------------

# layer 1
W_conv1 = weight_variable([3, 3, 5, 32])
bias_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + bias_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# layer 2
W_conv2 = weight_variable([3, 3, 32, 64])
bias_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + bias_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# layer 3 full connection

W_fc1 = weight_variable([3 * 3 * 64, 256])
bias_fc1 = bias_variable([256])

h_pool2_flat = tf.reshape(h_pool2, [-1, 3 * 3 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + bias_fc1)

# drop
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output y

W_fc2 = weight_variable([256, 1])
bias_fc2 = bias_variable([1])

# y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)  + bias_fc2)
y_conv = tf.matmul(h_fc1_drop, W_fc2)  + bias_fc2
# network --------------------------------------------------------------------------------------------------end
# mse = tf.reduce_mean(tf.squared_difference(out, Y))
# cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
loss = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_conv)))

train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())

def read_data(file):
    f = h5py.File(file, 'r')
    x_data = f['data'][:]
    x_data = x_data.transpose((0, 3, 2, 1))
    y_data = f['label'][:]
    y_data = np.reshape(y_data, (-1, 1))
    return x_data, y_data

train_file_name = '/home/zwj/Desktop/recommend/movielens/use_data/u1_train_features5.hd5'
test_file_name = '/home/zwj/Desktop/recommend/movielens/use_data/u1_test_features5.hd5'

x_train, y_train = read_data(train_file_name)
x_test, y_test = read_data(test_file_name)
epochs = 10

# saver = tf.train.Saver(max_to_keep=3)
# # train samples:55648
# saver.restore(sess, "/home/zwj/Desktop/recommend/movielens/use_data/model/model.ckpt-10")
print "original test loss %g" %(loss.eval(feed_dict = {
    x:x_test, y_:y_test, keep_prob:1.0}))
for e in range(epochs):
    for i in range(869): #1739
        batch = 64
        if i%100 == 0:
            train_loss = loss.eval(feed_dict={
                x:x_train[i*batch:(i+1)*batch], y_:y_train[i*batch:(i+1)*batch], keep_prob:1.0})
            print "epoch %d step %d, training batch loss %g" %(e+1, i, train_loss)
        train_step.run(feed_dict={x:x_train[i*batch:(i+1)*batch], y_:y_train[i*batch:(i+1)*batch], keep_prob:0.5})

    print "epoch %d test loss %g" %(e+1, loss.eval(feed_dict = {
        x:x_test, y_:y_test, keep_prob:1.0}))
#     saver.save(sess,'/home/zwj/Desktop/recommend/movielens/use_data/model/model2.ckpt', global_step=e+1)
# output = y_conv.eval(feed_dict={
#    x:x_test, y_:y_test, keep_prob:1.0})
# label = y_.eval(feed_dict={
#    x:x_test, y_:y_test, keep_prob:1.0})
# print('notice output:', output.shape)
# print('notice label:', label.shape)
# np.savetxt('/home/zwj/Desktop/recommend/movielens/use_data/y_conv_gensim.txt', output, fmt='%.2f')
# np.savetxt('/home/zwj/Desktop/recommend/movielens/use_data/label_gensim.txt', label, fmt='%.2f')
