# coding=utf-8
import h5py
import numpy as np
import tensorflow as tf

# 开启交互session
sess = tf.InteractiveSession()

# 读取数据
def read_data(file):
    f = h5py.File(file, 'r')
    x_data = f['data'][:]
    # 数据原格式 [num, channel, width, height] 转换维度到tensorflow的格式 [num, height, width, channel]
    x_data = x_data.transpose((0, 3, 2, 1))
    y_data = f['label'][:]
    y_data = np.reshape(y_data, (-1, 1))
    return x_data, y_data

def shuffle_data(data, label):
    index = [i for i in range(len(data))]
    np.random.shuffle(index)
    data = data[index]
    label = label[index]
    return data, label

# 定义初始化函数等
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

    param:
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

# 搭建网络-----------------------------------------------------------------------begin
# shape=[num, height, width, channel]
x = tf.placeholder("float", shape = [None, 12, 12, 5])
y_ = tf.placeholder("float" , shape = [None, 1])
is_training = tf.placeholder(dtype=tf.bool)

# layer 1
# filter_size = [height, width, in_channels, out_channels]
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

# layer 3
W_conv3 = weight_variable([2, 2, 64, 128])
bias_conv3 = bias_variable([128])

# h_conv3 = tf.nn.relu(batch_norm_layer(conv2d(h_pool2,W_conv3) + bias_conv3,is_training=is_training))
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + bias_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# layer full connection
W_fc1 = weight_variable([2 * 2 * 128, 512])
bias_fc1 = bias_variable([512])

h_pool3_flat = tf.reshape(h_pool3, [-1, 2 * 2 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + bias_fc1)

# drop
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output y
W_fc2 = weight_variable([512, 1])
bias_fc2 = bias_variable([1])

y_conv = tf.matmul(h_fc1_drop, W_fc2)  + bias_fc2
# 搭建网络-----------------------------------------------------------------------end

# 计算误差
loss = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_conv)))
# 定义优化器
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
# 初始化变量
sess.run(tf.global_variables_initializer())

# 读取数据
train_file_name = '/home/zwj/Desktop/recommend/movielens/use_data/u1_train_features5.hd5'
test_file_name = '/home/zwj/Desktop/recommend/movielens/use_data/u1_test_features5.hd5'

x_train, y_train = read_data(train_file_name)
x_test, y_test = read_data(test_file_name)

# 设置训练迭代次数
epochs = 10

# 开启模型保存或者读取
saver = tf.train.Saver(max_to_keep=3)
# 加载已训练的模型，初始化网络参数
saver.restore(sess, "/home/zwj/Desktop/recommend/movielens/use_data/model/cnn_mode.ckpt-9")

# 输出初始网络预测结果的误差
print "original test loss %g" %(loss.eval(feed_dict = {
    x:x_test, y_:y_test, keep_prob:1.0}))

# 训练与预测
for e in range(epochs):
    x_train, y_train = shuffle_data(x_train, y_train)
    for i in range(869): #1739
        batch = 64
        if i%100 == 0:
            train_loss = loss.eval(feed_dict={
                x:x_train[i*batch:(i+1)*batch], y_:y_train[i*batch:(i+1)*batch], keep_prob:1.0})
            print "epoch %d step %d, training batch loss %g" %(e+1, i, train_loss)
        train_step.run(feed_dict={x:x_train[i*batch:(i+1)*batch], y_:y_train[i*batch:(i+1)*batch], keep_prob:0.3})

    print "epoch %d test loss %g" %(e+1, loss.eval(feed_dict = {
        x:x_test, y_:y_test, keep_prob:1.0}))
    # 保存模型
    saver.save(sess,'/home/zwj/Desktop/recommend/movielens/use_data/model/cnn_mode_xu.ckpt', global_step=e+1)

# 输出标签和预测结果
# output = y_conv.eval(feed_dict={
#    x:x_test, y_:y_test, keep_prob:1.0})
# label = y_.eval(feed_dict={
#    x:x_test, y_:y_test, keep_prob:1.0})
# print('notice output:', output.shape)
# print('notice label:', label.shape)
# np.savetxt('/home/zwj/Desktop/recommend/movielens/use_data/y_conv_gensim.txt', output, fmt='%.2f')
# np.savetxt('/home/zwj/Desktop/recommend/movielens/use_data/label_gensim.txt', label, fmt='%.2f')
