# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)
sess = tf.InteractiveSession()

#权重加噪音，避免完全对称，比如截断的正太分布噪音，标准差设为0.1
#tf.truncated_normal（）截断正太分布的值 shape表示生成张量的维度，mean是均值，stddev是标准差
#如果正太分布的值如果与均值差大于两倍的标准差，那就重新生成
def weigth_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#偏置加一些小的正值，避免死亡节点
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#卷积层和池化层复用
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

#因为要利用空间结构信心，将1D的输入向量转为2D的图片结构，1X784 =》 28X28
#-1表示样本数量不固定，1代表样本通道
x_image = tf.reshape(x, [-1, 28, 28, 1])

#定义第一个卷积层
#5X5卷积核的大小，1通道，32个不同的卷积核
W_conv1 = weigth_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#定义第二个卷积层
#提取64种特征
W_conv2 = weigth_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#经过两次2X2的最大池化，向量变为7X7，第二个卷积核数量为64，其输出的tensor尺寸即为7X7X64
#使用tf.reshape函数对第二个卷积层的输出tensor进行变形  2D = 》 1D
W_fc1 = weigth_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weigth_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
correct_prediction = tf.Print(y_conv, [tf.argmax(y_conv,1)], "argmax(y)=")
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()
#for i in range(20000):
for i in range(1):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {x:batch[0], y_:batch[1],
                                                    keep_prob:1.0})
        print ("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

print ("test accuracy %g"%accuracy.eval(feed_dict = {x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))