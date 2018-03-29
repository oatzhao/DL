# -*- coding: UTF-8 -*-
import tensorflow as tf

#https://www.zhihu.com/question/41252833 交叉熵  交叉熵损失函数


# （1）定义算法公式，也就是神经网络的forward时的计算
# （2）定义loss，选定优化器，并指定优化器优化loss
# （3）迭代的对数据进行训练
# （4）再测试集或验证集上对准确率进行评测



def load():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("data/", one_hot=True)

    print (mnist.train.images.shape)
    # print (mnist.test.images.shape, mnist.test.labels.shape)
    # print (mnist.validation.images.shape, mnist.validation.labels.shape)

    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, 784])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x ,w) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    tf.global_variables_initializer().run()
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        train_step.run({x:batch_xs, y_:batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print (accuracy.eval({x:mnist.test.images, y_:mnist.test.labels}))
if __name__ == '__main__':
    load()
