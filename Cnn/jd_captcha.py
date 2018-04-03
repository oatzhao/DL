# -*- coding: UTF-8 -*-
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf
import random

im = Image.open('zztr.png')
im = im.convert('L')
im.show()

IMAGE_HEIGHT = 35
IMAGE_WIDTH = 65
MAX_CAPTCHA = 4
CHAR_SET_LEN = 26

def get_name_and_image():
    all_image = os.listdir('/Users/zhaoyan20/Downloads/JD_LOGIN/')
    random_file = random.randint(0, 5592)
    base = os.path.basename('/Users/zhaoyan20/Downloads/JD_LOGIN/' + all_image[random_file])
    #图片名称
    name = os.path.splitext(base)[0]
    #将图片转换为数组
    image = Image.open('/Users/zhaoyan20/Downloads/JD_LOGIN/' + all_image[random_file])
    image = np.array(image)
    return name, image

#eg:['A']=>[00000000000001000000000000]
def name2vec(name):
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    for i, c in enumerate(name):
        idx = i * 26 + ord(c) - 97
        vector[idx] = 1
    return vector

def vec2name(vec):
    name = []
    for i in vec:
        a = chr(i + 97)
        name.append(a)
    return "".join(name)

#生成一个训练batch
def get_next_batch(batch_size = 64):
    batch_x = np.zeros([batch_size, IMAGE_WIDTH * IMAGE_HEIGHT * 3])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    for i in range(batch_size):
        name, image = get_name_and_image()
        batch_x[i, :] = 1 * (image.flatten())
        batch_y[i, :] = name2vec(name)
    return batch_x, batch_y
#get_next_batch(1)

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH *3])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)

#3个卷积层一个全连接层
def captch_cnn(w_alpha = 0.01, b_alpha = 0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    #3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 3, 32],  stddev=0.01))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)



# Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([5 * 9 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out



# 训练
def train_crack_captcha_cnn():
    output = captch_cnn()
    #交叉熵损失函数
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    #调整学习步长
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        for i in range(200):
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
            print(step, loss_)

            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc)
                saver.save(sess, "./crack_capcha.model", global_step=step)
                # 如果准确率大于50%,保存模型,完成训练
                # if acc > 0.5:
                #     saver.save(sess, "./crack_capcha.model", global_step=step)
                #     break

            step += 1

train_crack_captcha_cnn()

#预测
# def crack_captcha():
#     output = crack_captcha_cnn()
#
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         saver.restore(sess, tf.train.latest_checkpoint('.'))
#         n = 1
#         while n <= 10:
#             text, image = get_name_and_image()
#             image = 1 * (image.flatten())
#             predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
#             text_list = sess.run(predict, feed_dict={X: [image], keep_prob: 1})
#             vec = text_list[0].tolist()
#             predict_text = vec2name(vec)
#             print("正确: {}  预测: {}".format(text, predict_text))
#             n += 1
#
# crack_captcha()





























