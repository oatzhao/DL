# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
#https://www.cnblogs.com/huangshiyu13/p/6721805.html

# state = tf.Variable(0, name='counter')
# one = tf.constant(1)
#
# new_value = tf.add(state, one)
# update = tf.assign(state, new_value)
#

#
# with tf.Session() as sess:
#     sess.run(init)
#     for _ in range(3):
#         sess.run(update)
#         print sess.run(state)

#
# w = [[1,2,3],[4,5,6],[1,2,3]]
# w_v = tf.Variable(w)
# x = [[2,3,2]]
# x_v = tf.Variable(x)
# y = tf.matmul(x, w)
# sess = tf.Session()
# sess.run(init)
# print sess.run(y)


#1、通过Session.run()获取变量值
# x = tf.placeholder(tf.float32)
# y = tf.placeholder(tf.float32)
# bias = tf.Variable(1.0)
#
# y_pred = x ** 2 + bias
# loss = (y - y_pred) ** 2
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
# print ('Loss(x, y) = %.3f' % sess.run(loss, feed_dict ={x: 3.0, y:9.0}))
# print ('pred_y(x) = %.3f' % sess.run(y_pred, {x: 3.0}))
# print ('bias = %.3f' % sess.run(bias))

def my_func(x):
    return np.sinh(x)

inp = tf.placeholder(tf.float32, [3])
y = tf.py_func(my_func, [inp], [tf.float32])

#https://github.com/ericjang/tdb


