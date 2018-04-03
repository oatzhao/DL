# -*- coding: UTF-8 -*-
from datetime import datetime
import math
import time
import tensorflow as tf
#卷积核数量  64 =》192 =》348 =》256 =》256

batch_size = 32
num_batches = 100

def print_active(t):
    print (t.op.name, ' ', t.get_shape().as_list())

def inference(images):
    parameters = []

    #covn1
    with tf.name_scope('conv1')as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64],
                                                 dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding= 'SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name="biases")
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print print_active(conv1)
        parameters += [kernel, biases]
    #LRN1
    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')
    #max pool1
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool1')
    print_active(pool1)


    #covn2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192],
                                                 dtype=tf.float32, stddev=1e-1, name='weights'))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192],
                                         dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_active(conv2)

    #LRN2
    lrn2 = tf.nn.lrn(conv2, 4, bias = 1.0, alpha=0.001/9, beta=0.75, name='lrn2')
    #max pool2
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')
    print_active(pool2)


    #covn3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32, stddev=1e-1, name='weights'))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384],
                                         dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_active(conv3)


    #covn4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32, stddev=1e-1, name='weights'))
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256],
                                         dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_active(conv4)

    #conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32, stddev=1e-1, name='weights'))
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256],
                                         dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_active(conv5)
    #max pool5
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool5')
    print_active(pool5)
    return pool5, parameters

    #fullc1
    #经过5个卷积层，2个池化层之后，feature map变为6*6*256个
    W_fc1 = tf.Variable(tf.truncated_normal([6*6*256  , 4096],
                                            dtype=tf.float32, stddev=1e-1, name='weights'))
    b_fc1 = tf.Variable(tf.constant(0.0, shape=[4096],
                                    dtype=tf.float32), trainable=True, name='biases')
    #将2维的6*6的feature变为一维，进行全连接
    pool5_flat = tf.reshape(pool5, [-1, 6*6*256])
    #池化
    h_fc1 = tf.nn.relu(tf.matmul(pool5_flat, W_fc1) + b_fc1)

    #fullc2
    W_fc2 = tf.Variable(tf.truncated_normal([4096, 4096],
                                            dtype=tf.float32, stddev=1e-1, name='weights'))
    b_fc2 = tf.Variable(tf.constant(0.0, shape=[4096],
                                     dtype=tf.float32), trainable=True, name='biases')
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    #fullc3
    W_fc3 = tf.Variable(tf.truncated_normal([4096, 1000],
                                            dtype=tf.float32, stddev=1e-1, name='weights'))
    b_fc3 = tf.Variable(tf.constant(0.0, shape=[1000],
                                    dtype=tf.float32), trainable=True, name='biases')
    #使用softmax函数进行分类
    h_fc3 = tf.nn.softmax((tf.matmul(h_fc2, W_fc3) + b_fc3))


#实现一个评估AlexNet每一轮计算时间的函数
#这个函数第一个参数是TensorFlow的Session，第二个变量是需要评测的运算算子，第三个变量是测试的名称
#num_steps_burn_in是程序预热轮数，因为开头几轮迭代有显存的加载、cache命中等问题因此可以跳过，只考虑10轮迭代之后的计算时间
#info_string是测试名称
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    durationList = []

    #进行num_batches + num_steps_burn_in迭代计算，使用time.time（）记录时间，每次通过session.run(target)执行
    #在初始热身的num_steps_burn_in次迭代后，每10轮迭代显示当前迭代所需要的时间。每轮将total_duration和total_duration_squared累加
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _= session.run(target)
        duration = time.time() - start_time
        durationList.append(duration)
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %(datetime.now(), i - num_steps_burn_in, duration))
        total_duration += duration
        total_duration_squared += duration * duration

    #计算每轮迭代的平均耗时mn和标准差sd
    mn = total_duration / num_batches

    total = 0.0
    for dr in durationList:
        temp = (dr-mn)*(dr-mn)
        total += temp
    vr = total/num_batches
    #vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f + /- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches, mn, sd))

#主函数
#由于ImageNet数据集太大了，所以自己生成随机图片数据测试前馈、反馈计算的耗时
#tf.random_normal函数构造正太分布的随机tensor，第一个维度是batch_size,即每轮迭代的样本数
#第二个和第三个维度是图片的尺寸为了和ImageNet保持一直使用224，第四个维度是图片的颜色通道数
def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size,3],
                                               dtype=tf.float32,
                                               stddev=1e-1))

        #使用inference函数构建整个AlexNet网络
        pool5, parameters = inference(images)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        #forward测评
        time_tensorflow_run(sess, pool5, "Fowward")

        #Forward-backward测评
        #output = sum(t ** 2) / 2 L2正则
        objective = tf.nn.l2_loss(pool5)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, "Forward-backward")

run_benchmark()


'''
结论：有无LRN层时间相差3倍左右
backward运算时间大概是forward耗时的3倍左右
CNN的训练需要多次迭代，所以目前瓶颈主要还是在训练，迭代问题不大
'''




