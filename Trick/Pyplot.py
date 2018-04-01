# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

c = tf.truncated_normal(shape=[2, 2], mean=0, stddev=1)
with tf.Session() as sess:
    m = sess.run(c)
    print m

# plt.plot([1, 2, 3, 4],[1, 4, 9, 16], 'ro')
# plt.plot(m, 'ro')
# plt.ylabel('some numbers')
# plt.axis([0, 6, 0 ,20])
# t = np.arange(0., 5., 0.2)
#
# plt.plot(t ,t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

# def f(t):
#     return np.exp(-t)*np.cos(2*np.pi *t)
#
# t1 = np.arange(0.0, 5.0, 0.1)
# t2 = np.arange(0.0, 5.0, 0.02)
#
# plt.figure("2subplot")
# plt.subplot(211)
# plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
#
# plt.subplot(212)
# plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
# plt.show()

# plt.figure(1)
# plt.subplot(211)
# plt.plot([1, 2, 3])
# plt.subplot(212)
# plt.plot([4, 5, 6])
# plt.show()
#
# plt.figure(2)
# plt.plot([4, 5, 6])
#
# plt.figure(1)
# plt.subplot(111)
# plt.title('Easy as 1, 2, 3')
# plt.show()


# mu, sigma = 100, 15
# t = np.random.randn(10000)
# x = mu + sigma * t
# print t, x
#
#
# n, bins, pathches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)
#
# t =plt.xlabel('Smarts', fontsize=14, color='red')
# plt.setp(t, color='blue')
# plt.ylabel('Probability')
# plt.title('Histogram of IQ')
# plt.title(r'$\sigma_i=15$')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.axis([40, 160, 0, 0.03])
# plt.grid(True)
# plt.show()

# ax = plt.subplot(111)
# t = np.arange(0.0, 5.0, 0.01)
# s = np.cos(2*np.pi*t)
# line, = plt.plot(t, s, lw = 2)
#
# plt.annotate('local max', xy = (2,1), xytext=(3, 1.5),
#              arrowprops=dict(facecolor='black', shrink=0.05))
#
# plt.ylim(-2, 2)
# plt.show()

#np.random.normal为高斯分布
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y  < 1)]
y.sort()
x = np.arange(len(y))

plt.figure(1)

#linear
plt.subplot(221)
plt.plot(x, y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)

#对数
plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid(True)

#symmetric log
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthreshy = 0.05)
plt.title('symlog')
plt.grid(True)

#logit
plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)

plt.show()