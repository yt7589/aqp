import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ann.linear_regression import LinearRegression

class LinearRegressionEngine(object):
    def __init__(self):
        self.name = 'LinearRegressionEngine'

    def startup(self):
        # 生成测试数据
        self.generate_ds()
        self.run_analysis()

    def generate_ds(self):
        self.num_samples, self.w, self.b = 20, 0.5, 2
        self.xs = np.asarray(range(self.num_samples))
        self.ys = np.asarray([
            x * self.w + self.b + np.random.normal() for x in range(self.num_samples)
        ])

    def run_analysis(self):
        xtf = tf.placeholder(tf.float32, [self.num_samples], 'xs')
        ytf = tf.placeholder(tf.float32, [self.num_samples], 'ys')
        model = LinearRegression()
        model_output = model(xtf)
        cov = tf.reduce_sum( (xtf - tf.reduce_mean(xtf)) * (ytf - tf.reduce_mean(ytf)) )
        var = tf.reduce_sum(tf.square(xtf - tf.reduce_mean(xtf)))
        w_hat = cov / var
        b_hat = tf.reduce_mean(ytf) - w_hat*tf.reduce_mean(xtf)
        solve_w = model.w.assign(w_hat)
        solve_b = model.b.assign(tf.reduce_mean(ytf) - w_hat*tf.reduce_mean(xtf))
        with tf.train.MonitoredSession() as sess:
            sess.run([solve_w, solve_b], feed_dict={xtf: self.xs, ytf: self.ys})
            preds = sess.run(model_output, feed_dict={xtf: self.xs, ytf: self.ys})
        plt.scatter(self.xs, self.ys)
        plt.plot(self.xs, preds)
        plt.show()