import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class LinearRegression(object):
    def __init__(self):
        tf.reset_default_graph()
        self.w = tf.get_variable('w', 
                    dtype=tf.float32, shape=[], 
                    initializer=tf.zeros_initializer()
        )
        self.b = tf.get_variable(
            'b', dtype=tf.float32, shape=[],
            initializer=tf.zeros_initializer()
        )

    def __call__(self, x):
        return self.w * x + self.b

    def startup(self):
        # 生成测试数据
        self.generate_ds()
        self.run_analysis()

    def generate_ds(self):
        self.num_samples, w, b = 20, 0.5, 2
        self.xs = np.asarray(range(self.num_samples))
        self.ys = np.asarray([
            x * w + b + np.random.normal() for x in range(self.num_samples)
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
        print('step 1')
        with tf.train.MonitoredSession() as sess:
            print('step 2')
            sess.run(tf.global_variables_initializer())
            print('step 3')
            sess.run([solve_w, solve_b], feed_dict={xtf: self.xs, ytf: self.ys})
            print('step 4')
            preds = sess.run(model_output, feed_dict={xtf: self.xs, ytf: self.ys})
            print('step 5')
        plt.scatter(self.xs, self.ys)
        print('step 6')
        plt.plot(self.xs, preds)
        print('step 7')
        plt.show()
        