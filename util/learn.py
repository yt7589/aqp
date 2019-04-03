import numpy as np
#import tensorflow as tf

class Learn(object):
    def __init__(self):
        self.name = 'Learn'

'''
    def startup(self):
        #self.test_cond()
        #self.while_loop()
        self.dynamic_unrolling()

    def dependency_control(self):
        x = tf.get_variable('x', shape=(), initializer=tf.zeros_initializer())
        assign_x = tf.assign(x, 10.0)
        with tf.control_dependencies([assign_x]):
            z = x + 1
        with tf.train.MonitoredSession() as sess:
            print(sess.run(z))

    def test_cond(self):
        v1 = tf.get_variable('v1', shape=(), initializer = tf.zeros_initializer())
        v2 = tf.get_variable('v2', shape=(), initializer = tf.zeros_initializer())
        rst = tf.placeholder(tf.bool)
        cond1 = tf.cond(rst, 
            lambda: tf.assign(v1, 1.0),
            lambda: tf.assign(v2, 2.0)
        )
        with tf.train.MonitoredSession() as sess:
            #cond1_v, v1_v, v2_v =sess.run([cond1, v1, v2], feed_dict={rst: False})
            #print('False: v1={0}, v2={1}, cond1={2}'.format(v1_v, v2_v, cond1_v))
            cond1_v, v1_v, v2_v =sess.run([cond1, v1, v2], feed_dict={rst: True})
            print('True: v1={0}, v2={1}, cond1={2}'.format(v1_v, v2_v, cond1_v))

    def while_loop(self):
        k = tf.constant(2)
        matrix = tf.ones([2, 2])
        condition = lambda i, _: i<k
        body = lambda i, m: (i+1, tf.matmul(m, matrix))
        final_i, power = tf.while_loop(
            cond = condition,
            body = body,
            loop_vars = (0, tf.diag([1.0, 1.0]))
        )
        with tf.train.MonitoredSession() as sess:
            i1, p1 = sess.run([final_i, power], feed_dict={})
            print('i1={0}, p1={1}'.format(i1, p1))

    def dynamic_unrolling(self):
        #x = tf.reshape(tf.range(10), [10, 1, 1])
        x = tf.zeros((10, 1, 1))
        fib_seq = tf.nn.dynamic_rnn(
            cell=FibonacciCell(),
            inputs=x,
            dtype=tf.float32,
            time_major=True
        )
        with tf.train.MonitoredSession() as sess:
            fc, x_v = sess.run([fib_seq, x])
        print(fc)
        print('x={0}'.format(x_v))

    def run_gpu(self):
        with tf.device('/cpu:0'):
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        with tf.device('/gpu:0'):
            c = tf.matmul(a, b)

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            print(sess.run(c))



class FibonacciCell(object):
    def __init__(self):
        self.output_size = 1
        self.state_size = tf.TensorShape([1, 1])

    def __call__(self, input, state):
        return state[0] + state[1], (state[1], state[0] + state[1])

    def zero_state(self, batch_size, dtype):
        return (tf.zeros((batch_size, 1), dtype=dtype),
            tf.ones((batch_size, 1), dtype=dtype)
        )
    
    def initial_state(self, batch_size, dtype):
        return self.zero_state(batch_size, dtype)
        
        
'''