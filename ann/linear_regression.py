import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt

class LinearRegression(object):
    def __init__(self):
        '''
        self.w = tf.get_variable('w', 
                    dtype=tf.float32, shape=[], 
                    initializer=tf.zeros_initializer()
        )
        self.b = tf.get_variable(
            'b', dtype=tf.float32, shape=[],
            initializer=tf.zeros_initializer()
        )
        '''

    def __call__(self, x):
        return self.w * x + self.b

        

    
        